"""
LSTM 3-Class Triple-Barrier Model
Binance 5-min Bars | PyTorch Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class BarSequenceDataset(Dataset):
    """
    X : np.ndarray [n_samples, seq_len, n_features]  — z-score normalised
    y : np.ndarray [n_samples]  — labels in {0, 1, 2}
            0 = down barrier hit  (-1 original)
            1 = no hit / timeout  ( 0 original)
            2 = up   barrier hit  (+1 original)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class TripleBarrierLSTM(nn.Module):
    """
    Exact layer order:

        Input [batch, seq=20, features=N]
            │
            ├── Input Dropout (p=0.10)          # applied across feature dim per timestep
            │
            ├── LSTM Layer 1                    # hidden=128, return_sequences=True
            │   └── Recurrent Dropout (p=0.20)  # dropout on h_t between layers (not within)
            │
            ├── LSTM Layer 2                    # hidden=64,  return_sequences=False
            │   └── Recurrent Dropout (p=0.20)  # same — applied via dropout arg in nn.LSTM
            │
            ├── BatchNorm1d(64)                 # normalise the final hidden state vector
            │
            ├── Dense(64 → 32) + ReLU
            │
            ├── Dropout (p=0.20)                # applied AFTER activation, BEFORE output
            │
            └── Dense(32 → 3) + Softmax         # P(dn), P(no_hit), P(up)

    Note on PyTorch recurrent dropout:
        nn.LSTM dropout= applies dropout on outputs between stacked layers only,
        NOT on recurrent connections within a single layer.
        For true recurrent (Gal & Ghahramani) dropout you need a custom cell.
        The standard nn.LSTM dropout= is used here — practical and sufficient.
    """

    def __init__(
        self,
        n_features:       int   = 25,
        seq_len:          int   = 20,
        lstm1_hidden:     int   = 128,
        lstm2_hidden:     int   = 64,
        dense_hidden:     int   = 32,
        input_dropout:    float = 0.10,
        recurrent_dropout:float = 0.20,
        dense_dropout:    float = 0.20,
        n_classes:        int   = 3,
    ):
        super().__init__()

        self.n_features    = n_features
        self.seq_len       = seq_len
        self.lstm2_hidden  = lstm2_hidden

        # ── input dropout ────────────────────────────────────────────────────
        # nn.Dropout applied to [batch, seq, features] — zeros full feature
        # values at random timesteps
        self.input_dropout = nn.Dropout(p=input_dropout)

        # ── LSTM stack ───────────────────────────────────────────────────────
        # dropout= in nn.LSTM applies between layer 1 output → layer 2 input.
        # num_layers=2 with dropout= is equivalent to:
        #   LSTM1(return_seq=True) → Dropout → LSTM2(return_seq=False)
        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = lstm1_hidden,
            num_layers    = 2,
            batch_first   = True,       # [batch, seq, features]
            dropout       = recurrent_dropout,
            bidirectional = False,
        )

        # projection from lstm1_hidden (128) → lstm2_hidden (64)
        # nn.LSTM with num_layers=2 outputs hidden_size of the LAST layer.
        # Since both layers share hidden_size, we add a linear projection
        # to get the 128 → 64 taper explicitly.
        self.hidden_proj = nn.Linear(lstm1_hidden, lstm2_hidden)

        # ── batch normalisation ───────────────────────────────────────────────
        # applied to the final context vector [batch, lstm2_hidden]
        self.batch_norm = nn.BatchNorm1d(lstm2_hidden)

        # ── classification head ───────────────────────────────────────────────
        self.dense = nn.Sequential(
            nn.Linear(lstm2_hidden, dense_hidden),
            nn.ReLU(),
            nn.Dropout(p=dense_dropout),
            nn.Linear(dense_hidden, n_classes),
        )

        # softmax for inference — not needed during training (CrossEntropyLoss
        # applies log-softmax internally)
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for LSTM gates; zero bias."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)   # orthogonal init for recurrent weights
            elif "bias" in name:
                param.data.fill_(0)
                # set forget gate bias to 1 — helps with long sequences
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)

        for layer in self.dense:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, return_logits: bool = True):
        """
        Args:
            x             : [batch, seq_len, n_features]
            return_logits : True during training (CrossEntropyLoss expects logits)
                            False during inference (returns probabilities)
        Returns:
            [batch, 3]  — logits or probabilities
        """
        # ── input dropout ────────────────────────────────────────────────────
        x = self.input_dropout(x)                       # [batch, seq, features]

        # ── LSTM ─────────────────────────────────────────────────────────────
        out, (h_n, _) = self.lstm(x)                    # out: [batch, seq, 128]
                                                        # h_n: [2, batch, 128]

        # take the final hidden state from the last LSTM layer
        context = h_n[-1]                               # [batch, 128]

        # project 128 → 64 (taper)
        context = self.hidden_proj(context)             # [batch, 64]

        # ── batch norm ───────────────────────────────────────────────────────
        context = self.batch_norm(context)              # [batch, 64]

        # ── classification head ───────────────────────────────────────────────
        logits = self.dense(context)                    # [batch, 3]

        if return_logits:
            return logits
        return self.softmax(logits)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model:       nn.Module,
        device:      torch.device,
        lr:          float = 1e-3,
        patience:    int   = 10,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.patience = patience

        # standard cross-entropy — no weighting needed (balanced distribution)
        self.criterion = nn.CrossEntropyLoss()

        self.optimiser = optim.Adam(model.parameters(), lr=lr)

        # halve LR when val_loss plateaus for 5 epochs
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.best_val_loss  = float("inf")
        self.best_weights   = None
        self.epochs_no_improve = 0

    def _run_epoch(self, loader: DataLoader, training: bool):
        self.model.train(training)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(training):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch, return_logits=True)
                loss   = self.criterion(logits, y_batch)

                if training:
                    self.optimiser.zero_grad()
                    loss.backward()
                    # gradient clipping — important for LSTM stability
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimiser.step()

                total_loss += loss.item() * len(y_batch)
                preds       = logits.argmax(dim=-1)
                correct    += (preds == y_batch).sum().item()
                total      += len(y_batch)

        return total_loss / total, correct / total

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, max_epochs: int = 100):
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(1, max_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   training=False)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f} | "
                f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
            )

            # ── early stopping ────────────────────────────────────────────────
            if val_loss < self.best_val_loss:
                self.best_val_loss    = val_loss
                self.best_weights     = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        # restore best weights
        if self.best_weights:
            self.model.load_state_dict(self.best_weights)
            print(f"Restored best weights (val_loss = {self.best_val_loss:.4f})")

        return history


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    """
    Label mapping (matches dataset convention):
        0 = down barrier hit  (-1 original)
        1 = no hit / timeout  ( 0 original)
        2 = up   barrier hit  (+1 original)
    """
    CLASS_NAMES = {0: "DOWN", 1: "NO_HIT", 2: "UP"}

    def __init__(self, model: nn.Module, device: torch.device, conviction_threshold: float = 0.20):
        self.model     = model.to(device)
        self.device    = device
        self.threshold = conviction_threshold

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X : [n_samples, seq_len, n_features]  — already z-score normalised
        Returns:
            probs : [n_samples, 3]  — P(dn), P(no_hit), P(up)
        """
        self.model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        probs  = self.model(tensor, return_logits=False)
        return probs.cpu().numpy()

    def predict_signal(self, X: np.ndarray) -> dict:
        """
        Returns trading signal using conviction margin logic:
            conviction_up = P(up) − max(P(no_hit), P(dn))
            conviction_dn = P(dn) − max(P(no_hit), P(up))

        Only fires if conviction > threshold. Naturally abstains when uncertain.
        """
        probs = self.predict_proba(X)            # [n, 3]
        p_dn, p_no_hit, p_up = probs[:, 0], probs[:, 1], probs[:, 2]

        conviction_up = p_up     - np.maximum(p_no_hit, p_dn)
        conviction_dn = p_dn     - np.maximum(p_no_hit, p_up)

        signals = np.full(len(X), "NO_TRADE", dtype=object)
        signals[conviction_up > self.threshold] = "LONG"
        signals[conviction_dn > self.threshold] = "SHORT"

        # if both fire (shouldn't happen, but guard against it)
        both = (conviction_up > self.threshold) & (conviction_dn > self.threshold)
        signals[both] = "NO_TRADE"

        return {
            "signal":        signals,
            "conviction_up": conviction_up,
            "conviction_dn": conviction_dn,
            "p_up":          p_up,
            "p_no_hit":      p_no_hit,
            "p_dn":          p_dn,
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── dummy data ────────────────────────────────────────────────────────────
    N_SAMPLES  = 1024
    SEQ_LEN    = 20
    N_FEATURES = 25   # Groups I+J+K+L+M

    X_dummy = np.random.randn(N_SAMPLES, SEQ_LEN, N_FEATURES).astype(np.float32)
    y_dummy = np.random.randint(0, 3, size=N_SAMPLES)

    split    = int(0.8 * N_SAMPLES)
    train_ds = BarSequenceDataset(X_dummy[:split], y_dummy[:split])
    val_ds   = BarSequenceDataset(X_dummy[split:], y_dummy[split:])

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=False)  # shuffle=False preserves temporal order
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = TripleBarrierLSTM(
        n_features        = N_FEATURES,
        seq_len           = SEQ_LEN,
        lstm1_hidden      = 128,
        lstm2_hidden      = 64,
        dense_hidden      = 32,
        input_dropout     = 0.10,
        recurrent_dropout = 0.20,
        dense_dropout     = 0.20,
        n_classes         = 3,
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── training ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, device, lr=1e-3, patience=10)
    history = trainer.fit(train_loader, val_loader, max_epochs=5)   # 5 epochs for smoke test

    # ── inference ─────────────────────────────────────────────────────────────
    predictor = Predictor(model, device, conviction_threshold=0.20)
    result    = predictor.predict_signal(X_dummy[:8])

    print("\n── Sample Inference ─────────────────────────────────────────────")
    for i in range(8):
        print(
            f"  [{i}] signal={result['signal'][i]:<8}  "
            f"P(up)={result['p_up'][i]:.3f}  "
            f"P(no_hit)={result['p_no_hit'][i]:.3f}  "
            f"P(dn)={result['p_dn'][i]:.3f}  "
            f"conv_up={result['conviction_up'][i]:+.3f}  "
            f"conv_dn={result['conviction_dn'][i]:+.3f}"
        )
