 """
LSTM 3-Class Triple-Barrier Model
Binance 5-min Bars | PyTorch Implementation
XII architecture-BASE
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # add focal loss
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
# from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN    = 20

HIDDEN_NODES = 128 # upgrade
DROPOUT = 0.20 

BATCH_SIZE = 512
MAX_EPOCHS = 50 # 20-30 for comparison
LR         = 1e-3
PATIENCE   = 15

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "mlData"
TRAIN_PATH = DATA_DIR / "202603-vX-train.jsonl"
VAL_PATH   = DATA_DIR / "202603-vX-val.jsonl"
TEST_PATH  = DATA_DIR / "202603-vX-test.jsonl"

MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}   # original → PyTorch class index
CLASS_NAMES = ["DOWN", "NO_HIT", "UP"]


def load_jsonl(path: Path, feature_cols: list) -> tuple:
    """
    Load JSONL file and return (X, y) arrays.
    Skips rows with missing/null labels.
    Label mapping: -1→0 (DOWN), 0→1 (NO_HIT), +1→2 (UP)
    """
    X_rows, y_rows = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            lbl = row.get("label")
            if lbl is None:
                continue
            X_rows.append([row[c] for c in feature_cols])
            y_rows.append(LABEL_MAP[int(lbl)])
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


class BarSequenceDataset(Dataset):
    """
    Sliding-window dataset for temporal bar sequences.
    Window X[idx : idx+seq_len] → label at the last bar of that window.
    Temporal order is always preserved — never shuffle.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]
        # return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]


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
            │   └── Recurrent Dropout (p=0.20)+  # dropout on h_t between layers (not within)
            │
            ├── LSTM Layer 2                    # hidden=128, return_sequences=False
            │   └── Recurrent Dropout (p=0.20)+  # same — applied via dropout arg in nn.LSTM
            │
            ├── Linear(128 → 128)              # projection on h_n[-1] — no compression
            │
            ├── BatchNorm1d(128)               # normalise the final hidden state vector
            │
            ├── Dense(128 → 64) + ReLU
            │
            ├── Dropout (p=0.20)+                # applied AFTER activation, BEFORE output
            │
            └── Dense(64 → 3) + Softmax         # P(dn), P(no_hit), P(up)

    Note on PyTorch recurrent dropout:
        nn.LSTM dropout= applies dropout on outputs between stacked layers only,
        NOT on recurrent connections within a single layer.
        For true recurrent (Gal & Ghahramani) dropout you need a custom cell.
        The standard nn.LSTM dropout= is used here — practical and sufficient.
    """

    def __init__(
        self,
        n_features:        int,
        seq_len:           int   = SEQ_LEN,
        lstm1_hidden:      int   = HIDDEN_NODES,
        lstm2_hidden:      int   = 64, # 64 and 
        dense_hidden:      int   = 32, # 32
        input_dropout:     float = 0.10,
        recurrent_dropout: float = DROPOUT,
        dense_dropout:     float = DROPOUT,
        n_classes:         int   = 3,
    ):
        super().__init__()

        self.n_features   = n_features
        self.seq_len      = seq_len
        self.lstm2_hidden = lstm2_hidden

        # ── input dropout ────────────────────────────────────────────────────
        self.input_dropout = nn.Dropout(p=input_dropout)

        # ── LSTM stack ───────────────────────────────────────────────────────
        # dropout= applies between layer 1 output → layer 2 input
        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = lstm1_hidden,
            num_layers    = 2,
            batch_first   = True,
            dropout       = recurrent_dropout,
            bidirectional = False,
        )

        # explicit 128 → 64 taper applied to h_n[-1] after the LSTM stack
        self.hidden_proj = nn.Linear(lstm1_hidden, lstm2_hidden)

        # ── batch normalisation ───────────────────────────────────────────────
        self.batch_norm = nn.BatchNorm1d(lstm2_hidden)

        # ── classification head ───────────────────────────────────────────────
        self.dense = nn.Sequential(
            nn.Linear(lstm2_hidden, dense_hidden),
            nn.ReLU(),
            nn.Dropout(p=dense_dropout),
            nn.Linear(dense_hidden, n_classes),
        )

        # softmax for inference — not needed during training
        self.softmax = nn.Softmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform for LSTM gates; orthogonal for recurrent weights; forget gate bias=1."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)   # forget gate bias = 1

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
        x = self.input_dropout(x)                       # [batch, seq, features]

        out, (h_n, _) = self.lstm(x)                    # h_n: [2, batch, 128]
        context = h_n[-1]                               # [batch, 128]
        context = self.hidden_proj(context)             # [batch, 64]
        context = self.batch_norm(context)              # [batch, 64]

        logits = self.dense(context)                    # [batch, 3]

        if return_logits:
            return logits
        return self.softmax(logits)


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
        Conviction margin logic:
            conviction_up = P(up) − max(P(no_hit), P(dn))
            conviction_dn = P(dn) − max(P(no_hit), P(up))
        Only fires if conviction > threshold.
        """
        probs = self.predict_proba(X)
        p_dn, p_no_hit, p_up = probs[:, 0], probs[:, 1], probs[:, 2]

        conviction_up = p_up - np.maximum(p_no_hit, p_dn)
        conviction_dn = p_dn - np.maximum(p_no_hit, p_up)

        signals = np.full(len(X), "NO_TRADE", dtype=object)
        signals[conviction_up > self.threshold] = "LONG"
        signals[conviction_dn > self.threshold] = "SHORT"

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
# ABLATION CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# edit here
GROUP_I = ["ROC_3", "ROC_5", "ROC_10", "MOM_3", "RETURNS_1"]
GROUP_J = ["ATR_5", "ATR_14", "ATR_RATIO", "ATR_NORM_ROC", "RANGE_RATIO"]
GROUP_K = ["RSI_14", "RSI_SLOPE", "STOCH_K", "CCI_5"]
GROUP_L1 = ["DELTA_1", "DELTA_3", "VOL_SPIKE"] # , "DELTA_DIV"
GROUP_L2 = ["DELTA_1", "DELTA_3", "VOL_SPIKE","DELTA_DIV"] # , 
GROUP_M = ["DIST_HIGH_5", "DIST_LOW_5", "DIST_HIGH_10", "DIST_LOW_10","RANGE_POS"]
# GROUP_N = ["DIST_HIGH_15STR", "DIST_HIGH_45STR","DIST_LOW_15STR", "DIST_LOW_45STR", "RANGE_15STR", "RANGE_45STR"]

ABLATION_RUNS = {
    "Run1_I":   {"groups": "I",     "features": GROUP_I},
    "Run2_I+J":   {"groups": "I+J",     "features": GROUP_I+GROUP_J},
    "Run3_I+J+K":   {"groups": "I+J+K",     "features": GROUP_I+GROUP_J+GROUP_K},
    "Run4_I+J+K+L-x":   {"groups": "I+J+K+L",     "features": GROUP_I+GROUP_J+GROUP_K+GROUP_L1},
    "Run5_x":       {"groups": "I+J+K+L+M",     "features": GROUP_I+GROUP_J+GROUP_K+GROUP_L1+GROUP_M},
    "Run5_I+J+K+L+M-x":       {"groups": "I+J+K+L+M+DELTA_DIV",     "features": GROUP_I+GROUP_J+GROUP_K+GROUP_L1+GROUP_M},
    # "Run6_L":   {"groups": "L",     "features": GROUP_L2},
    # "Run7_I+L": {"groups": "I+L",   "features": GROUP_I + GROUP_L2},
    # "Run8_IJKLN":   {"groups": "I+J+K+L+N",     "features": GROUP_I+GROUP_J+GROUP_K+GROUP_L+GROUP_N},
    # "Run9_IJKLMN":   {"groups": "I+J+K+L+M+N",     "features": GROUP_I+GROUP_J+GROUP_K+GROUP_L+GROUP_M+GROUP_N},
}


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _epoch_pass(model, loader, device, criterion, optimizer=None):
    """Single forward (+ optional backward) pass over a DataLoader.

    Returns (avg_loss, accuracy, y_true_array, y_pred_array).
    Pass optimizer=None for validation (no gradient update).
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_true, all_pred = [], []

    with torch.set_grad_enabled(training):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch, return_logits=True)
            loss   = criterion(logits, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            all_true.extend(y_batch.cpu().tolist())
            all_pred.extend(logits.argmax(dim=-1).cpu().tolist())

    n        = len(all_true)
    avg_loss = total_loss / n
    acc      = sum(t == p for t, p in zip(all_true, all_pred)) / n
    return avg_loss, acc, np.array(all_true), np.array(all_pred)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run_ablation(run_name: str, feature_cols: list, device: torch.device):
    """
    Full ablation run:
        1. Load & z-score normalise data (fit on train only)
        2. Build sliding-window datasets
        3. Train TripleBarrierLSTM with early stopping
        4. Print per-epoch stats + final classification report
    """
    n_feat     = len(feature_cols)
    wall_start = time.time()

    # ── header ─────────────────────────────────────────────────────────────────
    print("=" * 105)
    print(f"  RUN      : {run_name}")
    print(f"  Features : {feature_cols}")
    print(f"  N feat   : {n_feat}  |  Seq len: {SEQ_LEN}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Device   : {device}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 105)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"  Loading  : {TRAIN_PATH.name}  /  {VAL_PATH.name}")
    X_tr_raw, y_tr = load_jsonl(TRAIN_PATH, feature_cols)
    X_va_raw, y_va = load_jsonl(VAL_PATH,   feature_cols)
    print(f"  Rows     : train={len(X_tr_raw):,}  val={len(X_va_raw):,}")

    # ── z-score + winsorise (fit on train only) ─────────────────────────────
    # already applied
    # scaler = StandardScaler()
    # X_tr   = np.clip(scaler.fit_transform(X_tr_raw), -3, 3).astype(np.float32)
    # X_va   = np.clip(scaler.transform(X_va_raw),     -3, 3).astype(np.float32)
    X_tr = X_tr_raw.astype(np.float32)   # already clean from feature_pipeline
    X_va = X_va_raw.astype(np.float32)
    # ─────────────────────────────────────────────────────────────────────────

    # ── datasets & loaders ────────────────────────────────────────────────────
    train_ds     = BarSequenceDataset(X_tr, y_tr, SEQ_LEN)
    val_ds       = BarSequenceDataset(X_va, y_va, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Seqs     : train={len(train_ds):,}  val={len(val_ds):,}")

    # ── model & optimiser ─────────────────────────────────────────────────────
    model     = TripleBarrierLSTM(n_features=n_feat, seq_len=SEQ_LEN).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=1.5)
    _weights  = torch.tensor([1.00, 1.00, 1.00], dtype=torch.float32).to(device)  # DOWN, NO_HIT, UP
    criterion = nn.CrossEntropyLoss(weight=_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(f"  Params   : {n_params:,}")
    print(f"  Epochs   : up to {MAX_EPOCHS}  |  Early stop patience: {PATIENCE}")

    # ── epoch table ───────────────────────────────────────────────────────────
    print("-" * 105)
    print(
        f"{'Epoch':>6}  {'T-Loss':>8}  {'T-Acc':>7}  "
        f"{'V-Loss':>8}  {'V-Acc':>7}  {'V-F1mac':>8}  "
        f"{'LR':>8}  {'EpTime':>7}  {'Elapsed':>8}"
    )
    print("-" * 105)

    best_val_loss  = float("inf")
    best_weights   = None
    epochs_no_impr = 0
    history        = []

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_start = time.time()

        t_loss, t_acc, _, _              = _epoch_pass(model, train_loader, device, criterion, optimizer)
        v_loss, v_acc, y_true, y_pred    = _epoch_pass(model, val_loader,   device, criterion)

        scheduler.step(v_loss)
        lr_now   = optimizer.param_groups[0]["lr"]
        v_f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
        ep_sec   = time.time() - ep_start
        wall_sec = time.time() - wall_start

        print(
            f"{epoch:>6d}  {t_loss:>8.4f}  {t_acc:>6.4f}  "
            f"{v_loss:>8.4f}  {v_acc:>6.4f}  {v_f1:>8.4f}  "
            f"{lr_now:.2e}  {ep_sec:>5.1f}s  {wall_sec/60:>6.1f}m"
        )

        history.append({
            "epoch": epoch, "t_loss": t_loss, "t_acc": t_acc,
            "v_loss": v_loss, "v_acc": v_acc, "v_f1": v_f1, "lr": lr_now,
        })

        # ── early stopping + checkpoint ───────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
            # save best checkpoint to disk immediately
            ckpt_path = MODEL_DIR / f"{run_name}_best.pt"
            torch.save({
                "epoch":        epoch,
                "run_name":     run_name,
                "feature_cols": feature_cols,
                "val_loss":     v_loss,
                "val_acc":      v_acc,
                "val_f1":       v_f1,
                "model_state":  best_weights,
                "model_kwargs": {
                    "n_features": n_feat,
                    "seq_len":    SEQ_LEN,
                },
            }, ckpt_path)
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}  (no val improvement for {PATIENCE} epochs).")
                break

    # ── restore best weights ──────────────────────────────────────────────────
    if best_weights:
        model.load_state_dict(best_weights)

    # ── final evaluation with best weights ────────────────────────────────────
    _, _, y_true_f, y_pred_f = _epoch_pass(model, val_loader, device, criterion)

    total_elapsed = time.time() - wall_start
    best_ep       = min(history, key=lambda h: h["v_loss"])

    print("-" * 105)
    print(
        f"\n  Best epoch : {best_ep['epoch']}"
        f"  |  val_loss={best_ep['v_loss']:.4f}"
        f"  val_acc={best_ep['v_acc']:.4f}"
        f"  val_F1mac={best_ep['v_f1']:.4f}"
    )
    print(f"  Total time : {total_elapsed:.1f}s  ({total_elapsed / 60:.1f} min)")
    print(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n  Classification report (val, best weights):")
    print(classification_report(y_true_f, y_pred_f, target_names=CLASS_NAMES, digits=4))

    # ── test set evaluation ───────────────────────────────────────────────────
    if TEST_PATH.exists():
        X_te_raw, y_te = load_jsonl(TEST_PATH, feature_cols)
        X_te           = X_te_raw.astype(np.float32)
        test_ds        = BarSequenceDataset(X_te, y_te, SEQ_LEN)
        test_loader    = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        _, _, y_true_t, y_pred_t = _epoch_pass(model, test_loader, device, criterion)
        test_f1  = f1_score(y_true_t, y_pred_t, average="macro", zero_division=0)
        val_f1   = best_ep["v_f1"]
        gap      = val_f1 - test_f1

        print(f"  Val  F1 macro : {val_f1:.4f}")
        print(f"  Test F1 macro : {test_f1:.4f}")
        print(f"  Regime gap    : {gap:.4f}  ", end="")
        if   gap < 0.02:  print("← generalises well across regime shift")
        elif gap < 0.08:  print("← expected structural gap (normal)")
        elif gap < 0.15:  print("← WARN: moderate overfitting to bull regime")
        else:             print("← ALARM: severe regime overfitting")
        if test_f1 < 0.40:
            print(f"  ALARM: test F1 {test_f1:.4f} < 0.40 floor — not actionable for live trading")

        print(f"\n  Classification report (test, best weights):")
        print(classification_report(y_true_t, y_pred_t, target_names=CLASS_NAMES, digits=4))
    else:
        print(f"\n  Test set not found at {TEST_PATH} — skipping test evaluation.")

    # ── final confirmed model save ────────────────────────────────────────────
    final_path = MODEL_DIR / f"{run_name}_final.pt"
    torch.save({
        "run_name":     run_name,
        "feature_cols": feature_cols,
        "val_loss":     best_ep["v_loss"],
        "val_acc":      best_ep["v_acc"],
        "val_f1":       best_ep["v_f1"],
        "best_epoch":   best_ep["epoch"],
        "model_state":  best_weights,
        "model_kwargs": {
            "n_features": n_feat,
            "seq_len":    SEQ_LEN,
        },
    }, final_path)
    print(f"\n  Model saved  : {final_path}")
    print("=" * 105)
    print()

    return model, history

# Add focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — ablation runs 1-9
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # RUNS_TO_EXECUTE = ["Run1_I","Run2_I+J","Run3_I+J+K","Run4_I+J+K+L-x","Run5_I+J+K+L+M-x","Run6_L","Run7_I+L"]
    RUNS_TO_EXECUTE = ["Run5_I+J+K+L+M-x"]

    print(f"\nAblation Study  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs    : {RUNS_TO_EXECUTE}")
    print(f"Device  : {device}\n")

    results = {}
    study_start = time.time()

    for run_name in RUNS_TO_EXECUTE:
        set_seed(42)
        cfg                  = ABLATION_RUNS[run_name]
        model, history       = run_ablation(run_name, cfg["features"], device)
        results[run_name]    = history

    # ── comparison summary ─────────────────────────────────────────────────────
    study_elapsed = time.time() - study_start
    print("\n" + "=" * 65)
    print("  ABLATION COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Run':<14}  {'BestEp':>7}  {'V-Loss':>8}  {'V-Acc':>7}  {'V-F1mac':>8}")
    print("-" * 65)
    for run_name, hist in results.items():
        best = min(hist, key=lambda h: h["v_loss"])
        print(
            f"  {run_name:<14}  {best['epoch']:>7d}  "
            f"{best['v_loss']:>8.4f}  {best['v_acc']:>6.4f}  {best['v_f1']:>8.4f}"
        )
    print("=" * 65)
    print(f"\n  Total study time : {study_elapsed:.1f}s  ({study_elapsed / 60:.1f} min)")
    print(f"  Completed        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")