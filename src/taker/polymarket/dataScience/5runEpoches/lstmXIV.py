"""
LSTM 3-Class Triple-Barrier Model
Binance 5-min Bars | PyTorch Implementation
XIV — Run 3 UP Recall Interventions

Changes from lstmXII:
  1. W_UP class weight pulled into per-run config (Intervention 1)
  2. AsymmetricFocalLoss added (Intervention 2)
  3. Epoch table now prints UP recall and DN recall per epoch
  4. Comparison summary includes UP/DN recall columns
  5. Architecture reset to baseline (HIDDEN_NODES=128, proj=64, dense=32)
  6. MAX_EPOCHS=30, PATIENCE=10 (Run15 confirmed epoch 3 ceiling)
  7. Run config cleaned — only Run 3 intervention runs present
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN      = 20
HIDDEN_NODES = 128          # baseline — do not change for Run 3
DROPOUT      = 0.20

BATCH_SIZE   = 512
MAX_EPOCHS   = 30           # epoch 3 is confirmed ceiling; 30 is sufficient
LR           = 1e-3
PATIENCE     = 10

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


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────────────

class TripleBarrierLSTM(nn.Module):
    """
    Baseline architecture — unchanged across all Run 3 interventions.

        Input [batch, seq=20, features=N]
            │
            ├── Input Dropout (p=0.10)
            ├── LSTM Layer 1+2  (hidden=128, recurrent_dropout=0.20)
            ├── Linear(128 → 64)
            ├── BatchNorm1d(64)
            ├── Linear(64 → 32) + ReLU + Dropout(0.20)
            └── Linear(32 → 3) + Softmax
    """

    def __init__(
        self,
        n_features:        int,
        seq_len:           int   = SEQ_LEN,
        lstm1_hidden:      int   = HIDDEN_NODES,
        lstm2_hidden:      int   = 64,
        dense_hidden:      int   = 32,
        input_dropout:     float = 0.10,
        recurrent_dropout: float = DROPOUT,
        dense_dropout:     float = DROPOUT,
        n_classes:         int   = 3,
    ):
        super().__init__()

        self.n_features   = n_features
        self.seq_len      = seq_len
        self.lstm2_hidden = lstm2_hidden

        self.input_dropout = nn.Dropout(p=input_dropout)

        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = lstm1_hidden,
            num_layers    = 2,
            batch_first   = True,
            dropout       = recurrent_dropout,
            bidirectional = False,
        )

        self.hidden_proj = nn.Linear(lstm1_hidden, lstm2_hidden)
        self.batch_norm  = nn.BatchNorm1d(lstm2_hidden)

        self.dense = nn.Sequential(
            nn.Linear(lstm2_hidden, dense_hidden),
            nn.ReLU(),
            nn.Dropout(p=dense_dropout),
            nn.Linear(dense_hidden, n_classes),
        )

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
        x = self.input_dropout(x)
        out, (h_n, _) = self.lstm(x)
        context = h_n[-1]
        context = self.hidden_proj(context)
        context = self.batch_norm(context)
        logits  = self.dense(context)
        if return_logits:
            return logits
        return self.softmax(logits)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    """
    Intervention 2 — class-specific focal loss.

    Applies a different gamma per class so easy DOWN predictions contribute
    less gradient, forcing the model to attend more to harder UP patterns.

    Args:
        gamma_per_class : tensor [gamma_DOWN, gamma_NO_HIT, gamma_UP]
                          Higher gamma = easy examples down-weighted more.
                          Recommended: [2.5, 1.0, 1.0]
        weight          : optional class weight tensor [w_DOWN, w_NO_HIT, w_UP]
                          Stack with Intervention 1 by passing W_UP > 1.0 here.
    """
    def __init__(self, gamma_per_class: torch.Tensor, weight: torch.Tensor = None):
        super().__init__()
        self.register_buffer("gamma", gamma_per_class)
        self.register_buffer("weight", weight)          # None = no class weighting

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce           = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        probs        = F.softmax(logits, dim=-1)
        pt           = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        gamma        = self.gamma[targets]
        focal_weight = (1 - pt) ** gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  (unchanged)
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
        self.model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        probs  = self.model(tensor, return_logits=False)
        return probs.cpu().numpy()

    def predict_signal(self, X: np.ndarray) -> dict:
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
# FEATURE GROUPS  (22 features — Run 2 baseline, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

GROUP_I  = ["ROC_3", "ROC_5", "ROC_10", "MOM_3", "RETURNS_1"]
GROUP_J  = ["ATR_5", "ATR_14", "ATR_RATIO", "ATR_NORM_ROC", "RANGE_RATIO"]
GROUP_K  = ["RSI_14", "RSI_SLOPE", "STOCH_K", "CCI_5"]
GROUP_L  = ["DELTA_1", "DELTA_3", "VOL_SPIKE"]          # BUY_RATIO and DELTA_DIV excluded
GROUP_M  = ["DIST_HIGH_5", "DIST_LOW_5", "DIST_HIGH_10", "DIST_LOW_10", "RANGE_POS"]

FULL_FEATURES = GROUP_I + GROUP_J + GROUP_K + GROUP_L + GROUP_M   # 22 features


# ─────────────────────────────────────────────────────────────────────────────
# RUN 3 INTERVENTION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
#
# Each run entry has:
#   features  : feature list (all use FULL_FEATURES)
#   w_up      : UP class weight for CrossEntropyLoss  (Intervention 1)
#   loss_type : "weighted_ce" | "asymmetric_focal"
#   gamma_dn  : DOWN gamma for AsymmetricFocalLoss     (Intervention 2 only)
#
# Decision rule — stop as soon as success condition is met:
#   UP recall > 0.30  AND  DOWN recall > 0.45
#
# Run order: Run17 → Run18 → Run19 → Run20 → Run21
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_RUNS = {
    # ── Intervention 1 — UP class weight only ────────────────────────────────
    "Run17_W15": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      1.5,
        "gamma_dn":  None,          # not used for weighted_ce
    },
     # Run17y
    "Run17y_W107": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      1.07,
        "gamma_dn":  None,
    },
     # Run17x
    "Run17x_W105": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      1.05,
        "gamma_dn":  None,
    },
        # Run17b
    "Run17b_W110": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      1.1,
        "gamma_dn":  None,
    },
    # Run17c
    "Run17c_W120": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      1.2,
        "gamma_dn":  None,
    },
    # Run17d
    "Run17d_W130": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      1.3,
        "gamma_dn":  None,
    },
    "Run18_W20": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      2.0,
        "gamma_dn":  None,
    },
    "Run19_W25": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_up":      2.5,
        "gamma_dn":  None,
    },
    # ── Intervention 2 — AsymmetricFocalLoss (run only if Int 1 exhausted) ──
    "Run20_AFL": {
        "features":  FULL_FEATURES,
        "loss_type": "asymmetric_focal",
        "w_up":      1.0,           # no class weight — focal only
        "gamma_dn":  2.5,
    },
    "Run21a_135": {
        "features":  FULL_FEATURES,
        "loss_type": "asymmetric_focal",
        "w_up":      1.0,           # no class weight — focal only
        "gamma_dn":  1.35,
    },
    "Run20a_20": {
        "features":  FULL_FEATURES,
        "loss_type": "asymmetric_focal",
        "w_up":      1.0,           # no class weight — focal only
        "gamma_dn":  2.0,
    },
    "Run20_15": {
        "features":  FULL_FEATURES,
        "loss_type": "asymmetric_focal",
        "w_up":      1.0,           # no class weight — focal only
        "gamma_dn":  1.5,
    },
    "Run21_AFL_W20": {
        "features":  FULL_FEATURES,
        "loss_type": "asymmetric_focal",
        "w_up":      2.0,           # focal + class weight stacked
        "gamma_dn":  2.5,
    },
    "Run22a": {
    "features":  FULL_FEATURES,
    "loss_type": "asymmetric_focal",
    "w_up":      1.0,
    "w_dn":      1.0, 
    "gamma_dn":  1.42,
    },
    "Run22b": {
    "features":  FULL_FEATURES,
    "loss_type": "asymmetric_focal",
    "w_up":      1.0,
    "w_dn":      1.2,      # ← this is what was missing
    "gamma_dn":  1.5,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
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


def _build_criterion(cfg: dict, device: torch.device) -> nn.Module:
    """Build the loss function from run config."""
    w_up      = cfg["w_up"]
    loss_type = cfg["loss_type"]
    w_dn = cfg.get("w_dn", 1.0)

    class_weights = torch.tensor([w_dn, 1.0, w_up], dtype=torch.float32).to(device)

    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)

    elif loss_type == "asymmetric_focal":
        gamma_dn        = cfg["gamma_dn"] # also add gamma sideway
        gamma_per_class = torch.tensor([gamma_dn, 1.0, 1.0], dtype=torch.float32).to(device)
        # w_up=1.0 means no class weighting; pass None to skip weight in focal loss
        # weight_arg = class_weights if w_up != 1.0 else None
        weight_arg = class_weights if (w_up != 1.0 or w_dn != 1.0) else None
        return AsymmetricFocalLoss(gamma_per_class=gamma_per_class, weight=weight_arg)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(run_name: str, cfg: dict, device: torch.device):
    """
    Full ablation run:
        1. Load data
        2. Build criterion from run config
        3. Train TripleBarrierLSTM with early stopping
        4. Print per-epoch stats including UP/DN recall
        5. Final classification report on val + test
    """
    feature_cols = cfg["features"]
    n_feat       = len(feature_cols)
    wall_start   = time.time()

    # ── header ────────────────────────────────────────────────────────────────
    print("=" * 115)
    print(f"  RUN       : {run_name}")
    print(f"  Loss      : {cfg['loss_type']}  |  W_UP={cfg['w_up']}  |  W_DN={cfg.get('w_dn', 1.0)}  |  gamma_DN={cfg['gamma_dn']}")
    print(f"  Features  : {feature_cols}")
    print(f"  N feat    : {n_feat}  |  Seq len: {SEQ_LEN}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Device    : {device}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 115)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"  Loading   : {TRAIN_PATH.name}  /  {VAL_PATH.name}")
    X_tr_raw, y_tr = load_jsonl(TRAIN_PATH, feature_cols)
    X_va_raw, y_va = load_jsonl(VAL_PATH,   feature_cols)
    print(f"  Rows      : train={len(X_tr_raw):,}  val={len(X_va_raw):,}")

    X_tr = X_tr_raw.astype(np.float32)   # already normalised by feature_pipeline
    X_va = X_va_raw.astype(np.float32)

    # ── datasets & loaders ────────────────────────────────────────────────────
    train_ds     = BarSequenceDataset(X_tr, y_tr, SEQ_LEN)
    val_ds       = BarSequenceDataset(X_va, y_va, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Seqs      : train={len(train_ds):,}  val={len(val_ds):,}")

    # ── model, criterion, optimiser ───────────────────────────────────────────
    model     = TripleBarrierLSTM(n_features=n_feat, seq_len=SEQ_LEN).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = _build_criterion(cfg, device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    print(f"  Params    : {n_params:,}")
    print(f"  Epochs    : up to {MAX_EPOCHS}  |  Early stop patience: {PATIENCE}")

    # ── epoch table header ────────────────────────────────────────────────────
    print("-" * 115)
    print(
        f"{'Epoch':>6}  {'T-Loss':>8}  {'T-Acc':>7}  "
        f"{'V-Loss':>8}  {'V-Acc':>7}  {'V-F1mac':>8}  "
        f"{'UP-rec':>7}  {'DN-rec':>7}  "
        f"{'LR':>8}  {'EpTime':>7}  {'Elapsed':>8}"
    )
    print("-" * 115)

    best_val_loss  = float("inf")
    best_weights   = None
    epochs_no_impr = 0
    history        = []

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_start = time.time()

        t_loss, t_acc, _, _           = _epoch_pass(model, train_loader, device, criterion, optimizer)
        v_loss, v_acc, y_true, y_pred = _epoch_pass(model, val_loader,   device, criterion)

        scheduler.step(v_loss)
        lr_now    = optimizer.param_groups[0]["lr"]
        v_f1      = f1_score(y_true, y_pred, average="macro", zero_division=0)
        up_recall = recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
        dn_recall = recall_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
        ep_sec    = time.time() - ep_start
        wall_sec  = time.time() - wall_start

        print(
            f"{epoch:>6d}  {t_loss:>8.4f}  {t_acc:>6.4f}  "
            f"{v_loss:>8.4f}  {v_acc:>6.4f}  {v_f1:>8.4f}  "
            f"{up_recall:>7.4f}  {dn_recall:>7.4f}  "
            f"{lr_now:.2e}  {ep_sec:>5.1f}s  {wall_sec/60:>6.1f}m"
        )

        history.append({
            "epoch": epoch, "t_loss": t_loss, "t_acc": t_acc,
            "v_loss": v_loss, "v_acc": v_acc, "v_f1": v_f1,
            "up_recall": up_recall, "dn_recall": dn_recall, "lr": lr_now,
        })

        # ── early stopping + checkpoint ───────────────────────────────────────
        if v_loss < best_val_loss:
            best_val_loss  = v_loss
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
            ckpt_path = MODEL_DIR / f"{run_name}_best.pt"
            torch.save({
                "epoch":        epoch,
                "run_name":     run_name,
                "feature_cols": feature_cols,
                "val_loss":     v_loss,
                "val_acc":      v_acc,
                "val_f1":       v_f1,
                "up_recall":    up_recall,
                "dn_recall":    dn_recall,
                "model_state":  best_weights,
                "model_kwargs": {"n_features": n_feat, "seq_len": SEQ_LEN},
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

    print("-" * 115)
    print(
        f"\n  Best epoch : {best_ep['epoch']}"
        f"  |  val_loss={best_ep['v_loss']:.4f}"
        f"  val_acc={best_ep['v_acc']:.4f}"
        f"  val_F1mac={best_ep['v_f1']:.4f}"
        f"  UP_recall={best_ep['up_recall']:.4f}"
        f"  DN_recall={best_ep['dn_recall']:.4f}"
    )
    print(f"  Total time : {total_elapsed:.1f}s  ({total_elapsed / 60:.1f} min)")
    print(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n  Classification report (val, best weights):")
    print(classification_report(y_true_f, y_pred_f, target_names=CLASS_NAMES, digits=4))

    # ── success condition check ───────────────────────────────────────────────
    up_ok = best_ep["up_recall"] > 0.30
    dn_ok = best_ep["dn_recall"] > 0.45
    if up_ok and dn_ok:
        print(f"  ✓ SUCCESS: UP recall={best_ep['up_recall']:.4f} > 0.30  AND  DN recall={best_ep['dn_recall']:.4f} > 0.45")
        print(f"  → Stop intervention sweep. Use {run_name}.")
    else:
        print(f"  ✗ NOT MET: UP recall={best_ep['up_recall']:.4f} (need >0.30)  |  DN recall={best_ep['dn_recall']:.4f} (need >0.45)")
        if not dn_ok:
            print(f"  → DN recall below floor — Intervention 1 may be exhausted. Check before proceeding.")

    # ── test set evaluation ───────────────────────────────────────────────────
    if TEST_PATH.exists():
        X_te_raw, y_te = load_jsonl(TEST_PATH, feature_cols)
        X_te           = X_te_raw.astype(np.float32)
        test_ds        = BarSequenceDataset(X_te, y_te, SEQ_LEN)
        test_loader    = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        _, _, y_true_t, y_pred_t = _epoch_pass(model, test_loader, device, criterion)
        test_f1      = f1_score(y_true_t, y_pred_t, average="macro", zero_division=0)
        test_up_rec  = recall_score(y_true_t, y_pred_t, labels=[2], average="macro", zero_division=0)
        test_dn_rec  = recall_score(y_true_t, y_pred_t, labels=[0], average="macro", zero_division=0)
        val_f1       = best_ep["v_f1"]
        gap          = val_f1 - test_f1

        print(f"  Val  F1 macro : {val_f1:.4f}  (UP recall={best_ep['up_recall']:.4f}  DN recall={best_ep['dn_recall']:.4f})")
        print(f"  Test F1 macro : {test_f1:.4f}  (UP recall={test_up_rec:.4f}  DN recall={test_dn_rec:.4f})")
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

    # ── save final model ──────────────────────────────────────────────────────
    final_path = MODEL_DIR / f"{run_name}_final.pt"
    torch.save({
        "run_name":     run_name,
        "feature_cols": feature_cols,
        "loss_config":  {"loss_type": cfg["loss_type"], "w_up": cfg["w_up"], "gamma_dn": cfg["gamma_dn"]},
        "val_loss":     best_ep["v_loss"],
        "val_acc":      best_ep["v_acc"],
        "val_f1":       best_ep["v_f1"],
        "up_recall":    best_ep["up_recall"],
        "dn_recall":    best_ep["dn_recall"],
        "best_epoch":   best_ep["epoch"],
        "model_state":  best_weights,
        "model_kwargs": {"n_features": n_feat, "seq_len": SEQ_LEN},
    }, final_path)
    print(f"\n  Model saved  : {final_path}")
    print("=" * 115)
    print()

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Set which run to execute ──────────────────────────────────────────────
    # Run one at a time. Check success condition before proceeding to next.
    #
    # Intervention 1 order:  Run17_W15  →  Run18_W20  →  Run19_W25
    # Intervention 2 order:  Run20_AFL  →  Run21_AFL_W20
    #
    RUNS_TO_EXECUTE = ["Run22b"]

    print(f"\nAblation Study  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs    : {RUNS_TO_EXECUTE}")
    print(f"Device  : {device}\n")

    results      = {}
    study_start  = time.time()

    for run_name in RUNS_TO_EXECUTE:
        set_seed(42)
        cfg              = ABLATION_RUNS[run_name]
        model, history   = run_ablation(run_name, cfg, device)
        results[run_name] = history

    # ── comparison summary ────────────────────────────────────────────────────
    study_elapsed = time.time() - study_start
    print("\n" + "=" * 85)
    print("  ABLATION COMPARISON SUMMARY")
    print("=" * 85)
    print(f"  {'Run':<18}  {'BestEp':>7}  {'V-Loss':>8}  {'V-F1mac':>8}  {'UP-rec':>7}  {'DN-rec':>7}  {'W_UP':>5}")
    print("-" * 85)
    for run_name, hist in results.items():
        best = min(hist, key=lambda h: h["v_loss"])
        cfg  = ABLATION_RUNS[run_name]
        ok   = "✓" if best["up_recall"] > 0.30 and best["dn_recall"] > 0.45 else "✗"
        print(
            f"  {run_name:<18}  {best['epoch']:>7d}  "
            f"{best['v_loss']:>8.4f}  {best['v_f1']:>8.4f}  "
            f"{best['up_recall']:>7.4f}  {best['dn_recall']:>7.4f}  "
            f"{cfg['w_up']:>5.1f}  {ok}"
        )
    print("=" * 85)
    print(f"\n  Total study time : {study_elapsed:.1f}s  ({study_elapsed / 60:.1f} min)")
    print(f"  Completed        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")