"""
LSTM 3-Class Triple-Barrier Model
Binance 5-min Bars | PyTorch Implementation
XVII — Run 4: Directional Precision + Recall Objective

Objective change from Run 3:
  Run 3 optimised F1 macro — wrong target for a trading signal generator.
  Run 4 optimises directional precision AND recall.
  NO_HIT class performance is irrelevant. Accept it going to zero.

Root cause being addressed:
  The model uses NO_HIT as a default escape route for uncertain bars.
  42% of predictions in Run20_15 went to NO_HIT with only 0.436 precision.
  This depletes the directional signal pool — most missed DOWN/UP bars
  are being routed to NO_HIT rather than predicted directionally.

Approach — NO_HIT suppression:
  Three mechanisms implemented, selectable per run:

  A) "masked_ce"      — NO_HIT bars excluded from loss entirely.
                        Gradient computed only on UP and DOWN bars.
                        Strongest intervention. Model never learns from NO_HIT.

  B) "weighted_ce"    — NO_HIT included but down-weighted to near zero.
                        w_nohit=0.1 or 0.05. Softer than masking.
                        Model is aware NO_HIT exists but it barely matters.

  C) "afl_masked"     — AsymmetricFocalLoss on DOWN class (gamma_DN=1.50,
                        best finding from Run 3) COMBINED with NO_HIT masking.
                        Stacks the two best interventions.

Key changes from lstmXIV:
  1. _build_criterion() supports "masked_ce" and "afl_masked" loss types
  2. _epoch_pass() supports per-batch masking when loss_type="masked_ce"/"afl_masked"
  3. Early stopping criterion changed from val_loss → directional F1
       directional_f1 = (f1_UP + f1_DOWN) / 2
       Checkpoint saved when directional_f1 improves, not when val_loss decreases
       Rationale: val_loss is still dominated by NO_HIT even when w_nohit is low.
                  Stopping on val_loss will still find a NO_HIT-dominant solution.
  4. Epoch table adds UP-prec and DN-prec columns
  5. Success condition updated:
       UP   precision > 0.45  AND  recall > 0.35
       DOWN precision > 0.45  AND  recall > 0.45
       NO_HIT — not evaluated
  6. Run config: 4 runs covering approaches A, B, C in order
  7. LR scheduler now monitors directional_f1 (mode=max) not val_loss

Baseline: Run10 (Run52_x_final.pt)
  Test F1=0.3935, UP recall=0.186, DN recall=0.510
  22 features, Groups I+J+K+L+M, seed=42
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
from sklearn.metrics import (
    classification_report, f1_score,
    recall_score, precision_score
)
from torch.utils.data import DataLoader, Dataset


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SEQ_LEN      = 20
HIDDEN_NODES = 128          # Run10 baseline — do not change
DROPOUT      = 0.20

BATCH_SIZE   = 512
MAX_EPOCHS   = 30
LR           = 1e-3
PATIENCE     = 10           # patience on directional F1, not val loss

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "mlData"
TRAIN_PATH = DATA_DIR / "202603-vX-train.jsonl"
VAL_PATH   = DATA_DIR / "202603-vX-val.jsonl"
TEST_PATH  = DATA_DIR / "202603-vX-test.jsonl"

MODEL_DIR  = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUCCESS CONDITION
# ─────────────────────────────────────────────────────────────────────────────
#
# Directional targets — NO_HIT is not evaluated.
# Both precision and recall required per direction.
#
UP_PREC_TARGET = 0.45
UP_REC_TARGET  = 0.35
DN_PREC_TARGET = 0.45
DN_REC_TARGET  = 0.45


# ─────────────────────────────────────────────────────────────────────────────
# DATASET  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP   = {-1: 0, 0: 1, 1: 2}
CLASS_NAMES = ["DOWN", "NO_HIT", "UP"]


def load_jsonl(path: Path, feature_cols: list) -> tuple:
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
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.y       = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  (Run10 baseline — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class TripleBarrierLSTM(nn.Module):
    """
    Run10 baseline architecture. Unchanged for Run 4.

        Input [batch, seq=20, features=22]
            │
            ├── Input Dropout (p=0.10)
            ├── LSTM Layer 1+2  (hidden=128, recurrent_dropout=0.20)
            ├── Linear(128 → 64) + BatchNorm1d(64)
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
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1)
        for layer in self.dense:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, return_logits: bool = True):
        x = self.input_dropout(x)
        _, (h_n, _) = self.lstm(x)
        ctx = h_n[-1]
        ctx = self.hidden_proj(ctx)
        ctx = self.batch_norm(ctx)
        logits = self.dense(ctx)
        if return_logits:
            return logits
        return self.softmax(logits)


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class MaskedCELoss(nn.Module):
    """
    Approach A — CrossEntropyLoss with NO_HIT bars masked out.

    Only UP (label=2) and DOWN (label=0) bars contribute to the loss.
    NO_HIT bars (label=1) are excluded from gradient computation entirely.
    The model is forced to develop directional conviction on every bar.

    Optional class weights [w_dn, w_up] applied to remaining classes.
    """
    def __init__(self, w_dn: float = 1.0, w_up: float = 1.0):
        super().__init__()
        # weights for all 3 classes — NO_HIT weight is irrelevant since masked
        self.register_buffer(
            "weight",
            torch.tensor([w_dn, 1.0, w_up], dtype=torch.float32)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask   = targets != 1                          # exclude NO_HIT (label=1)
        if mask.sum() == 0:
            return logits.sum() * 0.0                 # empty batch safety
        return F.cross_entropy(
            logits[mask], targets[mask],
            weight=self.weight.to(logits.device)
        )


class AsymmetricFocalLoss(nn.Module):
    """
    Run 3 best finding — gamma_DN=1.50 applied to DOWN class.
    Used in Approach C (afl_masked) stacked with NO_HIT masking.

    gamma_per_class : [gamma_DOWN, gamma_NO_HIT, gamma_UP]
    weight          : optional class weight tensor
    mask_nohit      : if True, NO_HIT bars excluded from gradient
    """
    def __init__(
        self,
        gamma_per_class: torch.Tensor,
        weight:          torch.Tensor = None,
        mask_nohit:      bool         = False,
    ):
        super().__init__()
        self.register_buffer("gamma", gamma_per_class)
        self.register_buffer("weight", weight)
        self.mask_nohit = mask_nohit

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mask_nohit:
            mask   = targets != 1
            if mask.sum() == 0:
                return logits.sum() * 0.0
            logits  = logits[mask]
            targets = targets[mask]

        ce           = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        probs        = F.softmax(logits, dim=-1)
        pt           = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        gamma        = self.gamma[targets]
        focal_weight = (1 - pt) ** gamma
        return (focal_weight * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE  (unchanged from baseline)
# ─────────────────────────────────────────────────────────────────────────────

class Predictor:
    CLASS_NAMES = {0: "DOWN", 1: "NO_HIT", 2: "UP"}

    def __init__(self, model: nn.Module, device: torch.device, conviction_threshold: float = 0.20):
        self.model     = model.to(device)
        self.device    = device
        self.threshold = conviction_threshold

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        return self.model(tensor, return_logits=False).cpu().numpy()

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
# FEATURE GROUPS  (22 features — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

GROUP_I  = ["ROC_3", "ROC_5", "ROC_10", "MOM_3", "RETURNS_1"]
GROUP_J  = ["ATR_5", "ATR_14", "ATR_RATIO", "ATR_NORM_ROC", "RANGE_RATIO"]
GROUP_K  = ["RSI_14", "RSI_SLOPE", "STOCH_K", "CCI_5"]
GROUP_L  = ["DELTA_1", "DELTA_3", "VOL_SPIKE"]
GROUP_M  = ["DIST_HIGH_5", "DIST_LOW_5", "DIST_HIGH_10", "DIST_LOW_10", "RANGE_POS"]

FULL_FEATURES = GROUP_I + GROUP_J + GROUP_K + GROUP_L + GROUP_M


# ─────────────────────────────────────────────────────────────────────────────
# RUN CONFIG
# ─────────────────────────────────────────────────────────────────────────────
#
# loss_type options:
#   "weighted_ce"   — standard CE with w_nohit near zero (Approach B)
#   "masked_ce"     — NO_HIT bars excluded from loss (Approach A)
#   "afl_masked"    — AsymmetricFocalLoss(gamma_DN=1.50) + NO_HIT masking (Approach C)
#
# Run order: Run4a → Run4b → Run4c → Run4d
# Stop at first run meeting the success condition.
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_RUNS = {

    # ── Approach B — NO_HIT down-weighted, mild ──────────────────────────────
    "Run4a_WNH01": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_nohit":   0.1,       # NO_HIT weight — 10% of normal
        "w_dn":      1.0,
        "w_up":      1.0,
        "gamma_dn":  None,
    },

    # ── Approach B — NO_HIT down-weighted, strong ────────────────────────────
    "Run4b_WNH005": {
        "features":  FULL_FEATURES,
        "loss_type": "weighted_ce",
        "w_nohit":   0.05,      # NO_HIT weight — 5% of normal
        "w_dn":      1.0,
        "w_up":      1.0,
        "gamma_dn":  None,
    },

    # ── Approach A — NO_HIT masked out entirely ───────────────────────────────
    "Run4c_masked": {
        "features":  FULL_FEATURES,
        "loss_type": "masked_ce",
        "w_nohit":   0.0,       # irrelevant — masked bars receive no gradient
        "w_dn":      1.0,
        "w_up":      1.0,
        "gamma_dn":  None,
    },

    # ── Approach C — AFL gamma_DN=1.50 + NO_HIT masking ──────────────────────
    "Run4d_AFL_masked": {
        "features":  FULL_FEATURES,
        "loss_type": "afl_masked",
        "w_nohit":   0.0,       # irrelevant — masked
        "w_dn":      1.0,
        "w_up":      1.0,
        "gamma_dn":  1.50,      # best finding from Run 3
    },
    # ── Approach C — AFL gamma_DN=1.50 + NO_HIT masking ──────────────────────
    "Run4d_AFL_115": {
        "features":  FULL_FEATURES,
        "loss_type": "afl_masked",
        "w_nohit":   0.0,       # little relevant to bimodial ?
        "w_dn":      1.0,
        "w_up":      1.0,
        "gamma_dn":  1.15,      # lower Focal strength
    },

    # ── Approach A + asymmetric class weights ─────────────────────────────────
    # Run only if Run4c shows UP recall still low after masking
    "Run4e_masked_WUP": {
        "features":  FULL_FEATURES,
        "loss_type": "masked_ce",
        "w_nohit":   0.0,
        "w_dn":      1.0,
        "w_up":      1.5,       # mild UP boost on top of masking
        "gamma_dn":  None,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_criterion(cfg: dict, device: torch.device) -> nn.Module:
    loss_type = cfg["loss_type"]
    w_nohit   = cfg.get("w_nohit", 1.0)
    w_dn      = cfg.get("w_dn",    1.0)
    w_up      = cfg.get("w_up",    1.0)
    gamma_dn  = cfg.get("gamma_dn", None)

    if loss_type == "weighted_ce":
        weights = torch.tensor([w_dn, w_nohit, w_up], dtype=torch.float32).to(device)
        return nn.CrossEntropyLoss(weight=weights)

    elif loss_type == "masked_ce":
        return MaskedCELoss(w_dn=w_dn, w_up=w_up)

    elif loss_type == "afl_masked":
        gamma_per_class = torch.tensor(
            [gamma_dn, 1.0, 1.0], dtype=torch.float32
        ).to(device)
        return AsymmetricFocalLoss(
            gamma_per_class=gamma_per_class,
            weight=None,
            mask_nohit=True,
        )

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def _compute_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-class precision and recall for UP and DOWN only."""
    up_prec = precision_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
    dn_prec = precision_score(y_true, y_pred, labels=[0], average="macro", zero_division=0)
    up_rec  = recall_score(y_true, y_pred,    labels=[2], average="macro", zero_division=0)
    dn_rec  = recall_score(y_true, y_pred,    labels=[0], average="macro", zero_division=0)
    f1_up   = f1_score(y_true, y_pred,        labels=[2], average="macro", zero_division=0)
    f1_dn   = f1_score(y_true, y_pred,        labels=[0], average="macro", zero_division=0)
    dir_f1  = (f1_up + f1_dn) / 2.0
    return {
        "up_prec": up_prec, "up_rec": up_rec, "f1_up": f1_up,
        "dn_prec": dn_prec, "dn_rec": dn_rec, "f1_dn": f1_dn,
        "dir_f1":  dir_f1,
    }


def _check_success(m: dict) -> bool:
    return (
        m["up_prec"] > UP_PREC_TARGET and
        m["up_rec"]  > UP_REC_TARGET  and
        m["dn_prec"] > DN_PREC_TARGET and
        m["dn_rec"]  > DN_REC_TARGET
    )


def _epoch_pass(model, loader, device, criterion, optimizer=None):
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


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(run_name: str, cfg: dict, device: torch.device):
    feature_cols = cfg["features"]
    n_feat       = len(feature_cols)
    wall_start   = time.time()

    # ── header ────────────────────────────────────────────────────────────────
    print("=" * 125)
    print(f"  RUN       : {run_name}")
    print(
        f"  Loss      : {cfg['loss_type']}"
        f"  |  w_nohit={cfg.get('w_nohit', 1.0)}"
        f"  |  w_dn={cfg.get('w_dn', 1.0)}"
        f"  |  w_up={cfg.get('w_up', 1.0)}"
        f"  |  gamma_dn={cfg.get('gamma_dn', None)}"
    )
    print(f"  Objective : directional precision+recall  |  NO_HIT — not evaluated")
    print(f"  Target    : UP prec>{UP_PREC_TARGET}  rec>{UP_REC_TARGET}  |  DN prec>{DN_PREC_TARGET}  rec>{DN_REC_TARGET}")
    print(f"  Features  : {feature_cols}")
    print(f"  N feat    : {n_feat}  |  Seq len: {SEQ_LEN}  |  Batch: {BATCH_SIZE}  |  LR: {LR}")
    print(f"  Device    : {device}")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 125)

    # ── load data ─────────────────────────────────────────────────────────────
    print(f"  Loading   : {TRAIN_PATH.name}  /  {VAL_PATH.name}")
    X_tr_raw, y_tr = load_jsonl(TRAIN_PATH, feature_cols)
    X_va_raw, y_va = load_jsonl(VAL_PATH,   feature_cols)
    print(f"  Rows      : train={len(X_tr_raw):,}  val={len(X_va_raw):,}")

    X_tr = X_tr_raw.astype(np.float32)
    X_va = X_va_raw.astype(np.float32)

    # ── datasets & loaders ────────────────────────────────────────────────────
    train_ds     = BarSequenceDataset(X_tr, y_tr, SEQ_LEN)
    val_ds       = BarSequenceDataset(X_va, y_va, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Seqs      : train={len(train_ds):,}  val={len(val_ds):,}")

    # ── model ─────────────────────────────────────────────────────────────────
    model    = TripleBarrierLSTM(n_features=n_feat, seq_len=SEQ_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = _build_criterion(cfg, device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    # scheduler monitors directional_f1 — mode=max, reduce when it stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    print(f"  Params    : {n_params:,}")
    print(f"  Early stop: patience={PATIENCE} on directional F1  (not val loss)")
    print(f"  Epochs    : up to {MAX_EPOCHS}")

    # ── epoch table header ────────────────────────────────────────────────────
    print("-" * 125)
    print(
        f"{'Epoch':>6}  {'T-Loss':>8}  {'T-Acc':>7}  "
        f"{'V-Loss':>8}  {'V-Acc':>7}  "
        f"{'DirF1':>7}  "
        f"{'UP-prec':>8}  {'UP-rec':>7}  "
        f"{'DN-prec':>8}  {'DN-rec':>7}  "
        f"{'LR':>8}  {'EpTime':>7}  {'Elapsed':>8}"
    )
    print("-" * 125)

    best_dir_f1    = -1.0
    best_weights   = None
    epochs_no_impr = 0
    history        = []

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_start = time.time()

        t_loss, t_acc, _, _           = _epoch_pass(model, train_loader, device, criterion, optimizer)
        v_loss, v_acc, y_true, y_pred = _epoch_pass(model, val_loader,   device, criterion)

        m        = _compute_directional_metrics(y_true, y_pred)
        dir_f1   = m["dir_f1"]
        v_f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)

        scheduler.step(dir_f1)
        lr_now   = optimizer.param_groups[0]["lr"]
        ep_sec   = time.time() - ep_start
        wall_sec = time.time() - wall_start

        print(
            f"{epoch:>6d}  {t_loss:>8.4f}  {t_acc:>6.4f}  "
            f"{v_loss:>8.4f}  {v_acc:>6.4f}  "
            f"{dir_f1:>7.4f}  "
            f"{m['up_prec']:>8.4f}  {m['up_rec']:>7.4f}  "
            f"{m['dn_prec']:>8.4f}  {m['dn_rec']:>7.4f}  "
            f"{lr_now:.2e}  {ep_sec:>5.1f}s  {wall_sec/60:>6.1f}m"
        )

        history.append({
            "epoch":   epoch,
            "t_loss":  t_loss, "t_acc": t_acc,
            "v_loss":  v_loss, "v_acc": v_acc,
            "v_f1mac": v_f1_mac,
            "dir_f1":  dir_f1,
            **m,
            "lr": lr_now,
        })

        # ── early stopping on directional F1 — save best ──────────────────────
        if dir_f1 > best_dir_f1:
            best_dir_f1    = dir_f1
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_impr = 0
            ckpt_path = MODEL_DIR / f"{run_name}_best.pt"
            torch.save({
                "epoch":          epoch,
                "run_name":       run_name,
                "feature_cols":   feature_cols,
                "loss_config":    {
                    "loss_type": cfg["loss_type"],
                    "w_nohit":   cfg.get("w_nohit", 1.0),
                    "w_dn":      cfg.get("w_dn", 1.0),
                    "w_up":      cfg.get("w_up", 1.0),
                    "gamma_dn":  cfg.get("gamma_dn", None),
                },
                "dir_f1":         dir_f1,
                "up_prec":        m["up_prec"],
                "up_rec":         m["up_rec"],
                "dn_prec":        m["dn_prec"],
                "dn_rec":         m["dn_rec"],
                "val_loss":       v_loss,
                "val_f1mac":      v_f1_mac,
                "model_state":    best_weights,
                "model_kwargs":   {"n_features": n_feat, "seq_len": SEQ_LEN},
            }, ckpt_path)
        else:
            epochs_no_impr += 1
            if epochs_no_impr >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}  (no directional F1 improvement for {PATIENCE} epochs).")
                break

    # ── restore best weights ──────────────────────────────────────────────────
    if best_weights:
        model.load_state_dict(best_weights)

    # ── final evaluation on val ───────────────────────────────────────────────
    _, _, y_true_f, y_pred_f = _epoch_pass(model, val_loader, device, criterion)
    m_best    = _compute_directional_metrics(y_true_f, y_pred_f)
    v_f1_best = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)

    best_ep = max(history, key=lambda h: h["dir_f1"])

    print("-" * 125)
    print(
        f"\n  Best epoch : {best_ep['epoch']}"
        f"  |  dir_F1={best_ep['dir_f1']:.4f}"
        f"  |  UP prec={best_ep['up_prec']:.4f}  rec={best_ep['up_rec']:.4f}"
        f"  |  DN prec={best_ep['dn_prec']:.4f}  rec={best_ep['dn_rec']:.4f}"
        f"  |  val_F1mac={best_ep['v_f1mac']:.4f}"
    )
    print(f"  Total time : {time.time() - wall_start:.1f}s  ({(time.time() - wall_start) / 60:.1f} min)")
    print(f"  Finished   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n  Classification report (val, best weights):")
    print(classification_report(y_true_f, y_pred_f, target_names=CLASS_NAMES, digits=4))

    # ── success condition ─────────────────────────────────────────────────────
    if _check_success(m_best):
        print(
            f"  ✓ SUCCESS"
            f"  UP prec={m_best['up_prec']:.4f}>{UP_PREC_TARGET}"
            f"  rec={m_best['up_rec']:.4f}>{UP_REC_TARGET}"
            f"  |  DN prec={m_best['dn_prec']:.4f}>{DN_PREC_TARGET}"
            f"  rec={m_best['dn_rec']:.4f}>{DN_REC_TARGET}"
        )
        print(f"  → Stop sweep. Use {run_name}.")
    else:
        print(
            f"  ✗ NOT MET"
            f"  UP prec={m_best['up_prec']:.4f}(>{UP_PREC_TARGET})"
            f"  rec={m_best['up_rec']:.4f}(>{UP_REC_TARGET})"
            f"  |  DN prec={m_best['dn_prec']:.4f}(>{DN_PREC_TARGET})"
            f"  rec={m_best['dn_rec']:.4f}(>{DN_REC_TARGET})"
        )

    # ── test set evaluation ───────────────────────────────────────────────────
    if TEST_PATH.exists():
        X_te_raw, y_te = load_jsonl(TEST_PATH, feature_cols)
        test_ds        = BarSequenceDataset(X_te_raw.astype(np.float32), y_te, SEQ_LEN)
        test_loader    = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        _, _, y_true_t, y_pred_t = _epoch_pass(model, test_loader, device, criterion)
        m_test      = _compute_directional_metrics(y_true_t, y_pred_t)
        test_f1mac  = f1_score(y_true_t, y_pred_t, average="macro", zero_division=0)
        gap         = v_f1_best - test_f1mac

        print(
            f"\n  Val  dir_F1={best_ep['dir_f1']:.4f}  F1mac={v_f1_best:.4f}"
            f"  UP prec={m_best['up_prec']:.4f} rec={m_best['up_rec']:.4f}"
            f"  DN prec={m_best['dn_prec']:.4f} rec={m_best['dn_rec']:.4f}"
        )
        print(
            f"  Test dir_F1={m_test['dir_f1']:.4f}  F1mac={test_f1mac:.4f}"
            f"  UP prec={m_test['up_prec']:.4f} rec={m_test['up_rec']:.4f}"
            f"  DN prec={m_test['dn_prec']:.4f} rec={m_test['dn_rec']:.4f}"
        )
        print(f"  Regime gap (F1mac): {gap:.4f}  ", end="")
        if   gap < 0.02: print("← generalises well")
        elif gap < 0.08: print("← expected structural gap")
        elif gap < 0.15: print("← WARN: moderate overfitting")
        else:            print("← ALARM: severe overfitting")

        print(f"\n  Classification report (test, best weights):")
        print(classification_report(y_true_t, y_pred_t, target_names=CLASS_NAMES, digits=4))
    else:
        print(f"\n  Test set not found at {TEST_PATH} — skipping.")

    # ── save final model ──────────────────────────────────────────────────────
    final_path = MODEL_DIR / f"{run_name}_final.pt"
    torch.save({
        "run_name":     run_name,
        "feature_cols": feature_cols,
        "loss_config":  {
            "loss_type": cfg["loss_type"],
            "w_nohit":   cfg.get("w_nohit", 1.0),
            "w_dn":      cfg.get("w_dn", 1.0),
            "w_up":      cfg.get("w_up", 1.0),
            "gamma_dn":  cfg.get("gamma_dn", None),
        },
        "best_epoch":   best_ep["epoch"],
        "dir_f1":       best_ep["dir_f1"],
        "up_prec":      best_ep["up_prec"],
        "up_rec":       best_ep["up_rec"],
        "dn_prec":      best_ep["dn_prec"],
        "dn_rec":       best_ep["dn_rec"],
        "val_f1mac":    best_ep["v_f1mac"],
        "model_state":  best_weights,
        "model_kwargs": {"n_features": n_feat, "seq_len": SEQ_LEN},
    }, final_path)
    print(f"\n  Model saved  : {final_path}")
    print("=" * 125)
    print()

    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Set which run to execute ──────────────────────────────────────────────
    # Run one at a time. Check success condition before proceeding.
    #
    # Run order:
    #   Run4a_WNH01      Approach B — w_nohit=0.10  (mild)
    #   Run4b_WNH005     Approach B — w_nohit=0.05  (strong)
    #   Run4c_masked     Approach A — NO_HIT masked out entirely
    #   Run4d_AFL_masked Approach C — AFL gamma_DN=1.50 + masking
    #   Run4e_masked_WUP Approach A + w_up=1.5  (only if UP recall low after 4c)
    #
    RUNS_TO_EXECUTE = ["Run4d_AFL_115"]

    print(f"\nAblation Study  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs    : {RUNS_TO_EXECUTE}")
    print(f"Device  : {device}")
    print(f"Objective: directional precision+recall  |  NO_HIT — not evaluated\n")

    results     = {}
    study_start = time.time()

    for run_name in RUNS_TO_EXECUTE:
        set_seed(42)
        cfg               = ABLATION_RUNS[run_name]
        model, history    = run_ablation(run_name, cfg, device)
        results[run_name] = history

    # ── comparison summary ────────────────────────────────────────────────────
    study_elapsed = time.time() - study_start
    print("\n" + "=" * 105)
    print("  RUN 4 COMPARISON SUMMARY  (Directional Objective)")
    print("=" * 105)
    print(
        f"  {'Run':<20}  {'BestEp':>7}  {'DirF1':>7}  "
        f"{'UP-prec':>8}  {'UP-rec':>7}  "
        f"{'DN-prec':>8}  {'DN-rec':>7}  {'Loss':>14}"
    )
    print("-" * 105)
    for run_name, hist in results.items():
        best = max(hist, key=lambda h: h["dir_f1"])
        cfg  = ABLATION_RUNS[run_name]
        ok   = "✓" if _check_success(best) else "✗"
        print(
            f"  {run_name:<20}  {best['epoch']:>7d}  {best['dir_f1']:>7.4f}  "
            f"{best['up_prec']:>8.4f}  {best['up_rec']:>7.4f}  "
            f"{best['dn_prec']:>8.4f}  {best['dn_rec']:>7.4f}  "
            f"{cfg['loss_type']:>14}  {ok}"
        )
    print("=" * 105)
    print(f"\n  Total study time : {study_elapsed:.1f}s  ({study_elapsed / 60:.1f} min)")
    print(f"  Completed        : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
