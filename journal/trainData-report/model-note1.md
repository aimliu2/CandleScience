# Revised LSTM Architecture
see also [Project summary](../general/20260309_proj_summary.md)
**Triple-Barrier Label · 3-Class Softmax · Binance 5-min Bars**

---

## Why Single 3-Class Model

Label distribution confirmed near-perfect balance (33.3% / 34.0% / 32.7%). This eliminates every
original motivation for two-stage design — no class imbalance, no dominant timeout class, no
directional signal dilution. Standard 3-class softmax with cross-entropy trains cleanly as-is.

---

## Architecture Overview

```
Input Sequence
[batch, 20 timesteps, N features]
         │
         ▼
┌─────────────────────┐
│   Input Dropout     │  p = 0.10
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   LSTM Layer 1      │  hidden = 128, return_sequences = True
│   Recurrent Dropout │  p = 0.20
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   LSTM Layer 2      │  hidden = 64, return_sequences = False
│   Recurrent Dropout │  p = 0.20
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   BatchNorm         │  stabilises activations between LSTM and FC
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Dense 32, ReLU    │  lightweight projection layer
│   Dropout p = 0.20  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Dense 3, Softmax  │  P(up=1), P(no hit=0), P(dn=-1)
└────────┬────────────┘
         │
         ▼
  [P(up), P(no hit), P(dn)]
```

```
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
```

---

## Hyperparameters
**the params are not sorted by this order - This is table of contents**

| Hyperparameter | Value | Rationale |
|---|---|---|
| Sequence length | 20 bars (100 min) | Covers all feature lookbacks (max = 14 bars) with margin |
| LSTM Layer 1 hidden | 128 | Wider first layer captures richer temporal patterns |
| LSTM Layer 2 hidden | 64 | Compresses representation; reduces overfitting |
| Dense hidden | 32 | Lightweight — most learning already done in LSTM layers |
| Input dropout | 0.10 | Light noise on features; prevents co-adaptation |
| Recurrent dropout | 0.20 | Applied inside LSTM gates; more effective than output dropout |
| Dense dropout | 0.20 | Final regularisation before classification head |
| Output activation | Softmax | Produces proper probability distribution across 3 classes |
| Loss | Cross-entropy | Valid given balanced distribution — no weighting needed |
| Optimiser | Adam | lr = 1e-3, reduce on plateau |
| Batch size | 512 | Large batch stable given 229k samples; speeds convergence |
| Sequence stride | 1 bar | Rolling window; maximises training sequences |
| Input normalisation | Z-score per feature | Fit on train set only; applied to val/test |
| Max epochs | 100 | With early stopping — actual convergence typically 20–40 |
| Early stopping | patience = 10 | Monitor val loss; restore best weights |

---

## Loss Function

Standard categorical cross-entropy — no modification needed:

```python
loss = -Σ y_true * log(P_softmax)
```

**No focal loss.** No class weighting. Your distribution does not require it.

If future resampling or a different barrier changes the distribution past 60/20/20,
revisit focal loss with γ = 2 at that point.

---

## Input Normalisation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, N)).reshape(-1, 20, N)
X_val   = scaler.transform(X_val.reshape(-1, N)).reshape(-1, 20, N)
X_test  = scaler.transform(X_test.reshape(-1, N)).reshape(-1, 20, N)
```

Fit **only on train**. Never refit on val or test. Save scaler alongside model weights for inference.

---

## Train / Val / Test Split

**Use temporal split — never random shuffle.**

```
Total bars: 229,730

Train : first 70%  → ~160,800 bars
Val   : next  15%  → ~34,400  bars   ← hyperparameter tuning, early stopping
Test  : final 15%  → ~34,400  bars   ← held out, touch once at end
```

Random shuffle would cause **look-ahead leakage** — future bars bleeding into training sequences.

---

## Inference & Position Sizing

At inference, all three softmax outputs are used — not just the argmax:

```python
p_up, p_no_hit, p_dn = model.predict(x)

# Conviction margin — how much does the winner beat the field?
conviction_up = p_up - max(p_no_hit, p_dn)
conviction_dn = p_dn - max(p_no_hit, p_up)

# Trade trigger
if conviction_up > threshold:     # e.g. threshold = 0.20
    size = base_size * conviction_up   # scale position by conviction
elif conviction_dn > threshold:
    size = base_size * conviction_dn
else:
    pass  # no trade
```

This replaces a hard argmax with a **conviction-weighted signal** — naturally abstaining
when the model is uncertain across all three classes.

---

## Evaluation Metrics

| Metric | Target | Priority | Notes |
|---|---|---|---|
| F1 macro | Maximise | ★★★★★ | Primary metric — average across all 3 classes |
| MCC | Maximise | ★★★★☆ | Robust multi-class metric; handles near-balance well |
| Precision (up / dn) | Maximise | ★★★★☆ | False signals are costly in live trading |
| Recall (up / dn) | > 0.50 | ★★★★☆ | Model must not collapse to predicting no-hit only |
| AUC-ROC (OvR) | > 0.60 | ★★★☆☆ | One-vs-rest per class; threshold-agnostic |
| Calibration ECE | < 0.05 | ★★★☆☆ | Required if softmax outputs used for position sizing |
| Confusion matrix | Inspect | ★★★★★ | Watch for up/dn confusion specifically — costly errors |

---

## Calibration

Because softmax outputs drive position sizing, calibration is not optional:

```python
from sklearn.calibration import CalibratedClassifierCV

# After training, fit temperature scaling on validation set
# Temperature scaling: divide logits by T before softmax
# Tune T to minimise ECE on val set
```

A model that outputs `P(up) = 0.80` should be right ~80% of the time.
Without calibration, LSTM softmax outputs are typically overconfident.

---

## What Was Removed vs Original Design

| Original | Revised | Reason |
|---|---|---|
| Two-stage model | Single model | Label distribution is balanced — no need |
| Focal loss γ=2 | Cross-entropy | No class imbalance |
| Class weighting | None | 33/34/33 split needs no correction |
| Softmax 3-class | Kept | Still correct output layer |
| Hidden units = 64 flat | 128 → 64 tapered | Better representation capacity in Layer 1 |
| No BatchNorm | Added between LSTM and Dense | Stabilises training with deeper architecture |
| No input dropout | Added p=0.10 | Light regularisation on feature inputs |

---

> **Note:** All five feature groups (I–M) from the ablation study feed into this single architecture.
> Run the ablation schedule first to confirm which groups to include before full training.
> The architecture above assumes the full 25-feature set (Groups I+J+K+L+M, Run 5).