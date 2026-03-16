# Project Summary: LSTM Triple-Barrier Model
**Binance 5-min Bars · 3-Class Softmax · Local Memory Features**

> This document is a full record of all design decisions, rationale, and conclusions
> reached in prior conversations. Use this as context when starting a new session.

---

## 1. Problem Setup

- **Exchange:** Binance (crypto, spot or futures)
- **Timeframe:** 5-minute bars
- **Goal:** Predict whether price will hit an upper barrier, lower barrier, or time out within the next 3 bars (~15 minutes)

### Triple-Barrier Label

| Label | Meaning | Original value |
|---|---|---|
| 1 | Upper barrier hit | +1 |
| 0 | Timeout — neither barrier hit within 3 bars | 0 |
| -1 | Lower barrier hit | -1 |

- **Barrier size:** ±1× ATR_42 (42-bar ATR)
```
SMA gives cleaner regime detection signals (sharper transitions)
RMA gives better regime context (where you've been recently)
Using both as features can be useful — their ratio SMA/RMA is itself a regime indicator (>1 = volatility increasing, <1 = fading)
```
- **Timeout:** 3 bars
- **Label is kept as-is** — no modification to the labelling scheme

### Label Distribution (confirmed on actual dataset)

```
Total labeled  : 229,730
  Up   ( 1)    : 76,565  (33.3%)
  Down (-1)    : 78,157  (34.0%)
  Timeout (0)  : 75,008  (32.7%)
  NaN (warmup) : 41
```

Classes are near-perfectly balanced. This is a critical finding that shaped all downstream architecture decisions.

---

## 2. Key Design Decisions & Rationale

### 2.1 Why Fracdiff Was Rejected

Fracdiff was initially considered as a feature but was explicitly rejected. The core misalignment:

- Fracdiff preserves **long memory** — it encodes where price is relative to its 300–500 bar weighted history
- The triple-barrier label resolves in **3 bars (15 minutes)** — it asks a purely local, microstructure question
- The generating process of the label (momentum, order flow, local volatility) has **no meaningful dependence on price from 200+ bars ago**
- Using fracdiff features for a 3-bar label is analogous to using a telescope to answer a microscope question

**Rule established:** All features must encode **local memory ≤ 14 bars**, aligned with the 3-bar label horizon.

### 2.2 Why Two-Stage Model Was Rejected

A two-stage model (Stage 1: trade vs no-trade binary; Stage 2: direction binary) was considered and rejected because:

- The motivation for two-stage was a dominant label=0 class causing directional signal dilution
- **Actual distribution is 33/34/33** — label=0 is not dominant at all
- Two-stage would add complexity (two calibrations, compounded confidence, smaller Stage 2 training set) with no benefit
- **Single 3-class softmax is correct** given this distribution

### 2.3 Why Standard Cross-Entropy (No Focal Loss)

- Focal loss is designed for severe class imbalance (e.g. 80/10/10)
- Distribution is 33/34/33 — no imbalance exists
- Standard cross-entropy trains cleanly without any loss modification or class weighting

---

## 3. Feature Groups (Ablation I–M)

All features encode **local memory ≤ 14 bars**. No fracdiff. No long-run structural features.

### Group I — Rate of Change (Baseline)

| Feature | Formula | Lookback |
|---|---|---|
| ROC_3 | `(close_t / close_{t-3}) − 1` | 3 bars |
| ROC_5 | `(close_t / close_{t-5}) − 1` | 5 bars |
| ROC_10 | `(close_t / close_{t-10}) − 1` | 10 bars |
| MOM_3 | `(close_t − close_{t-3}) / ATR_42` | 3 bars |
| RETURNS_1 | `(close_t / close_{t-1}) − 1` | 1 bar |

**Key decisions for Group I:**
- **MOM_3 denominator is ATR_42** — same ATR used to set the barrier. This is intentional coupling, not leakage. The model learns "how much of the barrier has already been covered." ATR_42 is fully causal (uses bars t-41 through t only).
- **Simple return over log return for RETURNS_1** — at 5-minute bars with typical moves of 0.1–0.5%, simple and log returns are numerically indistinguishable. Log return advantages (time additivity, symmetry) are irrelevant at this timescale. Simple return maintains consistency with ROC_3/5/10 which are all simple returns.

### Group J — AVR Volatility Ratio

| Feature | Formula | Lookback |
|---|---|---|
| AVR_5 | `RMA(Abs(high−low),5)` | 5 bars |
| AVR_14 | `RMA(Abs(high−low),14)` | 14 bars |
| AVR_RATIO | `AVR_5 / AVR_42` | 5 / 42 bars |
| AVR_NORM_ROC | `ROC_3 / AVR_14` | 3 bars |
| RANGE_RATIO | `(high − low) / AVR_14` | 1 bar |

### Group K — RSI & Momentum Oscillators

| Feature | Formula | Lookback |
|---|---|---|
| RSI_5 | Wilder RSI, period=5 | 5 bars |
| RSI_14 | Wilder RSI, period=14 | 14 bars |
| RSI_SLOPE | `RSI_14_t − RSI_14_{t-3}` | 3 bars |
| STOCH_K | `(close − low_5) / (high_5 − low_5) × 100` | 5 bars |
| CCI_5 | `(close − SMA_5) / (0.015 × MAD_5)` | 5 bars |

### Group L — Order Flow Imbalance (Binance kline V-field)

| Feature | Formula | Lookback |
|---|---|---|
| DELTA_1 | `taker_buy_vol − taker_sell_vol` | 1 bar |
| DELTA_3 | `rolling_sum(DELTA_1, 3)` | 3 bars |
| BUY_RATIO | `taker_buy_vol / total_vol` | 1 bar |
| VOL_SPIKE | `total_vol_t / SMA(total_vol, 5)` | 5 bars |
| DELTA_DIV | `sign(ROC_3) ≠ sign(DELTA_3)` | 3 bars |

**Binance data source:** The kline WebSocket stream already includes `V` (taker buy base volume) per bar. No raw tick collection needed. Delta = `V − (total_vol − V)` computed directly from the kline.

### Group M — Distance to Swing High / Low

| Feature | Formula | Lookback |
|---|---|---|
| DIST_HIGH_5 | `(rolling_max(high, 5) − close) / ATR_14` | 5 bars |
| DIST_LOW_5 | `(close − rolling_min(low, 5)) / ATR_14` | 5 bars |
| DIST_HIGH_10 | `(rolling_max(high, 10) − close) / ATR_14` | 10 bars |
| DIST_LOW_10 | `(close − rolling_min(low, 10)) / ATR_14` | 10 bars |
| RANGE_POS | `(close − low_10) / (high_10 − low_10)` | 10 bars |

### Group N(10) — Distance to Swing High / Low on short term range (STR) and intermediate term range (ITR) on 15m timeframe

| Feature | Formula | Lookback |
|---|---|---|
| STR_CONFIRMED | `Binary: has peak/bottom confirm in last 8 bars on 15m tf,` | Dynamic |
| ITR_CONFIRMED | `Binary: has peak/bottom confirm in last 8 bars on 45m tf,` | Dynamic |
| BARSINCE_STR | `Int:Bars since last swing high/low from STR` | Dynamic |
| BARSINCE_ITR | `Int:Bars since last swing high/low from ITR` | Dynamic |
| DIST_HIGH_STR | `(last_str_key_hi_value − close) / ATR_42` | Dynamic |
| DIST_LOW_STR | `(last_str_key_low_value − close) / ATR_42` |  Dynamic |
| DIST_HIGH_ITR | `(last_itr_key_hi_value − close) / ATR_42` |  Dynamic |
| DIST_LOW_ITR | `(close − last_itr_key_low_value) / ATR_42` |  Dynamic |
| RANGE_STR | `(close − last_low_str) / (last_high_str − last_low_str)` | Dynamic |
| RANGE_ITR | `(close − last_low_itr) / (last_high_itr − last_low_itr)` | Dynamic |

---

## 4. Ablation Combination Schedule

| Run | Groups Active | # Features | Primary Question |
|---|---|---|---|
| Run 1 | I only | 5 | Does local ROC alone beat a naive baseline? |
| Run 2 | I + J | 10 | Does volatility context improve ROC? |
| Run 3 | I + J + K | 15 | Do oscillators add signal beyond ROC + vol? |
| Run 4 | I + J + K + L | 20 | How much does order flow lift the ensemble? |
| Run 5 | I + J + K + L + M | 25 | Full local-memory model — marginal gain of structure? |
| Run 6 | L only | 5 | Order flow in isolation — upper bound of signal? |
| Run 7 | I + L | 10 | Minimal viable model: momentum + flow |
| Run 8 | I + J + K + L + N | 30 | Full local-memory model — marginal gain of structure? |
| Run 9 | I + J + K + L + M + N | 35 | Full local-memory model — marginal gain of structure? |

---

## 5. LSTM Architecture

### Resolved Layer Order (PyTorch)

```
Input [batch, seq=20, n_features]
    │
    ├── Input Dropout (p=0.10)
    │
    ├── LSTM (hidden=128, num_layers=2, recurrent_dropout=0.20)
    │
    ├── h_n[-1]  →  take final hidden state  [batch, 128]
    │
    ├── Linear(128 → 64)          ← explicit taper projection
    │
    ├── BatchNorm1d(64)
    │
    ├── Linear(64 → 32) + ReLU
    │
    ├── Dropout(p=0.20)
    │
    └── Linear(32 → 3) + Softmax
```

**Note:** `nn.LSTM(num_layers=2)` in PyTorch shares `hidden_size` across all layers. The 128→64 taper is implemented via an explicit `nn.Linear(128, 64)` applied to `h_n[-1]` after the LSTM stack — not inside `nn.LSTM` itself.

### Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| Sequence length | 20 bars (100 min) | Covers all lookbacks (max=14 bars) with margin |
| LSTM Layer 1 hidden | 128 | Wider first layer; richer temporal patterns |
| LSTM Layer 2 hidden | 64 | Compressed via hidden_proj; reduces overfitting |
| Dense hidden | 32 | Lightweight — most learning done in LSTM |
| Input dropout | 0.10 | Light feature noise; prevents co-adaptation |
| Recurrent dropout | 0.20 | Between stacked LSTM layers (PyTorch nn.LSTM dropout=) |
| Dense dropout | 0.20 | Final regularisation before output |
| Output activation | Softmax 3-class | P(dn), P(no_hit), P(up) |
| Loss | Cross-entropy | No weighting — distribution is balanced |
| Optimiser | Adam lr=1e-3 | ReduceLROnPlateau, factor=0.5, patience=5 |
| Batch size | 512 | Stable given 229k samples |
| Sequence stride | 1 bar | Rolling window; maximises training sequences |
| Max epochs | 100 | With early stopping |
| Early stopping patience | 10 | Monitor val loss; restore best weights |
| Gradient clipping | max_norm=1.0 | Critical for LSTM stability |

### Weight Initialisation

- LSTM input weights: Xavier uniform
- LSTM recurrent weights: Orthogonal initialisation
- Forget gate bias: Set to 1.0 (helps long sequence learning)
- Dense weights: Xavier uniform, zero bias

---

## 6. Data Pipeline

### Normalisation

```python
# Step 1 — ATR normalisation (where applicable, e.g. MOM_3)
MOM_3 = (close_t - close_{t-3}) / ATR_42

# Step 2 — Z-score standardisation (ALL features)
scaler = StandardScaler()
X_train = scaler.fit_transform(...)   # fit on train only
X_val   = scaler.transform(...)       # transform only
X_test  = scaler.transform(...)       # transform only

# Step 3 — Winsorise after Z-scoring
X = np.clip(X, -3, 3)   # or -5, 5 depending on tolerance
```

**Never refit scaler on val or test.** Save scaler alongside model weights for inference.

### Train / Val / Test Split

**Temporal split only — never random shuffle.**

```
Train : first 70%  → ~160,800 bars
Val   : next  15%  → ~34,400  bars
Test  : final 15%  → ~34,400  bars
```

Random shuffle causes look-ahead leakage — future bars bleed into training sequences.

### Label Encoding for PyTorch

```python
# Original → PyTorch class index
-1  →  0   (down barrier hit)
 0  →  1   (timeout / no hit)
+1  →  2   (up barrier hit)
```

---

## 7. Inference & Position Sizing

Conviction margin logic — uses all three softmax outputs, not just argmax:

```python
p_dn, p_no_hit, p_up = model.predict(x)

conviction_up = p_up - max(p_no_hit, p_dn)
conviction_dn = p_dn - max(p_no_hit, p_up)

if conviction_up > threshold:       # e.g. 0.20
    size = base_size * conviction_up    # scale by conviction
elif conviction_dn > threshold:
    size = base_size * conviction_dn
else:
    pass  # no trade — model is uncertain
```

This naturally abstains when the model is split across classes, without needing a separate abstain mechanism.

---

## 8. Evaluation Metrics

| Metric | Target | Priority | Notes |
|---|---|---|---|
| F1 macro | Maximise | ★★★★★ | Primary — averaged across all 3 classes |
| MCC | Maximise | ★★★★☆ | Robust to near-balanced distributions |
| Precision (up/dn) | Maximise | ★★★★☆ | False signals are costly in live trading |
| Recall (up/dn) | > 0.50 | ★★★★☆ | Model must not collapse to no-hit prediction |
| AUC-ROC (OvR) | > 0.60 | ★★★☆☆ | One-vs-rest per class; threshold-agnostic |
| Calibration ECE | < 0.05 | ★★★☆☆ | Required — softmax outputs drive position sizing |
| Confusion matrix | Inspect | ★★★★★ | Watch up/dn confusion specifically |

### Calibration

Softmax outputs drive position sizing — calibration is mandatory:

```python
# Temperature scaling on validation set
# Tune scalar T to minimise ECE: divide logits by T before softmax
# Fit T on val set only, apply at inference
```

---

## 9. Files Produced

| File | Description |
|---|---|
| `lstm_ablation_study.md` | Full ablation design — Groups I–M, combination schedule, metrics |
| `lstm_architecture.md` | Revised architecture document with resolved layer order |
| `lstm_model.py` | PyTorch implementation — model, trainer, predictor, smoke test |
| `project_summary.md` | This file |

---

## 10. Decisions Still Open / Next Steps

- Run ablation schedule (Runs 1–7) and compare F1 macro across groups
- Determine which feature groups make the final model based on ablation results
- Hyperparameter tuning (hidden units, dropout, learning rate) after group selection
- Calibration (temperature scaling) on val set after final model training
- Conviction threshold tuning based on precision/recall trade-off on test set

---

> **Hard rules established in this project:**
> - No fracdiff or any feature with lookback > 14 bars
> - No random shuffle on any split — temporal order must be preserved always
> - Scaler fit on train set only — never refit on val or test
> - Single 3-class softmax — two-stage model was considered and rejected
> - All output files in Markdown format (.md)
