# Ablation
- while Polymarket pulled **last price** directly from **Chainlink**, It could not intuitively fetched historical data since it use **RoundId** instead of **timestamp** see [ChainlinkData](../../../../journal/reference/sampleChainlinkData.md)
- To obtain ohlcv and analyse historical data, I picked Binance since their API are open.

---

### LSTM Feature Ablation Study
**Triple-Barrier Label (ATR14x3=42 on 5m)· Local Memory Feature Groups · Binance 5-min Bars**

---

Each ablation group (I–M) is a self-contained feature set designed to answer the core question: *"does price hit ±ATR42 within the next 3 bars?"* Groups are trained independently and then combined incrementally to isolate each feature family's marginal contribution. All features encode local memory (≤14 bars) aligned with the 3-bar label horizon.

---

### Group I — Rate of Change
*Momentum over 3–10 bars | Baseline group*

| Feature | Lookback | Formula / Source | Why it matters for 3-bar label | Dtype |
|---|---|---|---|---|
| ROC_3 | 3 bars | `(close_t / close_{t-3}) − 1` | Direct 1-to-1 alignment with label horizon; captures the impulse already in motion | float |
| ROC_5 | 5 bars | `(close_t / close_{t-5}) − 1` | Slightly longer impulse; smooths single-bar noise while still local | float |
| ROC_10 | 10 bars | `(close_t / close_{t-10}) − 1` | Upper bound of 'local'; tests whether the last 2 candles add signal | float |
| MOM_3 | 3 bars | `close_t − close_{t-3}` (in price) | Raw price displacement — normalise by ATR before feeding to LSTM | float |
| RETURNS_1 | 1 bar | `(close_t / close_{t-1}) − 1` | Most recent 1-bar return; captures the last candle's directionality | float |

> **Hypothesis:** Recent momentum predicts continuation into the barrier.
> **Expected signal:** ROC_3 will have highest feature importance; ROC_10 marginal.
> **Ablation question:** Does adding ROC_10 improve recall on the minority class vs ROC_3 alone?

---

### Group J — ATR Volatility Ratio
*Volatility regime relative to barrier size | Scale-normalisation group*

| Feature | Lookback | Formula / Source | Why it matters for 3-bar label | Dtype |
|---|---|---|---|---|
| ATR_5 | 5 bars | `RMA of |high−low|, period=5` | Short-run volatility; adapts faster than ATR_14 to regime shifts | float |
| ATR_14 | 14 bars | `RMA of |high−low|, period=14` | Standard ATR used to SET the barrier — model needs to see what it's predicting against | float |
| ATR_RATIO | 5 / 42 | `ATR_5 / ATR_42` | Is current vol expanding vs. the long-run norm? Ratio > 1 = hot market | float |
| ATR_NORM_ROC | 3 bars | `ROC_3 / ATR_14` | ATR-normalised return; removes scale so LSTM sees % of barrier already covered | float |
| RANGE_RATIO | 1 bar | `(high − low) / ATR_14` | Single-bar range vs. barrier: >1 means price already moved a full barrier in this candle | float |

> **Hypothesis:** Model needs ATR_14 context to understand barrier difficulty; ATR_RATIO captures regime.
> **Expected signal:** ATR_RATIO and ATR_NORM_ROC will reduce false positives in low-vol regimes.
> **Ablation question:** Does removing ATR_RATIO degrade precision in low-vol windows specifically?

---

### Group K — RSI & Momentum Oscillators
*Overbought/oversold + momentum confirmation | Oscillator group*

| Feature | Lookback | Formula / Source | Why it matters for 3-bar label | Dtype |
|---|---|---|---|---|
| RSI_5 | 5 bars | `Wilder RSI, period=5` | Fast RSI; sensitive to local exhaustion within the 3-bar horizon | float [0,100] |
| RSI_14 | 14 bars | `Wilder RSI, period=14` | Standard RSI; >70 or <30 flags momentum extremes approaching barrier | float [0,100] |
| RSI_SLOPE | 3 bars | `RSI_14_t − RSI_14_{t-3}` | Is momentum accelerating or decelerating? Slope often leads price | float |
| STOCH_K | 5 bars | `%K = (close − low_5) / (high_5 − low_5) × 100` | Position within recent range; similar to RSI but range-anchored | float [0,100] |
| CCI_5 | 5 bars | `CCI = (close − SMA_5) / (0.015 × MAD_5)` | Extreme values (±100) signal potential reversal | float |

> **Hypothesis:** Oscillators help the model avoid chasing exhausted moves.
> **Expected signal:** RSI_SLOPE will carry more signal than RSI level alone for predicting near-term barrier hits.
> **Ablation question:** Group K vs Group I — does adding oscillators improve F1 beyond pure ROC features?

---

### Group L — Order Flow Imbalance
*Taker buy/sell pressure | Binance kline V-field group*

| Feature | Lookback | Formula / Source | Why it matters for 3-bar label | Dtype |
|---|---|---|---|---|
| DELTA_1 | 1 bar | `taker_buy_vol − taker_sell_vol` | Per-bar net aggressor pressure; most direct signal of who is controlling price | float |
| DELTA_3 | 3 bars | `rolling_sum(DELTA_1, 3)` | Cumulative delta over label horizon; are buyers/sellers winning the short battle? | float |
| BUY_RATIO | 1 bar | `taker_buy_vol / total_vol` | 0.5 = balanced; >0.65 = aggressive buying; scale-invariant version of delta | float [0,1] |
| VOL_SPIKE | 5 bars | `total_vol_t / SMA(total_vol, 5)` | Is this bar abnormally active? Spikes precede barrier breaks | float |
| DELTA_DIV | 3 bars | `sign(ROC_3) ≠ sign(DELTA_3)` | Delta divergence: price up but net selling = exhaustion flag | binary {0,1} |

> **Hypothesis:** Order flow is the most causally direct predictor of near-term price movement.
> **Expected signal:** Group L alone should outperform Groups I+J+K combined on directional accuracy.
> **Ablation question:** Is the marginal gain of L over I+J+K worth the Binance API dependency in production?

---

### Group M — Distance to Swing High / Low
*Structural support / resistance within local window | Microstructure group*

| Feature | Lookback | Formula / Source | Why it matters for 3-bar label | Dtype |
|---|---|---|---|---|
| DIST_HIGH_5 | 5 bars | `(rolling_max(high, 5) − close) / ATR_14` | Gap to 5-bar swing high; a small gap means barrier is near a resistance level | float ≥ 0 |
| DIST_LOW_5 | 5 bars | `(close − rolling_min(low, 5)) / ATR_14` | Gap to 5-bar swing low; floor proximity affects downside barrier probability | float ≥ 0 |
| DIST_HIGH_10 | 10 bars | `(rolling_max(high, 10) − close) / ATR_14` | Slightly wider swing high; captures the last two 5-min impulse peaks | float ≥ 0 |
| DIST_LOW_10 | 10 bars | `(close − rolling_min(low, 10)) / ATR_14` | Symmetric counterpart for support zone over 10-bar lookback | float ≥ 0 |
| RANGE_POS | 10 bars | `(close − low_10) / (high_10 − low_10)` | Where is close within the 10-bar range? 0 = bottom, 1 = top; proxy for mean-reversion risk | float [0,1] |

> **Hypothesis:** Price near a swing high faces resistance; proximity encodes barrier reachability.
> **Expected signal:** DIST_HIGH_5 will negatively correlate with upper-barrier hits; model uses it as a ceiling signal.
> **Ablation question:** Does Group M add signal beyond ATR_RATIO (J)? Both encode "how far can price go."

---

### Incremental Combination Schedule

| Run | Feature Groups Active | # Features | Primary Question |
|---|---|---|---|
| Run 1 | I only | 5 | Does local ROC alone beat a naive baseline? |
| Run 2 | I + J | 10 | Does volatility context improve ROC? |
| Run 3 | I + J + K | 15 | Do oscillators add signal beyond ROC + vol? |
| Run 4 | I + J + K + L | 20 | How much does order flow lift the ensemble? |
| Run 5 | I + J + K + L + M | 25 | Full local-memory model — marginal gain of structure? |
| Run 6 | L only | 5 | Order flow in isolation — upper bound of predictive signal? |
| Run 7 | I + L | 10 | Minimal viable model: momentum + flow |

Let's dive in. Start at [data science](../dataScience)

---

### Evaluation Metrics per Run

| Metric | Target | Priority | Why | Label Class |
|---|---|---|---|---|
| F1 (macro) | Maximise | ★★★★★ | Balances precision/recall across imbalanced classes | All |
| Precision (up/dn) | Maximise | ★★★★☆ | Reduces false barrier signals that cause bad trades | 1, -1 |
| Recall (up/dn) | > 0.50 | ★★★★☆ | Ensures model finds real barrier hits, not just abstains | 1, -1 |
| AUC-ROC | > 0.60 | ★★★☆☆ | Threshold-agnostic; useful for comparing group runs | Binary per class |
| Calibration ECE | < 0.05 | ★★★☆☆ | Softmax outputs will be used for position sizing | All |
| MCC | Maximise | ★★★★☆ | Matthews Corr. Coeff — robust to class imbalance | All |

---

### Drafted LSTM Architecture
*shared architecture across all runs*

| Hyperparameter | Value | Rationale |
|---|---|---|
| Sequence length | 20 bars (100 min) | Provides enough context for all lookbacks (max = 14 bars) |
| LSTM layers | 2 stacked | Sufficient depth; 3+ risks overfitting on 5-min crypto data |
| Hidden units | 64 | Start conservative given small feature count; tune per run |
| Dropout | 0.2 recurrent | Recurrent dropout is more effective than input dropout for LSTM |
| Output | Softmax 3-class | Classes: upper hit (1), no hit (0), lower hit (−1) |
| Loss | Focal loss γ=2 | Addresses class imbalance; down-weights easy negatives |
| Sequence stride | 1 bar | Rolling window; maximises training samples |
| Input normalisation | Z-score per feature | Per-feature standardisation; refit on train set only |

---

**Note:** I found out there were 2 ways to train the model using same data which depends on the label distribution. See [Training Note](../../../../journal/trainData/trainingNote.md)

**Note2:** 
for a perfect balanced Ternary label
```
|Issue|Conclusion|
|--|--|
|two-stage model needed? |No — single 3-class softmax is fine|
|focal loss needed?|No — standard cross-entropy works|
|class weighting needed?|No — already balanced|
|oversampling needed?|Nope|
```


