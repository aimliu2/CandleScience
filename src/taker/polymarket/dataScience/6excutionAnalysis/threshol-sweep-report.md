# Run 2 — Conviction Threshold Sweep Summary
**Model: Run52_x_final.pt · Test window: 2025-09-02 → 2026-03-11**

---

## Model & Test Set

| Property | Value |
|---|---|
| Model file | Run52_x_final.pt |
| Val F1 | 0.3979 |
| Test F1 | 0.3935 |
| Regime gap | 0.0044 |
| Test sequences | 54,569 |
| Test window | 2025-09-02 → 2026-03-11 (189 days) |
| Features | 22 (Groups I+J+K+L+M, BUY_RATIO and DELTA_DIV excluded) |

---

## Regime Distribution Across Test Window

| Regime | Sequences | % |
|---|---|---|
| BEARISH | 21,567 | 39.5% |
| ACCUMULATION | 13,207 | 24.2% |
| DISTRIBUTION | 7,192 | 13.2% |
| BULLISH | 8,337 | 15.3% |
| CORRECTION | 3,237 | 5.9% |
| RECOVERY | 1,029 | 1.9% |
| UNKNOWN | 0 | 0.0% |

The test window is bear-dominant. BEARISH alone covers 39.5%. DISTRIBUTION + CORRECTION + BEARISH combined = 58.6% of sequences in bearish-group regimes. RECOVERY is the smallest regime at only ~4.5 weeks.

---

## Key Finding 1 — LONG Signals Have Structurally Collapsed

The model cannot generate meaningful UP conviction anywhere in the test window.

| Metric | Value |
|---|---|
| conv_up max (all sequences) | 0.0531 |
| conv_up mean (all sequences) | negative |
| LONG signals above thr=0.06 | 0 |
| Mean P(UP) across all bars | 0.3366 |
| Mean P(DN) across all bars | 0.3596 |

The 4 LONG signals that exist above thr=0.05 all appear on 2025-09-13 and 2025-09-19 — the first two weeks of the test window. After September 19th there are zero LONG signals for the remaining 173 days.

**Root cause confirmed — structural softmax bias toward DOWN.** The model trained on 2.7× data with heavy 2021–2022 bear market coverage (~158,000 BEARISH bars) has learned bearish feature patterns more strongly than bullish ones. This is a training data regime imbalance problem, not a code error.

Evidence from the directional bias check:

| Regime | P(UP) > P(DN) % | Expected if unbiased |
|---|---|---|
| BULLISH | 7.6% | ~33% |
| ACCUMULATION | 27.5% | ~33% |
| ALL bars | 17.7% | ~33% |

Even in bars the regime classifier explicitly labels BULLISH, the model calls DOWN more confidently than UP in 92.4% of cases. P(UP) mean in BULLISH = 0.327 vs P(DN) mean = 0.363 — a persistent 0.036 gap favouring DOWN throughout. First half vs second half of BULLISH windows is nearly identical, eliminating LSTM hidden state lag as a cause.

**LONG signals are not usable in Run 2. The model is SHORT-only on this test window.**

---

## Key Finding 2 — SHORT Precision Is Real and Regime-Dependent

SHORT signals fire across all regimes with meaningful precision above baseline (~0.33).
Precision improves as threshold rises in most regimes.

### SHORT Precision by Regime at Key Thresholds

| Regime | thr=0.05 N | thr=0.05 Prec_DN | thr=0.07 N | thr=0.07 Prec_DN | thr=0.10 N | thr=0.10 Prec_DN |
|---|---|---|---|---|---|---|
| CORRECTION | 428 | 0.4813 | 155 | 0.5097 | 14 | 0.5714 |
| RECOVERY | 113 | 0.4779 | 49 | 0.6122 | 22 | 0.5455 |
| ACCUMULATION | 1,124 | 0.4511 | 568 | 0.4701 | 187 | 0.5187 |
| BULLISH | 1,592 | 0.4466 | 1,082 | 0.4603 | 543 | 0.4696 |
| BEARISH | 2,123 | 0.4249 | 1,143 | 0.4584 | 472 | 0.4682 |
| DISTRIBUTION | 765 | 0.3895 | 452 | 0.4093 | 182 | 0.4670 |

**CORRECTION** has the strongest and most consistent SHORT precision. Rises cleanly from 0.48 at thr=0.05 to 0.57 at thr=0.10.

**RECOVERY** has the highest peak SHORT precision (0.61 at thr=0.07) — counter-intuitive but supported by the data. The 7-regime classifier's precise RECOVERY definition (RSI crossing 48↑, ATR declining, 1H EMA crossing bullish) excludes dead-cat bounces, making real RECOVERY a cleaner environment than expected.

**DISTRIBUTION** is the weakest regime for SHORT signals. Precision stays below 0.42 until thr=0.09, reflecting conflicting signals during the topping phase. Only usable at high thresholds where signal count thins significantly.

---

## Key Finding 3 — Signal Count Drops Sharply Above thr=0.10

The SHORT signal pool is concentrated at low thresholds. Above thr=0.10 most regimes
have fewer than 20 signals — too thin for reliable precision estimates.

| Regime | Practical threshold range | Reasoning |
|---|---|---|
| CORRECTION | 0.05 – 0.10 | 428 signals at 0.05, 14 at 0.10 — collapses fast |
| RECOVERY | 0.05 – 0.09 | 113 at 0.05, 26 at 0.09 — very small regime |
| ACCUMULATION | 0.05 – 0.13 | Large pool, holds precision well |
| BULLISH | 0.05 – 0.14 | Largest pool, most stable across thresholds |
| BEARISH | 0.05 – 0.13 | Large pool, consistent if slightly lower precision |
| DISTRIBUTION | 0.07 – 0.12 | Below 0.07 precision is weak; above 0.12 count collapses |

---

## Proposed Gate Actions — SHORT Only

Based on the regime precision breakdown, the following gate actions are proposed
for SHORT signals. LONG signals are suppressed across all regimes pending model retraining.

| Regime | SHORT gate | Recommended thr | Reasoning |
|---|---|---|---|
| CORRECTION | ✅ Allow | 0.05 – 0.07 | Strongest SHORT regime, clean edge |
| RECOVERY | ✅ Allow | 0.06 – 0.08 | Surprisingly strong SHORT precision |
| ACCUMULATION | ✅ Allow | 0.07 – 0.10 | Solid precision, large pool |
| BULLISH | ✅ Allow | 0.07 – 0.10 | Solid precision, largest pool |
| BEARISH | ✅ Allow | 0.07 – 0.09 | Largest pool, slightly lower precision ceiling |
| DISTRIBUTION | ⚠️ Allow with caution | 0.09 – 0.11 | Weak edge below 0.09, thin above 0.12 |

---

## What This Means for Next Steps

**Immediate — Path A (SHORT-only execution):**

Gate actions above are based on precision only. Before locking thresholds,
the forward return analysis (N=3 fixed bar exit, threshold search per regime)
needs to confirm that SHORT precision translates into positive edge on actual
price movement. Precision above baseline is necessary but not sufficient —
edge depends on avg_win / avg_loss ratio at the exit bar.

**Future — Fix LONG signal collapse (Run 3):**

The LONG-side failure is a training data regime imbalance problem. Remediation
options in order of complexity:
1. Regime-balanced sampling — undersample BEARISH bars during training
2. Class-conditional loss weighting by regime
3. Separate LONG and SHORT models trained on regime-filtered subsets

Do not revisit LONG signals until the model is retrained with one of the above corrections.

---

## Open Items Before Gate Actions Are Locked

| Item | Status |
|---|---|
| Forward return analysis per regime (N=3, threshold search) | Not yet run |
| Signal clustering per regime | Not yet run |
| Hysteresis rule for regime switching | Not yet defined |
| LONG signal remediation approach for Run 3 | Decision pending |