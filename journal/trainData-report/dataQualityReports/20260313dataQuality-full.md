# Run2 — Data Quality Summary
**Binance 5-min Bars · Triple-Barrier Label · 546,413 Labeled Samples**

> Reference document for Run2 data quality verification.
> Records all test results, decisions, and changes from Run1.
> Use this when interpreting training results or debugging pipeline issues.

---

## Dataset Summary

| Property | Value |
|---|---|
| Date range | 2021-11-01 to 2026-03-11 |
| Total labeled samples | 546,413 |
| After ATR_42 warmup drop (41 rows) | 546,372 |
| After rolling Z-score warmup drop (500 rows) | 545,872 |
| Cleaned rows entering pipeline | 546,381 |
| Bar interval | 5 minutes |
| Exchange | Binance |

### Label Distribution

```
Up      (+1) : 174,148  (31.9%)
Down    (-1) : 179,672  (32.9%)
Timeout ( 0) : 192,593  (35.2%)
NaN warmup  :      41
```

Near-balanced. Timeout is the plurality at 35.2% but the 3.3pp spread does not warrant
class weighting or focal loss. CrossEntropyLoss with equal weights is appropriate.

### Market Regimes Covered

All five regimes are present in this dataset across the full 4.5-year window:
- Accumulation (2021–2022 base)
- Bull run (2023–2024 uptrend)
- Bear market (2022 drawdown)
- Correction (late 2024 – early 2025)
- Recovery (2025–2026)

This is the key structural improvement over Run1 (~800 days, incomplete regime coverage).

---

## Temporal Split — 80/10/10

| Split | Rows | Boundary |
|---|---|---|
| Train | 437,104 | index 0 – 437,103 |
| Val | 54,638 | index 437,104 – 491,741 |
| Test | 54,640 | index 491,742 – 546,381 |

### Why 80/10/10 over 70/15/15

Run1 used 70/15/15 because the test set needed to capture a deeper correction regime.
With full regime coverage now distributed across all 546k bars, that concern is resolved.
80/10/10 maximises training data (437k vs 160k in Run1) without sacrificing meaningful
val and test set sizes.

### Val Set Bias — Reduced

Run1 val set was structurally locked to peak-bull (99k–126k price range). With full
regime coverage, the 80/10/10 val boundary falls in mixed conditions. Check the actual
val price range and label distribution after splitting to confirm. Early stopping should
be more reliable than Run1 as a result.

---

## Feature Set

**23 features retained. 1 dropped (BUY_RATIO). 2 ablation candidates (Run10 vs Run11).**

All features are Groups I–M. Group N excluded — confirmed in Run1 to not add
generalising signal across regime shifts.

### Final Feature List

| Group | Feature | Lookback | Normalisation | Status |
|---|---|---|---|---|
| I | ROC_3 | 3 bars | Global Z | ✓ Retained |
| I | ROC_5 | 5 bars | Global Z | ✓ Retained |
| I | ROC_10 | 10 bars | Global Z | ✓ Retained |
| I | MOM_3 | 3 bars | Global Z | ✓ Retained |
| I | RETURNS_1 | 1 bar | Global Z | ✓ Retained |
| J | ATR_5 | 5 bars | Rolling Z | ✓ Retained |
| J | ATR_14 | 14 bars | Rolling Z | ✓ Retained |
| J | ATR_RATIO | 5/42 bars | Global Z | ✓ Retained |
| J | ATR_NORM_ROC | 3 bars | Global Z | ✓ Retained |
| J | RANGE_RATIO | 1 bar | Global Z | ✓ Retained |
| K | RSI_14 | 14 bars | Global Z | ✓ Retained |
| K | RSI_SLOPE | 3 bars | Global Z | ✓ Retained |
| K | CCI_5 | 5 bars | Global Z | ✓ Retained |
| L | DELTA_1 | 1 bar | Rolling Z | ✓ Retained |
| L | DELTA_3 | 3 bars | Rolling Z | ✓ Retained |
| L | BUY_RATIO | 1 bar | Rolling Z | ✗ Dropped |
| L | DELTA_DIV | 3 bars | Global Z | ⚡ Ablation |
| M | DIST_HIGH_5 | 5 bars | Rolling Z | ✓ Retained |
| M | DIST_LOW_5 | 5 bars | Rolling Z | ✓ Retained ← moved from Global Z |
| M | DIST_HIGH_10 | 10 bars | Rolling Z | ✓ Retained |
| M | DIST_LOW_10 | 10 bars | Rolling Z | ✓ Retained ← moved from Global Z |
| M | RANGE_POS | 10 bars | Rolling Z | ✓ Retained |

### Ablation Runs

| Run | Features | Count | Question |
|---|---|---|---|
| Run10 | All retained, without DELTA_DIV | 21 | Baseline without exhaustion flag |
| Run11 | All retained, with DELTA_DIV | 22 | Does exhaustion flag add signal? |

**Decision rule:** Keep DELTA_DIV if Run11 test F1 > Run10 by more than 0.002,
and regime gap does not widen. Otherwise drop and proceed with 21 features.

---

## Normalisation Map

Pipeline order is fixed. Apply in sequence:

```
Step 1 — Rolling Z-score (window=500)     applied to 10 features
Step 2 — Drop first 500 warmup rows
Step 3 — Temporal split (80/10/10)
Step 4 — Global Z-score (fit on train)    applied to 11/12 features
Step 5 — Winsorise ±3σ                    applied to all features (clamp, not drop)
Step 6 — Build LSTM sequences (seq=20)
```

### Rolling Z-score Features (10)

```
DELTA_1, DELTA_3
ATR_5, ATR_14
STOCH_K
DIST_HIGH_5, DIST_HIGH_10
DIST_LOW_5, DIST_LOW_10      ← moved from Global Z vs Run1
RANGE_POS
```

### Global Z-score Features

```
Run10 (11): ROC_3, ROC_5, ROC_10, MOM_3, RETURNS_1
            ATR_RATIO, ATR_NORM_ROC, RANGE_RATIO
            RSI_14, RSI_SLOPE, CCI_5

Run11 (12): same as Run10 + DELTA_DIV
```

### Key Change from Run1

`DIST_LOW_5` and `DIST_LOW_10` moved from Global Z to Rolling Z.
Reason: KPSS test on the full 546k dataset revealed persistent level shifts
(kpss_stat=2.070 and 2.777 respectively) that were not visible in the smaller
Run1 dataset. After moving to Rolling Z, both pass ADF+KPSS cleanly
(DIST_LOW_5 full: KPSS p=0.1000, DIST_LOW_10 full: KPSS p=0.0702).

---

## Verification Test Results

### Test 1 — Rolling Std Stability (window=500)

Assessed whether rolling std is stable across train/val/test boundaries.

| Feature | Train early | Train mid | Train late | Val | Test | Global std | Verdict |
|---|---|---|---|---|---|---|---|
| DELTA_1 | 58.50 | 71.54 | 41.98 | 34.68 | 35.09 | 29.90 | ✓ Pass |
| DELTA_3 | 114.51 | 135.02 | 83.50 | 71.02 | 71.65 | 58.78 | ✓ Pass |
| BUY_RATIO | 0.084 | 0.082 | 0.131 | 0.163 | 0.158 | 0.044 | ✓ Pass |
| ATR_5 | 69.27 | 25.41 | 68.86 | 82.31 | 88.13 | 45.30 | ✓ Pass |
| ATR_14 | 55.98 | 20.95 | 59.20 | 71.32 | 75.94 | 39.33 | ✓ Pass |
| STOCH_K | 29.07 | 29.10 | 31.31 | 32.23 | 31.24 | 2.04 | ✓ Pass |
| DIST_HIGH_5 | 0.850 | 0.907 | 0.946 | 0.992 | 0.981 | 0.123 | ✓ Pass |
| DIST_HIGH_10 | 1.088 | 1.135 | 1.216 | 1.301 | 1.281 | 0.170 | ✓ Pass |
| RANGE_POS | 0.281 | 0.277 | 0.299 | 0.310 | 0.300 | 0.019 | ✓ Pass |

**ATR_5 and ATR_14 note:** Train_mid collapses to 25.41 / 20.95 — the low-volatility
consolidation period (mid-2023 to early-2024) sitting in the middle of the training window.
Rolling Z-score is handling it correctly. Not actionable.

**Window=500 validated for all rolling features.**

---

### Test 2 — Stationarity (ADF + KPSS, subsample n=10,000)

#### Rolling Z-score features (post-normalisation, train + full segments)

| Feature | Segment | ADF stat | ADF p | ADF | KPSS stat | KPSS p | KPSS | Result |
|---|---|---|---|---|---|---|---|---|
| DELTA_1 | train | -100.55 | 0.0000 | ✓ | 0.040 | 0.1000 | ✓ | PASS |
| DELTA_1 | full | -44.60 | 0.0000 | ✓ | 0.210 | 0.1000 | ✓ | PASS |
| DELTA_3 | train | -99.61 | 0.0000 | ✓ | 0.252 | 0.1000 | ✓ | PASS |
| DELTA_3 | full | -49.83 | 0.0000 | ✓ | 0.328 | 0.1000 | ✓ | PASS |
| BUY_RATIO | train | -100.74 | 0.0000 | ✓ | 0.040 | 0.1000 | ✓ | PASS |
| BUY_RATIO | full | -45.21 | 0.0000 | ✓ | 0.109 | 0.1000 | ✓ | PASS |
| ATR_5 | train | -22.52 | 0.0000 | ✓ | 0.512 | 0.0390 | ✗ | ADF only |
| ATR_5 | full | -15.44 | 0.0000 | ✓ | 0.269 | 0.1000 | ✓ | PASS |
| ATR_14 | train | -23.96 | 0.0000 | ✓ | 0.691 | 0.0144 | ✗ | ADF only |
| ATR_14 | full | -15.83 | 0.0000 | ✓ | 0.289 | 0.1000 | ✓ | PASS |
| STOCH_K | train | -98.55 | 0.0000 | ✓ | 0.108 | 0.1000 | ✓ | PASS |
| STOCH_K | full | -59.24 | 0.0000 | ✓ | 0.185 | 0.1000 | ✓ | PASS |
| DIST_HIGH_5 | train | -99.97 | 0.0000 | ✓ | 0.098 | 0.1000 | ✓ | PASS |
| DIST_HIGH_5 | full | -58.56 | 0.0000 | ✓ | 0.141 | 0.1000 | ✓ | PASS |
| DIST_HIGH_10 | train | -101.27 | 0.0000 | ✓ | 0.219 | 0.1000 | ✓ | PASS |
| DIST_HIGH_10 | full | -101.33 | 0.0000 | ✓ | 0.166 | 0.1000 | ✓ | PASS |
| RANGE_POS | train | -100.45 | 0.0000 | ✓ | 0.160 | 0.1000 | ✓ | PASS |
| RANGE_POS | full | -102.49 | 0.0000 | ✓ | 0.360 | 0.0946 | ✓ | PASS |
| DIST_LOW_5 | train | -51.21 | 0.0000 | ✓ | 0.080 | 0.1000 | ✓ | PASS |
| DIST_LOW_5 | full | -50.21 | 0.0000 | ✓ | 0.258 | 0.1000 | ✓ | PASS |
| DIST_LOW_10 | train | -58.92 | 0.0000 | ✓ | 0.067 | 0.1000 | ✓ | PASS |
| DIST_LOW_10 | full | -101.62 | 0.0000 | ✓ | 0.416 | 0.0702 | ✓ | PASS |

**20/22 segments pass both ADF and KPSS.**
ATR_5 and ATR_14 train segments are ADF-only — same known pattern as Run1.
Both pass cleanly on the full series. Not actionable.

#### Global Z-score features (raw, full series)

| Feature | ADF p | KPSS p | Verdict |
|---|---|---|---|
| ROC_3 | 0.0000 | 0.1000 | PASS |
| ROC_5 | 0.0000 | 0.1000 | PASS |
| ROC_10 | 0.0000 | 0.1000 | PASS |
| MOM_3 | 0.0000 | 0.1000 | PASS |
| RETURNS_1 | 0.0000 | 0.1000 | PASS |
| ATR_RATIO | 0.0000 | 0.1000 | PASS |
| ATR_NORM_ROC | 0.0000 | 0.1000 | PASS |
| RANGE_RATIO | 0.0000 | 0.1000 | PASS |
| RSI_14 | 0.0000 | 0.1000 | PASS |
| RSI_SLOPE | 0.0000 | 0.1000 | PASS |
| CCI_5 | 0.0000 | 0.1000 | PASS |
| VOL_SPIKE | 0.0000 | 0.1000 | PASS |
| DELTA_DIV | 0.0000 | 0.0199 | WARN |
| DIST_LOW_5 (raw) | 0.0000 | 0.0100 | WARN → moved to Rolling Z |
| DIST_LOW_10 (raw) | 0.0000 | 0.0100 | WARN → moved to Rolling Z |

**DELTA_DIV WARN:** KPSS p=0.0199, kpss_stat=0.630. Binary feature — fires when
price direction and delta direction diverge. Frequency varies across regimes (rare in
trending markets, common in choppy corrections). Mild and not actionable on its own.
Ablation (Run10 vs Run11) determines whether it stays.

---

### Test 3 — ACF (lag 1, 3, 5, 10, 20)

All features show ACF consistent with their construction lookback. No feature
exceeds the seq=20 LSTM window in a problematic way.

| Category | Features | Max meaningful lag | Assessment |
|---|---|---|---|
| Near white noise | RETURNS_1, DELTA_DIV, DELTA_1, BUY_RATIO | ≤1 | Clean |
| Short memory | ROC_3, MOM_3, ATR_NORM_ROC, CCI_5, VOL_SPIKE, STOCH_K | 2–4 | Expected |
| Moderate memory | ROC_5, RSI_SLOPE, DELTA_3, DIST_HIGH_5, DIST_LOW_5 | 4–5 | Expected |
| Longer memory | ROC_10, DIST_HIGH_10, DIST_LOW_10, RANGE_POS, ATR_RATIO | 6–13 | Structural |
| High persistence | RSI_14, ATR_5, ATR_14 | 20+ | By construction |

**ATR_5 and ATR_14 note:** Extremely persistent ACF (ATR_14 lag20=+0.744). This is
the RMA smoothing nature of ATR. LSTM handles autocorrelated inputs by design.
High ACF here is informative about volatility regime persistence, not a problem.

---

### Test 4 — VIF (Variance Inflation Factor)

Drop threshold: VIF > 30. No feature exceeds this.

| Feature | VIF | Status |
|---|---|---|
| RANGE_POS | 20.35 | Watch — correlated cluster |
| STOCH_K | 20.29 | Watch — elevated vs Run1 (was ~9) |
| RSI_SLOPE | 15.44 | Watch |
| MOM_3 | 15.40 | Watch |
| ATR_5 | 11.20 | Watch |
| DIST_HIGH_10 | 10.04 | Watch |
| ATR_14 | 10.01 | Watch |
| DIST_HIGH_5 | 9.99 | Acceptable |
| DIST_LOW_5 | 9.72 | Acceptable |
| DIST_LOW_10 | 9.56 | Acceptable |
| CCI_5 | 8.77 | Acceptable |
| ROC_3 | 6.14 | Acceptable |
| RSI_14 | 4.62 | Good |
| ATR_NORM_ROC | 4.32 | Good |
| ROC_5 | 4.24 | Good |
| ROC_10 | 3.36 | Good |
| DELTA_3 | 2.63 | Good |
| DELTA_1 | 2.58 | Good |
| RETURNS_1 | 2.33 | Good |
| RANGE_RATIO | 2.32 | Good |
| ATR_RATIO | 2.15 | Good |
| VOL_SPIKE | 1.83 | Good |
| BUY_RATIO | 1.64 | — Dropped |
| DELTA_DIV | 1.04 | Good |

**STOCH_K VIF jump from ~9 (Run1) to 20.29 (Run2):** Caused by DIST_LOW_5
moving to Rolling Z, exposing shared range-position information between STOCH_K,
DIST_HIGH_5, and DIST_LOW_5. No action — 20.29 is well below the drop threshold.

**No features dropped on VIF grounds.**

---

### Test 5 — Mutual Information (k=3 and k=10)

| Feature | MI k=3 | MI k=10 | Stability | Notes |
|---|---|---|---|---|
| ATR_RATIO | 0.0346 | 0.0347 | Stable | Highest signal — same as Run1 |
| DELTA_DIV | 0.0190 | 0.0049 | Unstable | 4× drop — localised binary signal |
| RSI_14 | 0.0188 | 0.0187 | Stable | Strong and consistent |
| MOM_3 | 0.0172 | 0.0173 | Stable | Reliable |
| RANGE_RATIO | 0.0115 | 0.0111 | Stable | |
| ATR_5 | 0.0081 | 0.0076 | Stable | |
| DIST_HIGH_5 | 0.0070 | 0.0065 | Stable | |
| DELTA_3 | 0.0063 | 0.0064 | Stable | |
| DIST_HIGH_10 | 0.0054 | 0.0060 | Stable | |
| ROC_5 | 0.0053 | 0.0046 | Stable | |
| DELTA_1 | 0.0052 | 0.0057 | Stable | |
| VOL_SPIKE | 0.0052 | 0.0049 | Stable | |
| ROC_10 | 0.0049 | 0.0047 | Stable | |
| RSI_SLOPE | 0.0042 | 0.0052 | Stable | |
| ROC_3 | 0.0046 | 0.0033 | Stable | |
| RETURNS_1 | 0.0042 | 0.0026 | Stable | |
| DIST_LOW_5 | 0.0041 | 0.0037 | Stable | |
| DIST_LOW_10 | 0.0039 | 0.0037 | Stable | |
| RANGE_POS | 0.0036 | 0.0029 | Stable | |
| ATR_14 | 0.0034 | 0.0028 | Stable | |
| ATR_NORM_ROC | 0.0029 | 0.0029 | Stable | |
| STOCH_K | 0.0016 | 0.0003 | Unstable | Weak but kept — VIF and corr acceptable |
| CCI_5 | 0.0009 | 0.0010 | Stable | Weak but kept — theoretical value |
| BUY_RATIO | 0.0000 | 0.0000 | — | **Dropped** |

**ATR_RATIO MI note:** Dropped from 0.0387 (Run1) to 0.0346 (Run2). Expected —
larger dataset with more diverse regimes produces more conservative MI estimates.
Signal is real, ranking position unchanged.

---

### Test 6 — Spearman vs Label

**Large-sample interpretation note:** At 546k rows, p-values are essentially
meaningless — any non-zero correlation achieves p<0.05. Use |r| magnitude as
the operative filter. Features with p>0.05 despite 546k rows are the genuine
non-results.

| Feature | r | p | |r| rank | Note |
|---|---|---|---|---|
| DIST_LOW_5 | -0.0094 | 0.0000 | 1 | |
| STOCH_K | -0.0087 | 0.0000 | 2 | |
| DIST_HIGH_5 | +0.0078 | 0.0000 | 3 | |
| DELTA_1 | -0.0066 | 0.0000 | 4 | |
| RSI_SLOPE | -0.0064 | 0.0000 | 5 | |
| BUY_RATIO | -0.0064 | 0.0000 | — | **Dropped** |
| ATR_RATIO | -0.0060 | 0.0000 | 6 | |
| ATR_5 | -0.0059 | 0.0000 | 7 | |
| ATR_14 | -0.0059 | 0.0000 | 8 | |
| RETURNS_1 | -0.0057 | 0.0000 | 9 | |
| CCI_5 | -0.0052 | 0.0001 | 10 | |
| RSI_14 | +0.0050 | 0.0002 | 11 | |
| MOM_3 | -0.0048 | 0.0004 | 12 | |
| ROC_10 | +0.0047 | 0.0005 | 13 | |
| DELTA_3 | -0.0054 | 0.0001 | 14 | |
| ROC_5 | -0.0039 | 0.0044 | 15 | |
| DIST_LOW_10 | -0.0036 | 0.0078 | 16 | |
| DIST_HIGH_10 | +0.0031 | 0.0204 | 17 | |
| RANGE_RATIO | -0.0025 | 0.0674 | 18 | p>0.05 borderline |
| RANGE_POS | -0.0023 | 0.0962 | 19 | p>0.05 borderline |
| VOL_SPIKE | -0.0020 | 0.1313 | 20 | p>0.05 |
| ATR_NORM_ROC | -0.0019 | 0.1656 | 21 | p>0.05 |
| ROC_3 | -0.0017 | 0.2175 | 22 | p>0.05 |
| DELTA_DIV | +0.0006 | 0.6838 | 23 | p>0.05 — ablation candidate |

**Features non-significant at p>0.05 despite 546k rows:**
ROC_3, ATR_NORM_ROC, VOL_SPIKE, RANGE_POS, RANGE_RATIO, DELTA_DIV.
These six have genuinely negligible linear association with the label.
Kept because LSTM captures non-linear, sequential relationships that Spearman cannot.

**DELTA_DIV** has both the weakest |r| (0.0006) and the highest p (0.6838).
Combined with MI instability, it is the primary ablation candidate.

---

## Drop Summary

| Feature | Group | Reason | Evidence |
|---|---|---|---|
| BUY_RATIO | L | No signal on 546k dataset | MI=0.000 both k; r=-0.006, same as other drops but all other tests also weak |
| RSI_5 | K | High multicollinearity (Run1) | VIF=32.7 — not retested, confirmed drop |

---

## Ablation Plan

| Run | Features | Count | DELTA_DIV |
|---|---|---|---|
| Run10 | I+J+K+L+M without DELTA_DIV | 21 | No |
| Run11 | I+J+K+L+M with DELTA_DIV | 22 | Yes |

**Decision rule after runs:**

| Condition | Decision |
|---|---|
| Run11 test F1 > Run10 by > 0.002 AND gap does not widen | Keep DELTA_DIV (22 features) |
| Run11 ≤ Run10 OR gap widens | Drop DELTA_DIV (21 features) |

---

## Comparison with Run1

| Property | Run1 | Run2 | Change |
|---|---|---|---|
| Dataset size | 229,730 | 546,413 | +2.4× |
| Date range | ~800 days | 2021-11-01 to 2026-03-11 | Full regime coverage |
| Split ratio | 70/15/15 | 80/10/10 | More training data |
| Train rows | 160,839 | 437,104 | +2.7× |
| Features | 24 | 21/22 | BUY_RATIO dropped; DELTA_DIV ablation |
| DIST_LOW_5/10 normalisation | Global Z | Rolling Z | Corrected — KPSS failure on full dataset |
| Val set bias | Peak-bull (structural) | Mixed regimes | More reliable early stopping |
| Best Run1 result | Test F1 = 0.3987 | TBD | Baseline to beat |

---

## Pipeline Order (Final)

```
df_raw
  │
  ├─ Compute all features (Groups I–J–K–L–M)
  ├─ Apply rolling Z-score (window=500) to 10 features
  ├─ Drop first 500 warmup rows
  ├─ Temporal split 80/10/10
  │     Train : index 0       – 437,103
  │     Val   : index 437,104 – 491,741
  │     Test  : index 491,742 – 546,381
  ├─ Fit global Z-score on train → apply to val and test
  ├─ Winsorise ±3σ (clamp all features, do not drop rows)
  └─ Build LSTM sequences (seq=20, stride=1)
```