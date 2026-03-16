# Regime Classifier — Design Spec
**Session Summary · 7-Regime Rule-Based Classifier · Multi-Timeframe · Hierarchical 3+3 + UNKNOWN**

> Decisions made in this session. Reference this when returning with the
> extended dataset to begin parameter validation and implementation.
> Gate action for UNKNOWN: hold previous regime and suppress new position opens. Do not close existing positions — UNKNOWN means classifier uncertainty, not a bear signal.

---

## Context

The LSTM model (lstm2.py) produces signals that are not equally reliable across
all market conditions. A regime classifier sits upstream of the LSTM as a
**gating layer** — it reads raw market structure independently and decides
whether LSTM signals should be allowed, suppressed, or scaled for the current
environment.

```
Market Data (bar-by-bar)
        │
        ▼
┌─────────────────────┐
│  Regime Classifier  │  ← this document
└─────────────────────┘
        │
        ▼
  ┌─ BULLISH group ──────────────────────────────┐
  │  ACCUMULATION  │  BULLISH  │  RECOVERY        │
  └──────────────────────────────────────────────┘
  ┌─ BEARISH group ──────────────────────────────┐
  │  DISTRIBUTION  │  BEARISH  │  CORRECTION      │
  └──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│   LSTM Signal Gate  │
└─────────────────────┘
        │
        ▼
   Trade / No Trade / Position Size
```

---

## Decisions Made

### 1. Number of Regimes — 7 (Hierarchical 3+3 + UNKNOWN)

The 7 regimes are organised into two symmetric groups plus a special UNKNOWN
state. The 1H EMA crossover determines the group; the 15-min ATR ratio + RSI
determine the sub-state within that group. UNKNOWN is emitted when no rule fires
or during classifier warm-up.

```
BULLISH group               BEARISH group           Special
─────────────────────       ─────────────────────   ──────────────
ACCUMULATION                DISTRIBUTION          ← symmetric pair (vol forming)
BULLISH                     BEARISH               ← symmetric pair (trend running)
RECOVERY                    CORRECTION            ← symmetric pair (vol spike)
                                                    UNKNOWN       ← no rule fires
```

| Group | Regime | Price Behaviour | Volatility | Character |
|---|---|---|---|---|
| **BULLISH** | **ACCUMULATION** | Slow grind up, pullbacks absorbed | Low–moderate | Buyers gradually winning, trend forming |
| **BULLISH** | **BULLISH** | Sustained uptrend, higher highs | Moderate | Consistent buy pressure, trending |
| **BULLISH** | **RECOVERY** | Bounce from low, trend resuming | Moderate–high, declining | Buy pressure returning, vol compressing |
| **BEARISH** | **DISTRIBUTION** | Price topping, selling into strength | Expanding, RSI diverging | Smart money exiting, trend about to break |
| **BEARISH** | **BEARISH** | Sustained downtrend, lower highs | Moderate, persistent | Grinding lower, no recovery in sight |
| **BEARISH** | **CORRECTION** | Sharp drawdown, panic selling | Very high (spike) | Aggressive selling, brief, violent |
| **—** | **UNKNOWN** | No rule matches or warm-up not complete | — | Classifier uncertainty; hold previous, suppress new opens |

**Why 7 and not 6:** The 6-regime decision matrix has real coverage gaps where
no rule fires. These are not edge cases — a standard bull-market pullback (1H
still BULLISH, RSI drops to 40, ATR not declining) has no matching row.
UNKNOWN is the correct response: hold the previous regime label, suppress new
position opens, do not close existing positions.

**Known gaps that produce UNKNOWN:**

| Group | ATR (15m) | RSI (15m) | Gap description |
|---|---|---|---|
| BULLISH | < 1.3 | < 48 | Pullback within uptrend — not RECOVERY (no 48↑ cross yet) |
| BULLISH | > 1.5 spike | > 55 | Blow-off / breakout spike — not any defined sub-state |
| BEARISH | Any | 48–55 | Neutral momentum in bear market |
| BEARISH | < 1.0 | 40–48 | Quiet oversold — not CORRECTION (no spike) |

**Why 6 and not 5:** The original 5-regime flat design was asymmetric — 3 bullish
sub-states and only 2 bearish. DISTRIBUTION is the structural mirror of
ACCUMULATION: price is still near highs, 1H EMA still bullish, but RSI is
diverging downward and taker buy volume is declining. Without DISTRIBUTION the
classifier has no signal for the topping phase, causing the gate to trade
DISTRIBUTION at full BULLISH size — incorrect behaviour.

**Why DISTRIBUTION belongs to BEARISH group:** By the time DISTRIBUTION is
confirmed (RSI diverging, vol expanding into selling, taker buy ratio declining),
the 1H structure is already deteriorating. The gate action for DISTRIBUTION
(reduce/close longs) is aligned with BEARISH group suppression, not BULLISH
group allowance.

**RECOVERY placement — BULLISH group only:** RECOVERY is placed under BULLISH
because its confirmation signal (RSI crossing 50 upward, EMA fast reversing)
coincides with the 1H EMA crossing bullish. Dead-cat bounces within a bear
market — where 1H EMA is still bearish — are classified as CORRECTION (15m vol
spike subsiding) not RECOVERY, preventing misclassification.

---

### 2. Classifier Type — Rule-Based

Rule-based using EMA crossover + ATR ratio + RSI. A learned classifier
(HMM, clustering) was rejected because:

- Requires a second model to train, validate, and maintain
- Learned states do not self-label — manual regime assignment still required
- No reduction in human judgement, only added complexity
- Rule-based is transparent, fast to implement, directly testable

**Upgrade path:** If rule-based produces consistent misclassification at regime
boundaries on out-of-sample data, revisit a learned classifier at that point.

---

### 3. Timeframe — Multi-Timeframe (15-min + 1H)

| Timeframe | Role | Determines |
|---|---|---|
| **1H** | Macro: group assignment | BULLISH group or BEARISH group |
| **15-min** | Micro: sub-state within group | Which of the 3 sub-states |

Both timeframes are read directly from the pre-built JSONL feeds — no
resampling required.

**Two-layer classification logic:**

```
Step 1 — Group (1H EMA only):
  EMA_fast_1h > EMA_slow_1h  →  BULLISH group
  EMA_fast_1h < EMA_slow_1h  →  BEARISH group

Step 2 — Sub-state (15m ATR ratio + RSI):
  Within BULLISH group:
    RSI 48–55, ATR compressed          →  ACCUMULATION
    RSI > 55, ATR compressed–moderate  →  BULLISH
    RSI crossing 48↑, ATR declining    →  RECOVERY

  Within BEARISH group:
    RSI diverging ↓ from > 60, ATR expanding, taker_buy_ratio < 0.45  →  DISTRIBUTION
    RSI 40–48 persistent, ATR mild (1.0–1.3)                          →  BEARISH
    RSI < 40, ATR spike > 1.5                                         →  CORRECTION

Step 3 — Hysteresis:
  If 15m sub-state ambiguous → hold previous regime
  (natural hysteresis; no separate min hold period parameter needed)
```

**DISTRIBUTION-specific indicator — taker_buy_ratio:**

DISTRIBUTION requires one additional signal not needed by other regimes:

```python
taker_buy_ratio = taker_buy_basevol / vol  # already in raw JSONL
# < 0.45 means sellers are dominating despite price still near highs
```

Computed exclusively on the **15m feed** — consistent with all other sub-state indicators.
This field (`taker-buy-basevol` / `vol`) is already present in the 15m JSONL feed. No new data required.

---

### 4. Indicators — EMA Crossover + ATR Ratio + RSI + Taker Buy Ratio

| Indicator | Parameters | What it contributes | Used for |
|---|---|---|---|
| EMA crossover (fast/slow) | 2 | Trend direction | Group assignment (1H) |
| ATR ratio (short / long) | 2 (fixed) | Volatility regime | Sub-state (all) |
| RSI level | 1 (period 30) | Momentum confirmation | Sub-state (all) |
| Taker buy ratio | 0 (derived) | Order flow imbalance | DISTRIBUTION only |

**Total free parameters: 3** (EMA fast period, EMA slow period, ATR thresholds).
RSI period (30) is concluded — independent of lstm2.py's RSI 14. ATR periods (10, 50) are independent of lstm2.py's ATR 14/42 feature — the regime classifier computes its own ATR ratio solely to determine sub-state. Taker buy ratio is derived (`taker-buy-basevol / vol`), no new parameter.

**Parameter values:**

| Parameter | 15-min value | 1H value | Status | Notes |
|---|---|---|---|---|
| EMA fast | 20 bars | 50 bars | **Concluded** | ~5H on 15-min, ~50H on 1H (~2 days) |
| EMA slow | 60 bars | 200 bars | **Concluded** | ~15H on 15-min, ~200H on 1H (~8.3 days) |
| EMA confirmation buffer | — | 3 bars | Draft | Hold N bars before accepting group flip; see pre-computation.md |
| ATR periods (short/long) | 10/50 | 10/50 | **Concluded** | 5× spread; wider dynamic range than 14/42 — compresses sharper in quiet, spikes higher in panic |
| ATR expand threshold | 1.3 | 1.3 | **Concluded** | p89 of all bars — top 11%; meaningful vol expansion signal |
| ATR spike threshold | 1.5 | 1.5 | **Concluded** | p97 of all bars — genuine panic/spike events only |
| RSI period | 30 | 30 | **Concluded** | 7.5H smoothing on 15m — momentum indicator, not trend filter; independent of lstm2.py RSI 14 |
| RSI oversold (CORRECTION) | < 40 | < 40 | **Concluded** | ~p10 on RSI30 — genuine panic only |
| RSI bear band (BEARISH) | 40–48 | 40–48 | **Concluded** | Empirically confirmed 2022 bear market floor |
| RSI neutral (ACCUMULATION) | 48–55 | 48–55 | **Concluded** | Bull/bear separator; 48 = bull market floor, 55 = bear market ceiling |
| RSI bull momentum (BULLISH) | > 55 | > 55 | **Concluded** | Strong directional momentum |
| RSI DISTRIBUTION entry | > 60 | > 60 | **Concluded** | ~p88 on RSI30 — overbought, divergence detectable |
| Taker buy ratio threshold | < 0.45 (15m) | — | **Concluded** | 15m feed only; sellers dominating order flow into DISTRIBUTION |

**EMA rationale (1H 50/200):** Classic golden/death cross pair. EMA 200 on 1H = ~8.3 days
of price memory — stable across multi-week regime periods. Confirmed via plot: crossovers
align with known structural transitions (2022 bear entry, 2023 recovery). Extra flips
during mid-regime corrections are handled by the 3-bar confirmation buffer (live only;
tolerable as label noise in training). 15m EMA 20/60 confirmed as sub-state reference only.

**ATR rationale (10/50, thresholds 1.0/1.3/1.5):** Percentile analysis on 45k 1H bars and 182k 15m bars showed all regime periods have nearly identical ATR ratio means (~0.98) — the ratio is stationary regardless of regime. Discrimination comes only at the tails. The 5× period spread (10 vs 50) maximises dynamic range: quiet bars compress more clearly below 1.0, spike bars peak higher. Threshold mapping from data: compressed < 1.0 (below p52), expand > 1.3 (p89), spike > 1.5 (p97). The former 1.1 threshold was p74 — top quartile of all bars, no discriminative power.

**Taker buy ratio rationale (< 0.45, 15m only):** DISTRIBUTION is a macro topping process where smart money sells into bid-side liquidity. Taker buy ratio < 0.45 means sellers account for > 55% of aggressive order flow — a meaningful imbalance signal. Threshold 0.45 is a round number near the neutral 0.5 midpoint, chosen as the minimum detectable seller dominance. Computed on 15m feed only — consistent with all other sub-state indicators; 1H smoothing would lag the early topping signal.

**RSI rationale (period 30, thresholds 40/48/55/60):** Period exploration on 1H data (tested 20, 30, 50) showed RSI30 as the right balance — RSI20 whipsawed across thresholds; RSI50 became a slow trend filter redundant with the EMA, making thresholds like < 30 structurally unreachable. RSI30 = 7.5H smoothing on 15m, acting as a momentum indicator. Threshold calibration from visual plot validation: in the 2024 BULLISH period, RSI30 holds above 48 with occasional excursions to 60+; in the 2022 BEARISH period, RSI30 stays under 55 and mostly in the 40–48 band; CORRECTION events (Mar 2020, Nov 2022 FTX) drop through 40. The 48–55 zone is empirically confirmed as the neutral band where neither bulls nor bears have conviction (ACCUMULATION). All thresholds are independent of lstm2.py's RSI 14 feature.

---

### 5. Regime Decision Matrix (Draft)

**Signal thresholds:**
```
EMA fast > EMA slow (1H)  →  BULLISH group
EMA fast < EMA slow (1H)  →  BEARISH group

ATR_short / ATR_long > 1.5   →  high vol (spike)
ATR_short / ATR_long 1.0–1.3 →  mild vol expansion
ATR_short / ATR_long < 1.0   →  compressed vol

RSI > 55              →  momentum bullish (BULLISH sub-state)
RSI 48–55             →  momentum neutral (ACCUMULATION zone)
RSI 40–48             →  momentum bearish (BEARISH sub-state)
RSI < 40              →  extreme oversold (CORRECTION)
RSI > 60              →  overbought — divergence detectable (DISTRIBUTION entry)
RSI diverging down    →  RSI falling while price flat/up (DISTRIBUTION signal)
RSI crossing 48↑      →  momentum turning bullish (RECOVERY signal)

taker_buy_ratio < 0.45  →  sellers dominating order flow
```

**Full decision matrix:**

| Group (1H) | ATR ratio (15m) | RSI (15m) | Taker Buy | Sub-state |
|---|---|---|---|---|
| BULLISH | < 1.0 | 48–55 | — | ACCUMULATION |
| BULLISH | < 1.3 | > 55 | — | BULLISH |
| BULLISH | 1.0–1.5, declining | Crossing 48↑ | — | RECOVERY |
| BEARISH | Expanding (1.0–1.3) | Diverging ↓ from > 60 | < 0.45 | DISTRIBUTION |
| BEARISH | 1.0–1.3, persistent | 40–48 | — | BEARISH |
| BEARISH | > 1.5 (spike) | < 40 | — | CORRECTION |

**Symmetric pair distinctions:**

| | ACCUMULATION | DISTRIBUTION |
|---|---|---|
| 1H EMA | Bullish, widening | Bullish but flattening |
| RSI | Rising toward 55 | Falling from > 60 (divergence) |
| ATR | Compressed | Expanding (selling into bids) |
| Taker buy | Neutral–rising | < 0.45 (sellers absorbing) |
| Gate action | Allow, moderate size | Reduce / close longs |

| | BULLISH | BEARISH |
|---|---|---|
| 1H EMA | Fast clearly above slow | Fast clearly below slow |
| RSI | Persistently > 55 | Persistently 40–48 |
| ATR | Moderate, stable | Mild, persistent expansion |
| Gate action | Full size | Suppress longs |

| | RECOVERY | CORRECTION |
|---|---|---|
| 1H EMA | Crossing bullish | Still bearish (or sharply crossed) |
| RSI | Crossing 48↑ | Drops below 40, may bounce |
| ATR | 1.0–1.5, declining | > 1.5 spike |
| Duration | Days–weeks | Days–weeks |
| Gate action | Allow, reduced size | Suppress all |

---

### 6. Dataset — Extended to 2021-11-01

**Decision:** Refetch full dataset from 2021-11-01 to cover the 2021–2022
bear market. No gap handling required — the dataset is continuous.

| Period | Regime Coverage | Approx Bars |
|---|---|---|
| 2021-08 → 2021-11 | DISTRIBUTION (pre-crash topping) | ~26,000 |
| 2021-11 → 2022-11 | BEARISH | ~158,000 |
| 2022-11 → 2022-12 | CORRECTION (FTX collapse) | ~6,000 |
| 2022-12 → 2023-06 | RECOVERY (bear exit) | ~79,000 |
| 2023-06 → 2024-01 | ACCUMULATION | ~78,000 |
| 2024-01 → 2025-11 | BULLISH | ~229,730 (current) |
| 2025-08 → 2025-11 | DISTRIBUTION (pre-correction topping) | ~within above |
| 2025-11 → 2026-02 | CORRECTION + RECOVERY | ~within current |
| **Total** | **All 6 regimes** | **~545,000+ bars** |

**Note on DISTRIBUTION coverage:** Two clear DISTRIBUTION instances exist in the
dataset — the 2021-08 → 2021-11 topping phase before the bear market, and the
2025-08 → 2025-11 topping phase before the current correction. This gives at
least two historical validation points for the DISTRIBUTION rules.

**Why 2021-11-01 specifically:** BTC peaked at ~69k in November 2021, marking
the start of the sustained bear market. The 2018 bear market was not included —
different market structure, thinner liquidity, less representative of current
conditions.

**Fetch method:** Existing Binance 5-min fetch script, start date changed to
2021-11-01. No other script changes required.

---

### 7. Pipeline Re-run Required

Combining the extended dataset requires re-running the full pipeline:

```
1.  Fetch 2021-11-01 → current (~545k bars)
2.  Verify continuity — no gaps, clean bar count
3.  ATR_42 warmup drop (41 rows)
4.  Re-run all feature engineering (Groups I–M)
5.  Rolling Z-score warmup drop (500 rows)
6.  Re-run triple-barrier labelling
7.  Re-decide temporal split boundaries (70/15/15 on ~545k rows)
8.  Fit global Z-score on new train set only
9.  Re-run all ablation experiments
```

**Revised split estimate on ~545k rows (70/15/15):**

| Split | Rows | Approx period | Regimes |
|---|---|---|---|
| Train | ~381k | 2021-11 → ~2025-02 | Bear, Recovery, Accumulation, early Bull |
| Val | ~82k | ~2025-02 → ~2025-09 | Bull peak |
| Test | ~82k | ~2025-09 → 2026-02 | Correction, Recovery |

Val still lands in bull peak — same structural bias as documented in
dataset_characteristics.md. This is expected and unchanged.

---

### 8. Cold Start

Initial regime = **UNKNOWN**.

UNKNOWN's gate action (hold previous, suppress new opens) already handles this
correctly — with no previous regime, "hold previous" is a no-op and the gate
suppresses all new opens until the first clean classification is emitted.
No separate cold start logic is required.

---

## Open Items — To Resolve With Full Dataset

| Item | Status | When |
|---|---|---|
| EMA parameter validation (fast/slow values) | **Concluded** — 1H 50/200, 15m 20/60, 3-bar confirmation buffer (draft) | — |
| ATR threshold validation | **Concluded** — 10/50 periods, expand 1.3 (p89), spike 1.5 (p97) | — |
| RSI parameter validation | **Concluded** — period 30; thresholds 40 (CORRECTION) / 48 (bear/bull separator) / 55 (bull momentum) / 60 (DISTRIBUTION entry) | — |
| Taker buy ratio threshold for DISTRIBUTION | **Concluded** — < 0.45 on 15m feed; sellers dominating order flow | — |
| Hysteresis rule (min hold period, two-signal both directions) | Not yet defined | After fetch |
| Gate actions per regime (allow / suppress / scale factor) | Not yet defined | After regime validation |
| DISTRIBUTION validation on 2021-08 → 2021-11 topping | Not yet run | After fetch |
| DISTRIBUTION validation on 2025-08 → 2025-11 topping | Not yet run | After fetch |
| Regime classifier validation on 2022-08 → 2022-11 | Not yet run | After fetch |
| Visual regime annotation plot (6 regimes) | Not yet done | First step after fetch |
| Sampling ratio — DISTRIBUTION instances vs other regimes | Deferred | After seeing results |
| UNKNOWN frequency audit — what % of bars fall in coverage gaps | Not yet measured | After classifier first run |
| Gap-fill rules for persistent UNKNOWN runs (>N bars) | Not yet defined | After frequency audit |

---

## Recommended First Step on Return

**Before any code — plot close price across the full combined dataset and
manually annotate the 6 regime periods.** This visual confirmation serves two
purposes:

1. Confirms the fetch is clean and continuous
2. Gives you ground truth labels to validate the classifier rules against

Only after the plot confirms the regimes are visually distinct should parameter
selection begin.

---

## Known Limitations

- Val set structural bias toward bull peak is unchanged — documented in
  dataset_characteristics.md, expected and acceptable
- BEARISH regime validation is limited to 2021–2022 bear market only —
  one historical instance is not a guarantee of generalisation
- DISTRIBUTION validation is limited to two instances (2021-08 and 2025-08) —
  the taker_buy_ratio threshold (0.45) is a draft value, not empirically validated
- Gate actions per regime (allow / suppress / scale) are hypotheses only
  until validated against per-regime precision/recall breakdown from ablation runs
- 2018 bear market deliberately excluded — add only if 2021–2022 validation
  reveals the classifier needs more bear market diversity
- Dead-cat bounces within BEARISH are classified as CORRECTION (not RECOVERY) —
  this is intentional but creates a sharp boundary that may produce false
  RECOVERY signals near bear market lows if the 1H EMA crosses briefly
- UNKNOWN frequency is unknown until the classifier runs on real data — if gap
  coverage is high (>15% of bars), the decision matrix needs additional rules
  before the gate layer is meaningful
- Cold start initial state is UNKNOWN — the gate suppresses all new opens until
  the first clean classification, which is the correct and safe default
