# Label Reframing Notes

## Problem with "Next Bar Up/Down"
- Next bar direction is ~50/50 noise with near-zero autocorrelation
- No learnable structure — model ceiling is random chance
- Need labels that reflect directional follow-through, not tick-level noise

---

## Strategy 1: Fixed-Horizon with Threshold

Label = 1 if `close[t+N] > close[t] + threshold`, else 0

- N = 8–20 bars (2–5 hours on 15m)
- threshold = 0.3–0.5% (filters flat chop)
- Optionally use 3-class (up / flat / down), then drop flat rows

**Limitation:** still uses a fixed future point, just further out.

---

## Strategy 2: Triple-Barrier Method (Lopez de Prado) — Recommended

For each bar, set three barriers:
- **Upper barrier** (take profit): +X%
- **Lower barrier** (stop loss): -X%
- **Time barrier**: N bars max

Label = whichever barrier is hit first (1=upper, 0=lower, discard 0.5 if time-out).

```python
def triple_barrier_label(close, tp=0.005, sl=0.005, max_bars=20):
    labels = []
    for i in range(len(close) - max_bars):
        entry = close[i]
        label = 0.5  # default: time out
        for j in range(1, max_bars + 1):
            ret = (close[i+j] - entry) / entry
            if ret >= tp:
                label = 1
                break
            elif ret <= -sl:
                label = 0
                break
        labels.append(label)
    return labels
```

Discard timeout rows (label == 0.5) — they're ambiguous. Most principled approach.

**Recommended params for 15m BTCUSD:**
- `tp = sl = 0.4–0.6%` (~1–1.5x ATR on 15m BTC)
- `max_bars = 16` (4 hours)
- Expected non-timeout rate: ~40–50%

---

## Strategy 3: Zigzag / Swing Label

Label each bar by which swing extreme comes next.

```
if next_swing == HIGH → 1
if next_swing == LOW  → 0
```

Swing detection: `high[i] > high[i-k]` and `high[i] > high[i+k]` for k = 3–5 bars.

**Note:** Labels are look-ahead by construction — valid for supervised learning only, not live inference (requires confirmed swing after the fact).

---

## Strategy 4: Trend Regime Label

Use a slow indicator to define regime rather than predicting next bar:

```python
label = 1 if ema_fast[t] > ema_slow[t] else 0
# or
label = 1 if price > hma_200[t] else 0
```

Model learns trend continuation — more stable and persistent than point predictions.

---

## Recommendation

Start with **triple-barrier**. It answers a tradeable question: "did price move X% up before moving X% down?" which has real structure and is learnable.

---

## Why Triple-Barrier Stands Out for This Setup (15m BTCUSD, LSTM binary classifier)

**1. Matches how trading actually works**
The model output will eventually drive a trade with a target and a stop. Triple-barrier labels the data using the same logic — the model learns to predict outcomes that are directly tradeable, not abstract price direction.

**2. Filters out noise that killed the previous label**
"Next bar up/down" labeled every bar including sideways drift. Triple-barrier only produces a clean label when price *committed* to a direction. The timeout rows you discard are exactly the ambiguous, unlearnable cases.

**3. Asymmetry-aware**
`tp` and `sl` can differ. If BTC tends to spike up fast but bleed down slowly, barriers can reflect that. Fixed-horizon labels ignore this completely.

**4. Aligns with LSTM's temporal structure**
The LSTM sees a 200-bar sequence and outputs one label. Triple-barrier lookahead (16 bars) is well within the sequence length, so what the LSTM learns aligns with what the label actually measures.

**5. Most studied approach for this exact problem**
Lopez de Prado developed it specifically for financial time-series ML. Alternatives either require look-ahead at inference time (zigzag) or ignore the stop dimension entirely (fixed-horizon). Trend regime labels are valid but make the model a regime classifier, not a trade signal generator.

**Core point:** Turns "which direction did price go" into "did price move enough in one direction to matter" — a much cleaner and learnable signal.

---

## Variant Idea: Dual-Timeframe Triple-Barrier for Next-Bar Prediction

**Concept:** Keep the next-bar prediction target but improve *how* the label is measured.

### Setup
- **Features:** 15m + 5m OHLCV (dual timeframe)
- **Label:** triple-barrier evaluated on 5m bars
- **tp = sl:** 0.8–1.5x ATR (computed on 15m, applied as threshold)
- **max_bars = 3 on 5m** = exactly 1 15m bar of lookahead
- **Timeout rows:** drop them (equivalent to doji/indecision — unlearnable)
- **Label:** binary [1, -1]

### Why this works
- `max_bars=3 on 5m` is still next-bar prediction, but measured by path (via 5m) rather than endpoint
- ATR-scaling means thresholds adapt to volatility regime — avoids mislabeling a 0.1% move as "up" during a 1% ATR environment
- Dropping timeouts removes the indecision bars that pollute the signal
- 5m features directly support the label since the label is evaluated on 5m bars (intra-bar momentum visibility)

### ATR threshold guidance
- `0.8x ATR` — tighter, more labels retained, but noisier
- `1.0–1.2x ATR` — recommended sweet spot
- `1.5x ATR` — cleaner labels but higher timeout rate (risk: drop too many rows)

### Label balance
- With symmetric tp=sl, should land near 50/50 after dropping timeouts
- Drift only from trending periods in the dataset (BTC bull run → more 1s)
- If skewed >55/45, apply class weighting

### Risks
1. **Timeout rate** — at 1.5x ATR with only 3 bars, may drop 40–60% of rows. Run histogram before committing.
2. **ATR look-ahead** — compute ATR at bar `t` using only data up to `t`. No forward bleed.
3. **5m/15m sync** — each 15m bar `t` maps to 5m bars `[3t, 3t+1, 3t+2]`. Be explicit about this indexing.
