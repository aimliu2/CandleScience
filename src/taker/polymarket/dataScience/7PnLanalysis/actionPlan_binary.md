# Q3 — Exit Mechanics: Binary P&L Analysis (Revised)
**Fixed Exit N=3 bars · Threshold Search · One Position at a Time · Redefined P&L as 1R x Size**

## Objective
- Goal: with exit fixed at bar 3 (aligned with label horizon), find which
- conviction threshold maximises edge vs threshold.
- Threshold and direction usability are decided together from the same experiment.
- Enter at the open of the next actual bar after the signal (index i+1 in the sorted test sequence).
- Exit at the close of the 3rd actual bar after the signal (index i+3). No stop-loss. No take-profit.
- Entry and exit use index-based lookup on the sorted test sequence, so calendar gaps (exchange downtime, weekends) are handled correctly — the next real trading bar is always used regardless of timestamp distance.
- P&L will be count as a unit as following;
    - If The model's prediction was correct. the winning amount will be = `Conviction x 1 Unit`.
    - If The model's prediction was wrong. the losing amount will be = `Conviction x 1 Unit`.
    - If no signal were fired. `PnL = 0`
- Only signals whose entry bar opens on a 15-minute boundary (HH:00, HH:15, HH:30, HH:45) are tradeable. Signals with non-boundary entry times (HH:05, HH:10, etc.) are discarded before the one-position filter.
---



## Experiment Design

**What is fixed:**
- Exit bar: N=3 (close of bar 3 after signal)
- Entry: entry at the next bar open after the signal
- Position rule: one position at a time, skip signal if already in trade

**What varies:**
- Threshold: sweep 0.01 → 0.20 in steps of 0.01
- Groups: LONG / SHORT × correction / recovery (per Q1 findings)

**Primary outputs per threshold:**
- Edge    = 2 × win_rate − 1
- Max DD  = largest peak-to-trough drop in cumulative units across the trade sequence

The threshold where edge is maximised, with sufficient signal count, is the operating threshold in this case.

---

## Step 1 — Build the Signal Table

For each threshold in the sweep, construct a signal DataFrame. Because the
one-position filter depends on threshold (fewer signals at higher thresholds
means fewer skips), run the filter separately per threshold.

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path

HORIZONS    = 3          # fixed
THRESHOLDS  = np.arange(0.01, 0.21, 0.01)
BOUNDARY    = pd.Timestamp('2026-02-07', tz='UTC')   # tz-aware to match test_timestamps
MIN_SIGNALS = 30         # minimum to report a row; below this, suppress

IDX_DOWN, IDX_NONE, IDX_UP = 0, 1, 2

# ── load matched OHLC file ────────────────────────────────────────────────
# Source: data/mlData/processed/202603-new-v10-test-ohlc.jsonl
# Produced by match-feature-ohlcv.py.

MATCHED_PATH = Path("data/mlData/processed/202603-new-v10-test-ohlc.jsonl")

rows = [json.loads(l) for l in MATCHED_PATH.read_text().splitlines()]
rows = rows[SEQ_LEN:]   # drop first SEQ_LEN rows — sliding window has no prediction for them
                        # probs_arr[k] aligns with rows[k] after this trim

test_timestamps = pd.to_datetime([r["timestamp"] for r in rows], unit="ms", utc=True)
entry_open      = np.array([r["entry_open"]  for r in rows])
exit_close      = np.array([r["exit_close"]  for r in rows])

# test_probs = probs_arr  (alias assigned in notebook; shape (N, 3))
assert len(test_probs) == len(rows), (
    f"test_probs length {len(test_probs)} != matched file length {len(rows)}"
)
```

---

## Step 2 — One-Position Filter

Apply before computing any metrics. Signals skipped by this filter are
discarded — they were never tradeable.

```python
def apply_one_position_filter(signal_indices, timestamps, exit_bars=3):
    """
    Return only tradeable indices under the one-position rule, with an
    additional 15-minute boundary gate on the entry bar.

    Rules applied in order:
      1. Entry bar (signal_idx + 1) must open on HH:00, HH:15, HH:30, or HH:45.
         Signals with non-boundary entry times are discarded.
      2. One-position rule: next signal must start after current trade exits
         (exit occurs at signal_idx + exit_bars).

    Parameters
    ----------
    signal_indices : sorted int array of candidate signal positions
    timestamps     : DatetimeIndex aligned to the same index space
    exit_bars      : hold period in bars (default 3)
    """
    taken     = []
    last_exit = -1
    n         = len(timestamps)

    for idx in signal_indices:
        entry_idx = idx + 1
        if entry_idx >= n:
            continue                         # no entry bar available

        if timestamps[entry_idx].minute % 15 != 0:
            continue                         # not a 15-min boundary — skip

        if idx > last_exit:
            taken.append(idx)
            last_exit = idx + exit_bars

    return np.array(taken, dtype=np.int64)


def max_drawdown(outcomes):
    """
    Peak-to-trough maximum drawdown of cumulative binary P&L.
    outcomes : array of +1 (win) / -1 (loss) in trade time order.
    Returns the largest drop from any running peak, in units.
    """
    cum  = np.cumsum(outcomes)
    peak = np.maximum.accumulate(cum)
    return int((peak - cum).max())
```

---

## Step 3 — Compute Edge at Each Threshold

```python
results = []

for thr in THRESHOLDS:
    # Find all signals at this threshold
    long_mask  = test_probs[:, IDX_UP]   > thr
    short_mask = test_probs[:, IDX_DOWN] > thr

    long_idx  = np.where(long_mask)[0]
    short_idx = np.where(short_mask)[0]

    # Merge and sort by time, track direction
    all_idx  = np.concatenate([long_idx, short_idx])
    all_dir  = np.concatenate([np.ones(len(long_idx)), -np.ones(len(short_idx))])
    order    = np.argsort(all_idx)
    all_idx, all_dir = all_idx[order], all_dir[order]

    # Apply one-position filter + 15-min boundary gate
    taken_mask = apply_one_position_filter(all_idx, test_timestamps, exit_bars=HORIZONS)
    taken_set  = set(taken_mask)
    keep       = np.array([i for i, idx in enumerate(all_idx) if idx in taken_set])

    if len(keep) == 0:
        continue

    idx_taken = all_idx[keep]
    dir_taken = all_dir[keep]
    ts_taken  = test_timestamps[idx_taken]
    regime    = np.where(ts_taken < BOUNDARY, 'correction', 'recovery')

    # Drop signals too close to end of test set (no room for HORIZONS bars ahead)
    valid = (idx_taken + HORIZONS) < len(test_probs)
    idx_taken, dir_taken, regime = (
        idx_taken[valid], dir_taken[valid], regime[valid]
    )

    # Binary outcome: win if actual price moved in the predicted direction
    # LONG wins when exit_close > entry_open; SHORT wins when exit_close < entry_open
    price_up   = exit_close[idx_taken] > entry_open[idx_taken]
    price_down = exit_close[idx_taken] < entry_open[idx_taken]
    correct = np.where(dir_taken == 1, price_up, price_down)

    # Report per direction × regime
    for direction, dir_label in [(1, 'LONG'), (-1, 'SHORT')]:
        for rgm in ['correction', 'recovery']:
            mask = (dir_taken == direction) & (regime == rgm)
            if mask.sum() < MIN_SIGNALS:
                continue

            c    = correct[mask]
            n    = mask.sum()
            wr   = c.mean()
            edge = 2 * wr - 1                        # +1 per win, −1 per loss
            mdd  = max_drawdown(np.where(c, 1, -1))  # units

            results.append({
                'threshold': round(thr, 2),
                'direction': dir_label,
                'regime':    rgm,
                'n':         n,
                'win_rate':  round(wr,   4),
                'edge':      round(edge, 4),
                'max_dd':    mdd,
            })

results_df = pd.DataFrame(results)
```

---

## Step 4 — Report

Print four sub-tables, one per direction × regime combination.
Within each, rows are sorted by threshold ascending so the edge curve is readable.

```python
for direction in ['LONG', 'SHORT']:
    for regime in ['correction', 'recovery']:
        sub = results_df[
            (results_df.direction == direction) &
            (results_df.regime    == regime)
        ].sort_values('threshold')

        if sub.empty:
            continue

        best_row = sub.loc[sub['edge'].idxmax()]
        print(f"\n{'='*72}")
        print(f"  {direction} | {regime.upper()}")
        print(f"{'='*72}")
        print(sub.to_string(index=False))
        print(f"\n  ► Best threshold: {best_row.threshold}  "
              f"edge={best_row.edge:.4f}  n={int(best_row.n)}  "
              f"win_rate={best_row.win_rate:.3f}  max_dd={int(best_row.max_dd)}")
```

---

## Step 5 — Index Alignment Check (run before full sweep)

Alignment is verified by `validate-ohlc-match.py`, which confirms:
- `entry_open` and `exit_close` are matched at t+1 and t+3 respectively
- 3 rows at the tail of the test set are dropped (no t+3 bar available)
- 13.5% of directional signals have label ≠ actual return direction —
  this is why `correct` uses price sign, not label match

Run `validate-ohlc-match.py` again any time the matched file is regenerated.


---

## What to Look For in the Results

| Pattern | Interpretation |
|---|---|
| Edge rises then plateaus as threshold increases | Pick threshold at plateau onset — higher thresholds just reduce count without adding edge |
| Edge rises monotonically but n drops below 30 | Threshold is too aggressive; back off to last reliable row |
| Edge is flat across thresholds | Conviction score does not discriminate quality — threshold choice is arbitrary |
| LONG edge positive, SHORT edge near zero in recovery | Confirms Q1 finding; suppress shorts in recovery regardless of threshold |
| Max DD rises as threshold increases | Fewer trades means longer losing streaks before recovery — higher threshold is not always safer |
| Max DD large relative to n | Sequence has a deep hole; even a positive edge may be psychologically untradeable |

---

## Deliverables

1. Four edge tables (LONG/SHORT × correction/recovery) across threshold sweep
2. Best threshold per group with supporting n, win_rate, max_dd
3. Win_rate breakeven is 0.50 (edge = 0); confirm whether any group clears it reliably
   alongside an acceptable max_dd
4. A single operating threshold recommendation (or two if LONG/SHORT warrant different thresholds)

---

## Key Constraint — Per Q1

SHORT signals in recovery are suppressed regardless of experiment outcome.
Even if the sweep shows a positive edge row for SHORT/recovery at some threshold,
the signal count will be too thin to trust. Q1 already documented this.

---

## Findings

V10 results (2-class date-based regime proxy, sweep 0.01–0.20) have been moved to
[`edge-analysis/v10-edges.md`](edge-analysis/v10-edges.md).

Bravo results (7-regime classifier, extended sweep) are in
[`edge-analysis/Bravo/bravo-edges.md`](edge-analysis/Bravo/bravo-edges.md).

---

## Architecture Decision — Regime as Training Stratification (Option C)

**Decision date: 2026-03-13 · Status: Confirmed**

Regime labels are used as a **post-hoc slicing dimension** on LSTM outputs, not as a model input feature.

### What this means

- LSTM is trained with no regime column in the input features
- After inference, regime labels are attached to the output rows by timestamp
- Edge (win rate, max DD) is broken down per direction × regime
- Gate thresholds are set per-regime from the observed val/test breakdown

### Current regime proxy

The current `correction` / `recovery` split (line 165) is a **date-based proxy**:
```python
regime = np.where(ts_taken < BOUNDARY, 'correction', 'recovery')
```
This will be replaced with the 7-regime rule-based classifier labels from notebook 8 once the new pipeline (15m + extended dataset) is ready.

### Regime expansion plan

| Phase | Regime classes | Source |
|---|---|---|
| Current (v10 model) | correction / recovery (2 classes, date proxy) | Hard boundary 2026-02-07 |
| Next | ACCUMULATION / BULLISH / RECOVERY / DISTRIBUTION / BEARISH / CORRECTION / UNKNOWN (7 classes) | 8regime classifier on 15m-vX |

When the 7-regime labels are available, re-run this sweep with the regime column replaced. The loop at Step 3 will expand from `['correction', 'recovery']` to all 7 regime values. Regimes with `n < MIN_SIGNALS` are automatically suppressed.

### Why not a feature input

- Regime changes every hundreds of bars — low information density per 15m bar
- Risk of LSTM learning lazy regime shortcuts instead of price action patterns
- Post-hoc slicing gives the same gate decisions with a cleaner, independently validatable model
- This file already demonstrates the approach works: LONG/recovery edge emerged without regime in inputs