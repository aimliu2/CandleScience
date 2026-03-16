# Regime Classifier — Production Parameters & Known Issues

> Implementation notes to resolve before live deployment.
> Companion to regime-classifier-spec.md.

---

## Issue 1 — EMA Initialization Error (safe fetch floor)

EMA is recursive: `EMA[t] = α·price[t] + (1-α)·EMA[t-1]`. On a cold fetch with
no prior history, bar 1 initializes with `EMA[1] = price[1]`. This error decays
exponentially but is not negligible until ~`2 × period` bars have passed.

| EMA | Period | Error decayed after | Status at bar 1000 |
|---|---|---|---|
| 1H fast | 20 | ~40 bars | 960 bars past decay — negligible |
| 1H slow | 50 | ~100 bars | 900 bars past decay — negligible |

**Safe fetch floor: 200 × 1H bars minimum.**

At 1000 bars this is not a problem. But if the fetch window is ever reduced
(e.g. for performance), do not drop below 200 × 1H bars or the EMA slow
becomes unreliable and group assignment degrades silently.

**Status:** Not yet validated on live data. Confirm EMA stability
by comparing cold-start EMA to pre-warmed EMA on the same timestamp.

---

## Issue 2 — DISTRIBUTION Divergence Lookback (open parameter)

The DISTRIBUTION rule "RSI diverging ↓ from > 60" is a **comparison**, not a
threshold. It requires looking back N bars to find where RSI was last above 60
and comparing it to the current value while checking price has not made a lower
low.

```
divergence = RSI_peak_N_bars_ago > 60
             AND RSI_current < RSI_peak_N_bars_ago
             AND price_current >= price_at_RSI_peak   # price not lower
```

**The lookback window N is an open parameter — not yet defined.**

Draft starting value: N = 20 bars (5h on 15m). This is well within the 1000-bar
fetch budget regardless of value, but the parameter itself must be set and
validated against the 2021-08 and 2025-08 DISTRIBUTION periods before
deployment.

**Status:** Not yet set. Add to parameter validation pass after data fetch.

---

## Issue 3 — Last Bar Is Open (must drop in production)

**fixed pre-computation problem : use Async + cache**

The exchange API (Binance klines) returns the **currently open, not-yet-closed
bar** as the last entry in every response. If the classifier runs on it:

- `vol` and `taker-buy-basevol` are partial → `taker_buy_ratio` is wrong
- `high` and `low` are incomplete → ATR is underestimated
- `close` is the current mid-price, not the bar's final close → RSI and EMA
  are computed on a value that will change before the bar closes

**Fix:** always slice off the last bar before computing indicators.

```python
df = df.iloc[:-1]   # drop last (open) bar — classify on df.iloc[-1] after this
```

This is a one-line implementation rule but if omitted it will silently corrupt
DISTRIBUTION and CORRECTION signals on every live call, since both depend on
the most recent bar's volume and RSI values.

**Status:** Must be enforced in implementation. Add assertion in tests:
last bar timestamp must be a closed bar (timestamp + bar_duration <= now).
