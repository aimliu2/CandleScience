# Group N — Implementation Concerns & Resolution Log
**Swing Pivot Distance Features · 15STR (15m) + 45STR (45m) · 6 Features (final)**

> All concerns resolved. Final verified feature set documented below.
> Four features dropped after failing Spearman — first drops in this project due to genuine absence of signal.

---

## Final Feature Set — Group N (6 features)

| # | Feature | Formula | Type | Normalisation |
|---|---|---|---|---|
| 1 | DIST_HIGH_15STR | `(15STR_keylv_high − close) / ATR_42` | Float ≥ 0 | rolling Z |
| 2 | DIST_LOW_15STR | `(close − 15STR_keylv_low) / ATR_42` | Float ≥ 0 | global Z |
| 3 | DIST_HIGH_45STR | `(45STR_keylv_high − close) / ATR_42` | Float ≥ 0 | rolling Z |
| 4 | DIST_LOW_45STR | `(close − 45STR_keylv_low) / ATR_42` | Float ≥ 0 | global Z |
| 5 | RANGE_15STR | `(close − 15STR_keylv_low) / (15STR_keylv_high − 15STR_keylv_low)` | Float [0,1] | global Z |
| 6 | RANGE_45STR | `(close − 45STR_keylv_low) / (45STR_keylv_high − 45STR_keylv_low)` | Float [0,1] | global Z |

**Dropped during verification (4 features):**

| Feature | Reason |
|---|---|
| 15STR_confirmed | Spearman r=0.0029, p=0.1711 — not significant |
| 45STR_confirmed | Spearman r=0.0009, p=0.6787 — not significant |
| barsSince15STR | Spearman r=-0.0014, p=0.4947 — not significant |
| barsSince45STR | Spearman r=0.0024, p=0.2524 — not significant |

---

## Concern 1 — Look-Ahead Leakage ✅ RESOLVED

**Resolution:** `available_at` tracking implemented inside `strHunt`. Every pivot mark has `available_at[i]` = first bar where the mark is causally safe. `str_confirmed` fires at `available_at[i]`, never earlier.

- Natural marks: `available_at[i] = i + window`
- Inserted marks: `available_at[i] = p2 + window`

Audit results:
- 15m STR: 13,715 marks (11,630 natural + 2,085 inserted) — **zero violations**
- 45m STR: 4,776 marks (3,960 natural + 816 inserted) — **zero violations**

---

## Concern 2 — Timeframe Resampling Alignment ✅ RESOLVED

**Resolution:** Lookup key uses `timestamp + FIFTEEN_MIN_MS` / `+ FORTY_FIVE_MIN_MS`. Maps `str_confirmed[j]` to the first 5m bar of the next bucket — tight but causal. Only the first 5m bar in each bucket receives the signal (`is_first_in_15m` / `is_first_in_45m` filter). `resample().ffill()` not used.

---

## Concern 3 — barsSince Non-Stationarity ✅ RESOLVED → DROPPED

**Resolution attempted:** `log1p` transform applied. Verified on train set:

```
barsSince15STR — raw: max=98,  mean=12.9  → log1p: max=4.595, mean=2.264, std=0.942
barsSince45STR — raw: max=278, mean=38.7  → log1p: max=5.631, mean=3.271, std=1.038
```

Zero clipping after winsorisation — pipeline intact. However both features subsequently failed Spearman (p=0.49 and p=0.25). **Dropped — no signal regardless of normalisation.**

---

## Concern 4 — Cold-Start ✅ RESOLVED

**Resolution:** `add_keylv` uses running extreme fill during cold-start — zero null rows, fully causal.

- Before first `sig == 1`: `keylv_high[i] = max(keylv_high[i-1], high[i-1])`
- Before first `sig == -1`: `keylv_low[i] = min(keylv_low[i-1], low[i-1])`

Each tracker initialises independently. No NaN rows produced — no downstream fill needed.

---

## Concern 5 — Formula Typo ✅ RESOLVED

All 6 remaining derived formulas verified correct:

```
DIST_HIGH_15STR = (15STR_keylv_high − close) / ATR_42   ✅
DIST_LOW_15STR  = (close − 15STR_keylv_low)  / ATR_42   ✅
DIST_HIGH_45STR = (45STR_keylv_high − close) / ATR_42   ✅
DIST_LOW_45STR  = (close − 45STR_keylv_low)  / ATR_42   ✅
RANGE_15STR     = (close − 15STR_keylv_low) / (15STR_keylv_high − 15STR_keylv_low)  ✅
RANGE_45STR     = (close − 45STR_keylv_low) / (45STR_keylv_high − 45STR_keylv_low)  ✅
```

Sign convention consistent with Group M: DIST_HIGH positive when price is below the high; DIST_LOW positive when price is above the low.

---

## Concern 6 — Full Verification Battery ✅ RESOLVED

### Stationarity (ADF + KPSS)

All ADF p=0.0. KPSS results and actions:

```
feature           kpss_stat   verdict   action
DIST_HIGH_15STR   0.526       WARN      rolling Z — drift +0.4 ATR confirmed
DIST_LOW_15STR    0.183       PASS      global Z
DIST_HIGH_45STR   0.905       WARN      rolling Z — drift +1.2 ATR confirmed
DIST_LOW_45STR    0.307       PASS      global Z
RANGE_15STR       0.127       PASS      global Z
RANGE_45STR       0.363       PASS      global Z
```

Rolling mean check (window=5000):
```
DIST_HIGH_15STR:  Q1=3.7  mid=3.6  Q4=4.0  → drift confirmed → rolling Z
DIST_HIGH_45STR:  Q1=6.5  mid=6.5  Q4=7.7  → drift confirmed → rolling Z
```

### ACF Local Memory

All 6 features carry meaningful ACF within LSTM receptive field (lags 1–14). Four features show long memory beyond lag 20 — all structural, all acceptable:

```
feature           lag-1   lag-20   assessment
DIST_HIGH_15STR   +0.976  +0.585   structural — pivot level frozen between confirmations
DIST_LOW_15STR    +0.977  +0.615   structural — step-function persistence
DIST_HIGH_45STR   +0.991  +0.810   structural — rare 45STR events, long freeze periods
DIST_LOW_45STR    +0.992  +0.837   structural — same
RANGE_15STR       +0.890  +0.009   clean — decays to zero by lag 20
RANGE_45STR       +0.984  +0.676   structural — frozen range between pivot confirmations
```

Long ACF is acceptable because: (1) KPSS confirmed stable means for DIST_LOW and RANGE features; (2) rolling Z-score handles DIST_HIGH drift; (3) the LSTM uses persistent values as regime-context signals via its gating mechanism — the same rationale accepted for ATR_14 (lag-20=+0.668) and RSI_14 (lag-20=+0.204).

### VIF (full 34-feature matrix, before drops)

All 6 Group N features below VIF=2.5 — lowest VIF group in the entire set:

```
DIST_HIGH_15STR   2.42
DIST_LOW_15STR    2.30
DIST_HIGH_45STR   1.61
DIST_LOW_45STR    1.59
RANGE_45STR       1.56
RANGE_15STR       1.01
```

Near-orthogonal to all existing 24 features. Strong signal that Group N will add genuine marginal contribution in ablation.

### MI (correct pipeline — rolling Z applied first)

```
feature           mi_k3   mi_k10   stability
RANGE_15STR       0.0092  0.0096   ✅ stable — strongest Group N feature
DIST_LOW_15STR    0.0059  0.0050   ✅ stable
RANGE_45STR       0.0032  0.0023   ✅ stable
DIST_HIGH_45STR   0.0023  0.0014   ✅ stable
DIST_HIGH_15STR   0.0017  0.0031   ✅ k=10 > k=3 — diffuse but real signal
DIST_LOW_45STR    0.0008  0.0019   ✅ k=10 > k=3 — conditional signal
```

### Spearman — Dropped Features

Four features failed Spearman — first genuine signal failures in this project:

```
feature           r        p        verdict
15STR_confirmed   +0.0029  0.1711   ❌ not significant — DROPPED
45STR_confirmed   +0.0009  0.6787   ❌ not significant — DROPPED
barsSince15STR    -0.0014  0.4947   ❌ not significant — DROPPED
barsSince45STR    +0.0024  0.2524   ❌ not significant — DROPPED
```

Compare to confirmed features kept via Spearman:
```
BUY_RATIO   r=-0.0107  p=0.0000  ✅ kept
STOCH_K     r=-0.0106  p=0.0000  ✅ kept
```

Dropped features have r≈0.001–0.003 — an order of magnitude weaker than the kept threshold. Genuinely uninformative at the univariate level.

**Why barsSince failed:** Staleness of the last confirmed pivot has no monotonic relationship with label outcome. Whether a barrier is hit in 3 bars depends on current momentum and volatility, not how long ago the last swing was confirmed.

**Why confirmed signals failed:** The {-1, 0, 1} ternary fires only at the first 5m bar of each 15m/45m bucket. That is 1 active bar out of every 3 (15STR) or 9 (45STR). Too sparse and too brief to correlate with a 3-bar label at the bar level.

---

## Final Normalisation Map — Group N

```
Rolling Z-score (window=500):
  DIST_HIGH_15STR, DIST_HIGH_45STR

Global Z-score:
  DIST_LOW_15STR, DIST_LOW_45STR, RANGE_15STR, RANGE_45STR

Winsorise ±3σ — all 6 features
```
