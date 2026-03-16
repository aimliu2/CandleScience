# Trader Buffer Plan — Position Sizing Decision

**Decision date: 2026-03-12**
**Selected sizing rule: 2% dynamic (equity-based)**

---

## Problem

Flat $10/trade on $170 capital = 100% blowup risk.
LONG/correction max drawdown of 17 × $10 = $170 — full wipeout.

---

## Solution: 2% Dynamic Sizing

Position size = **current equity × 2%**, recalculated each trade.
Hard cap: **$500 per position** (cap triggers at ~$25,000 equity — not relevant at current scale).

### Starting parameters

| Parameter | Value |
|---|---|
| Starting capital | $170.00 |
| Initial position size | $3.40 |
| Sizing rule | 2% of current equity |
| Position cap | $500 |

---

## Worst-Case Drawdown Scenarios at 2%

| Group | Max DD | Equity after streak | Equity lost |
|---|---|---|---|
| LONG / correction | 17 | $120.58 | $49.42 (−29.1%) |
| LONG / recovery | 10 | $138.91 | $31.09 (−18.3%) |
| SHORT / correction | 12 | $133.40 | $36.60 (−21.5%) |

No blowup scenario exists — size shrinks automatically with each loss.

---

## Why Not Higher?

| % rule | Equity lost at worst case (dd=17) | Verdict |
|---|---|---|
| 2.0% | −29.1% | Selected — survivable, recoverable |
| 2.2% | −31.5% | Marginal gain, slightly more pain |
| 2.4% | −33.8% | Approaching uncomfortable |
| 2.6% | −36.1% | Moderate risk zone |
| 2.8% | −38.3% | Close to 3% territory |
| 3.0% | −40.4% | Rejected — too aggressive for $170 base |

2% chosen as the floor that keeps worst-case drawdown under 30% while still generating
meaningful position sizes as equity grows.

---

## Recovery Behaviour

Because sizing is % of equity, recovery is self-compounding:
- Losses reduce size → slows further damage
- Wins on reduced size rebuild equity gradually
- No manual intervention required

---

## Live Simulation Results — Alpha_final.pt · thr=0.40 · $200 start

*Run date: 2026-03-12 · 2,319 active trades · Win rate: 53.04% across all runs*

| Sizing | Final capital | Net P&L | Max DD ($) | Max DD (%) |
|---|---|---|---|---|
| 1.0% | $729.54 | +$529.54 (+265%) | $131.79 | 18.7% |
| **2.0%** | **$2,110.75** | **+$1,910.75 (+955%)** | **$709.85** | **34.7%** |
| 2.5% | $3,291.63 | +$3,091.63 (+1,546%) | $1,350.42 | 41.8% |

### Per-group P&L at selected sizing (2%)

| Group | n | Win rate | P&L |
|---|---|---|---|
| LONG / correction | 1,047 | 52.6% | +$290.23 |
| LONG / recovery | 286 | 53.5% | +$612.65 |
| SHORT / correction | 986 | 53.3% | +$1,007.88 |
| SHORT / recovery | — | suppressed | — |

### Decision

**2% selected.** 2.5% produces 62% more capital but nearly doubles the dollar drawdown
($1,350 vs $709) and pushes max DD% to 41.8% — psychologically untenable on a $200 base.
1% is too conservative; it leaves significant compounding gains on the table.

---

## Q3 — CLOSED ✓

**Sizing rule locked: 2% of current equity · $500 hard cap · $200 starting capital**
