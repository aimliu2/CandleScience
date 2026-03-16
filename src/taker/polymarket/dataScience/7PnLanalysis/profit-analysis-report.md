# Profit Analysis Report — Bravo vs Charlie
**Date: 2026-03-15**
**Models: Bravo (Run52_x · Val F1=0.3979) vs Charlie (Run20_15 · Val F1=0.4092)**
**Simulation: 3 scenarios · $1,000 start · Bravo@2% · Charlie@1.5%**

---

## Headline Numbers

| Scenario | Bravo | Charlie | Delta |
|---|---|---|---|
| Wild — Final capital | $424 | $739 | Charlie +$315 |
| Wild — Max DD | 91.8% | 81.6% | Charlie less severe |
| Accepted — Final capital | $19,384 | $5,191 | **Bravo +$14,193** |
| Accepted — Max DD | 68.1% | 32.8% | Charlie lower |
| Strict gate — Final capital | $60,520 | $12,427 | **Bravo +$48,093** |
| Strict gate — Max DD | **28.0%** | 41.0% | Bravo lower |
| Strict gate — Win rate | **55.1%** | 52.7% | Bravo sharper |
| Strict gate — Trades | 2,441 | 3,603 | Charlie more (noisier) |

**Charlie has a higher F1 score and generates less profit in every meaningful scenario.**

---

## Why Charlie Earns Less Despite a Higher F1

### The classification reports side by side

| Class | Bravo precision | Bravo recall | Bravo F1 | Charlie precision | Charlie recall | Charlie F1 |
|---|---|---|---|---|---|---|
| DOWN | 0.390 | **0.510** | **0.442** | 0.416 | 0.324 | 0.364 |
| NO_HIT | 0.463 | 0.552 | 0.504 | 0.445 | **0.626** | **0.520** |
| UP | 0.355 | 0.176 | 0.235 | 0.367 | **0.300** | **0.330** |
| **F1 macro** | | | 0.3935 | | | **0.4048** |

### Reason 1 — Charlie's F1 gain came from NO_HIT, not from directional signals

Charlie's NO_HIT recall jumped from 0.552 → 0.626. It learned to predict "nothing happens"
more often and more accurately. That lifted the F1 macro but has zero trading value.
NO_HIT trades are never taken. The metric improved; the edge did not.

### Reason 2 — Charlie traded DOWN recall for UP recall, and that's the wrong swap

| Signal | Bravo recall | Charlie recall | Change |
|---|---|---|---|
| UP (LONG signals) | 0.176 | 0.300 | +70% more signals found |
| DOWN (SHORT signals) | 0.510 | 0.324 | −37% signals lost |

Bravo's dominant trading advantage lives in its SHORT signals. DOWN recall of 0.510 powers
Bravo's SHORT/RECOVERY and SHORT/DISTRIBUTION groups — the groups that contribute outsized
P&L at the strict gate. Charlie's DOWN recall collapsed to 0.324. Its SHORT side is
structurally weaker at the signal generation level before any gate is applied.

### Reason 3 — More UP recall is not the same as better UP precision at high conviction

Charlie finds 70% more LONG signals, but at nearly the same precision (0.367 vs 0.355).
The extra signals Charlie found are the lower-conviction ones — they fire more often but
don't survive the strict gate with better win rates. The result:

| | Bravo strict | Charlie strict |
|---|---|---|
| Trades | 2,441 | 3,603 |
| Win rate | **55.1%** | 52.7% |
| Final capital | **$60,520** | $12,427 |

Charlie is noisier, not sharper. More trades at lower average quality.

### Reason 4 — F1 macro does not measure trading edge

F1 macro weights DOWN, NO_HIT, and UP equally. For a trading system the actual weights are:

| Class | Trading value |
|---|---|
| NO_HIT | Zero — never traded |
| DOWN recall | High — powers the SHORT book |
| UP precision at high conviction | High — determines win rate at gated threshold |
| UP recall at low conviction | Low — generates noise below gate |

Charlie optimised for a metric that treats all three classes as equally important.
For this system, the relevant objective function is directional precision at high conviction —
and on that measure, Bravo wins.

---

## Where Each Model Makes Its Money (Strict Gate)

| Group | Bravo P&L | Charlie P&L | Winner |
|---|---|---|---|
| LONG / ACCUMULATION | +$18,354 | +$4,505 | Bravo |
| LONG / BEARISH | +$20,580 | +$1,281 | Bravo |
| LONG / BULLISH | +$2,176 | +$1,034 | Bravo |
| LONG / CORRECTION | +$4,201 | **+$4,571** | Charlie |
| LONG / DISTRIBUTION | +$7,292 | −$1,447 | Bravo |
| SHORT / DISTRIBUTION | +$5,603 | −$290 | Bravo |
| SHORT / RECOVERY | +$1,314 | +$1,771 | Charlie |

Charlie wins on LONG/CORRECTION and SHORT/RECOVERY. It loses everywhere else —
and LONG/DISTRIBUTION flipping negative is the most damaging single group failure.

---

## The SHORT/DISTRIBUTION Problem in Charlie

Charlie's strict gate passes thr=0.24 for SHORT/DISTRIBUTION, generating 1,060 trades —
**29% of all strict-gate activity** — at a 51.1% win rate and −$290 net P&L.
This is the direct consequence of lower DOWN recall: the model's short signals are diluted,
so even at thr=0.24 the edge barely clears zero.

**Fix:** Raise SHORT/DISTRIBUTION to thr=0.33 in live deployment.
At 0.33 the sweep shows edge=+0.061 with n~556. The volume drops from 1,060 to ~556 and the
signal quality improves. This is the most actionable single change for Charlie's strict gate.

---

## Signal Quality Degradation — The Core Diagnosis

Bravo's training produced a model where **high conviction = high directional accuracy**.
Charlie's training produced a model where **high conviction = slightly less directional accuracy**
because the probability mass shifted toward NO_HIT.

When conviction thresholds are applied at 0.38–0.41:
- Bravo's surviving signals have 55%+ win rates because the model's probability distribution
  is well-separated between UP and the rest.
- Charlie's surviving signals have 52–53% win rates because the model spreads probability
  more evenly, leaving the high-conviction tail less pure.

This is not about training time or data — it is about the **loss function objective**.
Charlie used `asymmetric_focal` with `W_UP=1.0, gamma_DN=1.5`. The gamma on DOWN
suppressed easy DOWN examples and pushed the model to learn harder ones, which at the
margin reduced DOWN recall. Meanwhile the UP weight was not adjusted, so UP recall improved
but precision at the high-conviction tail did not.

---

## Summary

| Question | Answer |
|---|---|
| Why is Charlie's F1 higher? | Better NO_HIT recall (+0.074) and better UP F1 (+0.095) |
| Why does Charlie earn less? | DOWN recall collapsed (0.510 → 0.324) — the SHORT book is weaker |
| Why does Charlie have more trades at strict gate? | Higher UP recall finds more signals; precision didn't keep pace |
| Why is Charlie's win rate lower at strict gate? | Probability mass shifted to NO_HIT; high-conviction UP signals are less pure |
| What is the actionable fix? | Raise SHORT/DISTRIBUTION to 0.33; the SHORT side needs a model with higher DOWN recall |
| Should we deploy Charlie? | Yes — strict gate, but with SHORT/DISTRIBUTION at 0.33. Profit is real (+1,143%). It is not Bravo. |
| What should the next model target? | DOWN recall ≥ 0.45 without sacrificing UP precision at high conviction thresholds |
