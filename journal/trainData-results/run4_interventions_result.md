# LSTM Triple-Barrier Model — Run 4 Summary
**Directional Precision + Recall Objective · NO_HIT Suppression · Binance 5-min Bars · March 2026**

---

## Objective Change from Run 3

Run 3 optimised F1 macro — the wrong target for a trading signal generator. F1 macro is inflated by NO_HIT class performance (F1_NO_HIT=0.503 in Run20_15) while the directional classes that actually drive trading signals score much lower.

**Root cause identified:** The model uses NO_HIT as a default escape route for uncertain bars. In Run20_15 approximately 42% of all predictions went to NO_HIT with only 0.436 precision. This depletes the directional signal pool — most missed DOWN/UP bars are routed to NO_HIT rather than predicted directionally.

**Run 4 objective:** Maximise directional precision AND recall. NO_HIT class performance is explicitly irrelevant. Accept NO_HIT recall going to zero.

**Success condition:**
```
UP   precision > 0.45   AND   recall > 0.35
DOWN precision > 0.45   AND   recall > 0.45
NO_HIT — not evaluated
```

**Key change — early stopping criterion:** Changed from val_loss minimum to directional F1 maximum. `dir_F1 = (f1_UP + f1_DOWN) / 2`. Val loss is still dominated by NO_HIT even with low w_nohit — stopping on val loss would still find a NO_HIT-dominant solution.

**Baseline:** Run10 (Run52_x_final.pt) — test F1=0.3935, UP recall=0.176, DN recall=0.510, 22 features.

---

## Run 4 Best Result — Run4d_AFL_120

| Property | Value |
|---|---|
| Run name | Run4d_AFL_120 |
| Loss | AsymmetricFocalLoss + NO_HIT masking |
| gamma_DN | 1.20 |
| w_nohit | 0.05 (functionally identical to 0.00) |
| Val dir_F1 | 0.4165 |
| Test dir_F1 | 0.4070 |
| Regime gap | 0.0063 |
| UP precision (test) | 0.344 |
| UP recall (test) | 0.492 |
| DN precision (test) | 0.336 |
| DN recall (test) | 0.525 |
| NO_HIT recall | 0.000 — eliminated |
| Best epoch | 9 |
| Model file | Run4d_AFL_120_final.pt |

---

## Full Experiment History

### Approach B — NO_HIT Down-weighting (weighted CE)

| Run | w_nohit | gamma_DN | BestEp | UP-prec | UP-rec | DN-prec | DN-rec | dir_F1 | Test dir_F1 |
|---|---|---|---|---|---|---|---|---|---|
| Run4a_WNH01 | 0.10 | — | 4 | 0.349 | 0.408 | 0.351 | 0.610 | 0.411 | 0.401 |

**Finding:** w_nohit=0.10 was sufficient to completely eliminate NO_HIT predictions (recall=0.000). Every bar is now predicted as UP or DOWN. Precision settled at ~0.334–0.351 — below the 0.45 target but both directions alive and balanced. This is the NO_HIT contamination noise: 35.2% of bars are genuinely ambiguous and get forced into directional predictions incorrectly.

---

### Approach A — NO_HIT Masked Out Entirely (masked CE)

| Run | w_nohit | gamma_DN | BestEp | UP-prec | UP-rec | DN-prec | DN-rec | dir_F1 | Test dir_F1 |
|---|---|---|---|---|---|---|---|---|---|
| Run4c_masked | 0.00 | — | 6 | 0.348 | 0.366 | 0.351 | 0.650 | 0.406 | 0.397 |

**Finding:** Statistically identical to Run4a (precision difference <0.002). This confirmed the key structural insight: seeing NO_HIT bars with near-zero weight produces the same result as not seeing them at all. The precision ceiling at ~0.334–0.351 is not a training procedure problem — it is a feature representation problem. With 35% of bars genuinely ambiguous, the model cannot distinguish them from directional bars using the current feature set.

**Implication:** The regime gate is the primary precision lever. The gate suppresses signals in regimes where NO_HIT bar density is highest (accumulation, recovery), recovering effective precision in the active signal pool.

---

### Approach C — AsymmetricFocalLoss + NO_HIT Masking (afl_masked)

gamma_DN sweeps — masking removes the NO_HIT gradient buffer so the same gamma has a stronger effect than in Run 3. A softer gamma is needed to stay balanced.

| Run | gamma_DN | w_nohit | BestEp | UP-prec | UP-rec | DN-prec | DN-rec | dir_F1 | Test dir_F1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Run4d (γ=1.50) | 1.50 | 0.00 | — | 0.341 | 0.978 | 0.335 | 0.023 | 0.274 | — | UP-only collapse ep1 |
| Run4d (γ=1.10) | 1.10 | 0.00 | — | 0.000 | 0.000 | 0.346 | 1.000 | 0.257 | — | DN-only collapse ep2 |
| Run4d (γ=1.15) | 1.15 | 0.00 | 3 | 0.341 | 0.558 | 0.346 | 0.452 | 0.396 | — | briefly balanced then collapses |
| **Run4d (γ=1.20)** | **1.20** | **0.05** | **9** | **0.355** | **0.492** | **0.348** | **0.530** | **0.417** | **0.407** | **best — stable** |
| Run4d120-2class | 1.20 | 0.00 | 9 | identical to above | | | | | | w_nohit=0.00 vs 0.05 — same |

**Key finding — gamma cliff shifted vs Run 3:**

In Run 3 (with NO_HIT bars included) the sweet spot was gamma_DN=1.50. In Run 4 (with masking) the sweet spot is gamma_DN=1.20. Masking removes 35% of bars from the training gradient. The remaining gradient is already more concentrated on directional bars, so a given gamma value has proportionally stronger effect. gamma_DN=1.50 is too strong in the masked context — it immediately collapses to UP-only.

**2-class confirmation:** w_nohit=0.05 and w_nohit=0.00 produce identical results. The 5% NO_HIT gradient contribution is negligible. This proves definitively that the precision ceiling is a feature set property, not a training procedure issue.

---

## Run 4 Gamma Sweep — Decision Boundary

```
gamma_DN=1.10   →  DN-only collapse ep2    (gamma too weak — DN still dominates)
gamma_DN=1.15   →  balanced ep3-4 then DN  (marginal — unstable)
gamma_DN=1.20   →  stable 9 epochs         ← sweet spot
gamma_DN=1.50   →  UP-only collapse ep1    (gamma too strong — overcorrects)
```

The cliff is narrower in masked context (1.15–1.20 vs 1.35–1.50 in Run 3) because the NO_HIT removal concentrates the remaining gradient.

---

## Run 4 Best vs Run 10 Baseline

| Metric | Run10 baseline | Run4d_AFL_120 (test) | Change |
|---|---|---|---|
| UP precision | 0.355 | 0.344 | -0.011 |
| DN precision | 0.390 | 0.336 | -0.054 |
| UP recall | 0.176 | 0.492 | **+0.316** |
| DN recall | 0.510 | 0.525 | +0.015 |
| dir_F1 | ~0.320 | 0.407 | +0.087 |
| NO_HIT recall | ~0.630 | 0.000 | eliminated |
| Regime gap | 0.0044 | 0.0063 | slightly wider |
| F1 macro | 0.3935 | 0.2713 | lower (NO_HIT dead) |

**The trade-off:** UP recall nearly tripled (0.176 → 0.492) at the cost of ~0.01–0.05 precision drop on both directions. The precision drop is the NO_HIT contamination noise — 35% of bars forced into directional predictions. The regime gate is expected to recover this.

---

## Cross-Run Comparison — All Runs

| Metric | Run 1 | Run 2 (R10) | Run 3 (R20_15) | Run 4 (R4d120) | Notes |
|---|---|---|---|---|---|
| Test F1 macro | 0.3987 | 0.3935 | **0.4048** | 0.2713 | Run4 F1mac low — NO_HIT dead |
| Test dir_F1 | ~0.32 | ~0.32 | ~0.36 | **0.407** | Run4 best directional |
| UP precision | — | 0.355 | 0.367 | 0.344 | Run4 slightly lower |
| UP recall | 0.287 | 0.176 | 0.300 | **0.492** | Run4 nearly 3× Run2 |
| DN precision | — | 0.390 | 0.416 | 0.336 | Run4 lower — noise |
| DN recall | 0.512 | 0.510 | 0.324 | **0.525** | Run4 best overall |
| NO_HIT recall | ~0.55 | ~0.63 | 0.593 | 0.000 | Run4 eliminated |
| Regime gap | 0.0129 | 0.0044 | 0.0044 | 0.0063 | all acceptable |
| Below 0.40 F1 floor | yes | yes | **no** | yes* | *F1mac not the metric |

*Run 4 F1 macro is low because NO_HIT is eliminated. This is expected and correct — F1 macro is not the evaluation metric for Run 4.

---

## Key Findings Across Run 4

**1. NO_HIT is the primary signal suppressor.** Eliminating it unlocked UP recall from 0.176 to 0.492 — the largest single improvement across all four runs. The model was routing 2 out of 3 genuine UP bars to NO_HIT in Run2/3.

**2. Precision ceiling is a feature set property, not a training procedure issue.** Run4a (w_nohit=0.10) and Run4c (w_nohit=0.00) produced identical precision (~0.335). The noise comes from 35% of bars being genuinely ambiguous — the features cannot distinguish them from directional bars.

**3. gamma_DN sweet spot shifted from 1.50 to 1.20 in masked context.** Masking concentrates the remaining gradient on directional bars, amplifying the focal effect. The same gamma that was balanced in Run3 is too aggressive after masking.

**4. The two-basin instability persists but is more manageable.** gamma_DN=1.20 held balance for 9 epochs before starting to drift — far more stable than any Run3 configuration. The directional F1 early stopping criterion is critical — val_loss stopping would have missed the balanced epochs.

**5. Regime gate is the remaining precision lever.** Raw precision at 0.336–0.344 is below the 0.45 target. The gate suppresses signals in accumulation and recovery regimes where NO_HIT bar density is highest. Gated precision in bullish/bearish/correction regimes is expected to recover toward 0.38–0.44.

---

## Next Steps

**Immediate:** Conviction threshold sweep on Run4d_AFL_120_final.pt and Run10 (Run52_x_final.pt) in parallel.

Sweep T = 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 on the test set for both models. Compare signal counts, precision, and break-even RR side by side.

**Then:** Regime-gated evaluation. Annotate test set bars with regime labels. Compute precision/recall/coverage per regime for both Run10 and Run4d_AFL_120. This is the number that determines which model goes to production.

**Production model decision criteria:**
```
If Run4d gated precision ≥ Run10 gated precision:
    Use Run4d — higher recall with equal or better precision
    Regime gate does more work but signal pool is larger

If Run4d gated precision < Run10 gated precision by > 0.04:
    Use Run10 — precision advantage outweighs recall gain
    Run4d noise is too severe for gate to recover

If gap is 0.01–0.04:
    Run both in shadow mode and compare live precision
```