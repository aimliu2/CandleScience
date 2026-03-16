# yo the feature produce strong signal in bear market
# but did the opposite in the bull market !? - ridiculous

# Pre-Run 3 Notes
**LSTM Triple-Barrier Model · UP Recall Collapse Investigation & Fix Plan**

---

## Context

Run 2 execution analysis was initiated but halted after the conviction threshold
sweep revealed that LONG signals are not functional on the test window. The root
cause was traced through four hypotheses before being conclusively identified.
This document records the full diagnostic chain and the planned interventions for Run 3.

---

## Run 2 Baseline (Reference)

| Property | Value |
|---|---|
| Model file | Run52_x_final.pt |
| Features | 22 (Groups I+J+K+L+M, BUY_RATIO and DELTA_DIV excluded) |
| Val F1 macro | 0.3979 |
| Test F1 macro | 0.3935 |
| Regime gap | 0.0044 |
| Best epoch | 3 |
| Test window | 2025-09-02 → 2026-03-11 (189 days, 54,569 sequences) |

---

## The Problem — UP Recall Collapse

The model structurally abstains from UP predictions. This is visible directly
in the classification reports from training.

**Val set (best epoch 3):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| DOWN | 0.4031 | 0.5331 | 0.4590 |
| NO_HIT | 0.4480 | 0.5319 | 0.4864 |
| UP | 0.3723 | **0.1863** | 0.2483 |

**Test set:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| DOWN | 0.3898 | 0.5097 | 0.4418 |
| NO_HIT | 0.4630 | 0.5523 | 0.5037 |
| UP | 0.3546 | **0.1759** | 0.2352 |

UP recall = 0.18 on both val and test. The model finds only 18% of actual UP
bars — missing 82% of genuine UP opportunities. DOWN recall is 0.51–0.53,
nearly 3× higher. This is not a test window composition issue — it is present
on the val set too, which covers a different regime.

**Conviction threshold evidence:**

- conv_up.max across 54,569 test sequences = 0.0531
- Zero LONG signals above thr=0.06 in any regime
- BULLISH regime: P(UP) > P(DN) in only 7.6% of bars
- BULLISH true labels: UP=34.6%, DOWN=32.6% — UP actually outnumbers DOWN
- The model predicts DOWN more than UP in a regime where UP labels dominate

---

## Diagnostic Chain — Four Causes Investigated

### Cause A — Label imbalance within regimes in training data
**Status: Eliminated**

Label distribution across all 546,381 training bars by regime:

| Regime | DOWN % | NONE % | UP % |
|---|---|---|---|
| ACCUMULATION | 34.7% | 34.5% | 30.8% |
| BULLISH | 31.4% | 34.7% | 33.8% |
| RECOVERY | 28.8% | 42.2% | 29.0% |
| DISTRIBUTION | 32.2% | 35.4% | 32.4% |
| BEARISH | 32.4% | 35.3% | 32.2% |
| CORRECTION | 33.0% | 38.2% | 28.8% |

Near-perfect balance within every regime. The model is not seeing
disproportionately more DOWN labels in any regime during training.

### Cause B — LSTM hidden state lag from bearish test window opening
**Status: Eliminated**

LONG signals appear from 2025-09-03 through 2026-03-10 in both ACCUMULATION
and BULLISH regimes — spread across the full 188-day test window, not
concentrated at the opening. P(UP) first half vs second half of BULLISH windows
is nearly identical (0.328 vs 0.319). No temporal improvement. Hidden state
persistence from the DISTRIBUTION opening is not the cause.

### Cause C — Test window features genuinely bearish at label level
**Status: Eliminated**

True label distribution in the test set by regime:

| Regime | DOWN % | NONE % | UP % |
|---|---|---|---|
| ACCUMULATION | 35.2% | 32.2% | 32.6% |
| BULLISH | 32.6% | 32.8% | **34.6%** |
| RECOVERY | 28.9% | 39.5% | 31.7% |
| DISTRIBUTION | 31.9% | 34.7% | 33.4% |
| BEARISH | 33.7% | 32.9% | 33.4% |
| CORRECTION | 33.8% | 35.7% | 30.6% |

BULLISH in the test set has MORE UP labels than DOWN (34.6% vs 32.6%).
The model outputs DOWN-biased softmax in a regime where ground truth disagrees.
The test window is not structurally bearish at the label level.

### Cause D — Feature representation asymmetry (CONFIRMED)
**Status: Root cause**

The features in this dataset are more discriminative for DOWN than UP.
When price falls, features like DELTA_1, DELTA_3, RSI_SLOPE, and ROC_3
produce strong, consistent negative values. When price rises, the same
features are noisier — bull moves tend to be more varied and less consistent
in their feature signatures. The loss surface during training has clearer
gradients for DOWN predictions, so the model gravitates there even with
balanced labels. The UP recall collapse is present from epoch 1 and the
model converges to this solution genuinely — early stopping is correct,
the model is simply converging to a DOWN-biased local optimum.

**Note on regime imbalance:** The full dataset regime distribution (15m bars):

| Regime | Count | % |
|---|---|---|
| BEARISH | 60,252 | 33.0% |
| ACCUMULATION | 57,698 | 31.6% |
| BULLISH | 35,633 | 19.5% |
| DISTRIBUTION | 16,876 | 9.2% |
| CORRECTION | 8,322 | 4.6% |
| RECOVERY | 4,014 | 2.2% |

Bearish group (BEARISH + CORRECTION + DISTRIBUTION) = 46.8%.
Bullish group (ACCUMULATION + BULLISH + RECOVERY) = 53.3%.
The dataset is slightly bullish-dominant. Regime resampling is not the fix.

---

## Run 3 — Planned Interventions

The binding constraint is UP recall. Target metrics for Run 3:

| Metric | Run 2 | Run 3 Target |
|---|---|---|
| UP recall | 0.18 | > 0.30 |
| DOWN recall | 0.51 | > 0.45 (must not collapse) |
| Val F1 macro | 0.3979 | > 0.40 |
| Test F1 macro | 0.3935 | > 0.40 |

Val F1 macro alone is no longer a sufficient training signal. UP recall must
be tracked explicitly at every epoch alongside val F1.

### Intervention 1 — UP Class Loss Weighting (Run First)

Apply a higher loss multiplier to UP misclassifications in CrossEntropyLoss.
No architecture change required.

```python
# Sweep UP class weight: [1.5, 2.0, 2.5]
# DOWN and NO_HIT weights held at 1.0
class_weights = torch.tensor([1.0, 1.0, W_UP])  # [DOWN, NO_HIT, UP]
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

Run three ablation experiments — W_UP = 1.5, 2.0, 2.5. Each run is ~27 minutes.
Pick the weight where UP recall crosses 0.30 without DOWN recall dropping below 0.45.

**Expected tradeoff:** Higher UP weight will reduce DOWN precision as the model
shifts probability mass toward UP. The acceptable tradeoff is UP recall >0.30
with DOWN precision not falling below 0.38.

### Intervention 2 — Class-Specific Focal Loss (If Intervention 1 Insufficient)

Apply higher gamma to DOWN class in focal loss, making it harder for the model
to get gradient from easy DOWN predictions. Forces attention toward UP.

```python
# Higher gamma for DOWN = down-weight easy DOWN examples more aggressively
gamma_per_class = torch.tensor([2.5, 1.0, 1.0])  # [DOWN, NO_HIT, UP]
```

This is more surgical than uniform loss weighting — it specifically targets
the mechanism (easy DOWN gradients dominating) rather than just penalising
UP misses.

### Intervention 3 — Separate UP/DOWN Conviction Heads (If Interventions 1+2 Insufficient)

Replace the single 3-class softmax with two independent binary classifiers:
- HEAD_UP: UP vs not-UP
- HEAD_DN: DOWN vs not-DOWN

Each head learns its own decision boundary independently. UP is no longer
competing against a stronger DOWN signal in a shared softmax.

This requires architecture change but no new features.

---

## Execution Order for Run 3

1. Add UP recall logging to training loop — track at every epoch alongside val F1
2. Run Intervention 1 sweep: W_UP = 1.5, 2.0, 2.5 (three ~27-min runs)
3. Evaluate: does UP recall cross 0.30 with DOWN recall still above 0.45?
4. If yes — lock best W_UP, run full evaluation and conviction threshold sweep
5. If no — proceed to Intervention 2, repeat evaluation
6. If still insufficient — proceed to Intervention 3 (architecture change)

---

## What Does Not Change in Run 3

- Feature set: 22 features unchanged (Groups I+J+K+L+M, BUY_RATIO and DELTA_DIV excluded)
- Architecture: 220k parameter baseline unless Intervention 3 is needed
- Dataset: same 546,381 bars, same 80/10/10 split
- Regime classifier: 7-regime hierarchical spec unchanged
- Execution analysis: resume from conviction threshold sweep once UP recall is fixed

---

## Execution Analysis Status at Pause Point

The Run 2 execution analysis was paused after the conviction threshold sweep.
The regime-segmented sweep confirmed SHORT signals are functional with genuine
edge across most regimes. LONG signals are not usable until Run 3 fixes UP recall.

When Run 3 produces a model with UP recall > 0.30, restart the execution
analysis from the conviction threshold sweep on the Run 3 test set.
Do not carry over any Run 2 threshold or gate action conclusions to Run 3 —
rerun everything from scratch on the new model outputs.