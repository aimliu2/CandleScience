# Training Notes

### 2026-03-08 - use Binary interpretation instead of Ternary

- Train a binary classifier (upper hit vs lower hit) but skip timesteps where label = 0, while preserving the sequential order for LSTM's hidden state.

- The method was to skip the Gradient Update, But Feed the Sample Forward
```
sequence: [t1=1, t2=0, t3=-1, t4=0, t5=1]
forward pass:  all timesteps feed through LSTM → hidden state updates normally
backward pass: only compute loss on t1, t3, t5 (label ≠ 0)
```

- This works and is theoretically sound. The LSTM sees all market context including the "nothing happened" bars, but is only penalised on decisive moments. This is essentially masked loss — well established in NLP for padding tokens and sequence labelling.

- But the model will have selection bias problem:
The training set now only contains bars where a barrier was eventually hit. At inference time, the model receives all bars including the majority that resolve to 0. The model has never learned what "ambiguous, low-conviction" market structure looks like — it has only seen decisive moments.
```
Training distribution:  P(features | label ∈ {1, -1})
Inference distribution: P(features | label ∈ {-1, 0, 1})
```

- These are different distributions. The model will be overconfident — it will force every bar into either "up" or "down" because it was never trained to say "neither."


- A simpler idea, separable prediction — first decide if something happens, then decide what. That's a two-stage model, which actually works:
```
Stage 1 (all bars):     
Binary → will a barrier be hit?   P(hit | features)
Stage 2 (hit bars only): 
Binary → which direction?         P(up | hit, features)
```
- Stage 2 trains exactly on Binary interpretation but now the selection bias is explicit and controlled. Stage 1 's confident must be controlled first, and only run Stage 2 when Stage 1 is confident.

- This also maps cleanly onto trading logic: Stage 1 is trade or sit-on-the-hand, Stage 2 is direction.

```
          ┌─────────────────────┐
          │   LSTM Stage 1      │  trained on ALL bars
          │   "Trade Filter"    │  label: {0=no hit, 1=hit (either dir)}
          └────────┬────────────┘
                   │ P(hit) > threshold (e.g. 0.60)
                   ▼
          ┌─────────────────────┐
          │   LSTM Stage 2      │  trained ONLY on hit bars (your original idea)
          │   "Direction Bet"   │  label: {1=up, -1=down}
          └────────┬────────────┘
                   │
                   ▼
          P(up) = P(hit) × P(up|hit)   → long signal
          P(dn) = P(hit) × P(dn|hit)   → short signal
```

- On the other hand, Ternary label was easier to maintain

### Conclusion : When to use Two-stage model
- If the label=0 frequency is very high (>70% of bars), the two-stage approach will likely win meaningfully. 

- If it's closer to 50%, the single model with focal loss may be competitive and simpler to maintain.