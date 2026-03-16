# Regime Analysis

### Number of Regimes — 7 (Hierarchical 3+3 + UNKNOWN)

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

read more on [regime-spec-final](./regime-spec-final.md)