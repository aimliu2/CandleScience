# New dataset (2021-08-01)

# All 24 features retained. No drops.

# One normalisation change from Run1:
  DIST_LOW_5  →  rolling Z  (was global Z)
  DIST_LOW_10 →  rolling Z  (was global Z)

# Final normalisation map:
  Rolling Z (11): DELTA_1, DELTA_3, ATR_5, ATR_14,
                  STOCH_K, DIST_HIGH_5, DIST_HIGH_10, RANGE_POS,
                  DIST_LOW_5, DIST_LOW_10
  Global Z  (13): ROC_3, ROC_5, ROC_10, MOM_3, RETURNS_1,
                  ATR_RATIO, ATR_NORM_ROC, RANGE_RATIO,
                  RSI_14, RSI_SLOPE, CCI_5,
                  VOL_SPIKE, DELTA_DIV <-- may have to drop

Winsorise ±3σ — all features after Z-score