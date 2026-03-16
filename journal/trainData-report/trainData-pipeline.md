# The Full Sequence End to End

Summarize on how to build the training data (including Data pipe line)
```
1. Fetch raw OHLCV + taker_buy_vol from Binance
        │
2. Compute ATR_42
        │
3. Compute triple-barrier labels
        │
4. Drop 41 NaN warmup rows (ATR_42 warmup)
        │                          ← df_clean starts here
5. run_pipeline(df_clean)
        │
        ├── Step 1: compute ROC, MOM_3, DELTA_1, etc.
        ├── Step 2: rolling Z-score on DELTA_1, DELTA_3, BUY_RATIO
        ├── Step 3: drop first 500 rows (rolling Z-score warmup)
        ├── Step 4: temporal split 70/15/15
        ├── Step 5: global Z-score (fit on train only)
        ├── Step 6: winsorise ±3σ
        └── Step 7: build LSTM sequences
```