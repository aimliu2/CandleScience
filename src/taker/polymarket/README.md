# Polymarket 

### Choosing - Crypto Binary Option
- On Polymarket. there were [Crypto] up down in 5m,15m,1D,etc. section
- Fundamentally, It was binary option on blockchain
- The [Crypto] price was referenced from **Chainlink** Data

### Ideas
- My idea was to create [Taker-Bot]
- Sudden idea was to predict the up, down of the next candle. Eventually, I have to concluded that idea was silly, too random like a coin flip. No **ML** model could learn to predict a 50/50 outcome without a pattern.
- Thus, **assumption1** : the outcome can not be framed as a coin flip. 
- And **assumption2** : there should be an input(features) that has enough pattern to be recognized.

### Assumption1: Outcome Reframed
- Given the subject **is next 15m candle up or down ?**
- Instead of up,down let's reframe the outcome to **will the price move _+range_ upward in the next 3 5m bars?**
- I picked **ATR** for ease of computation. 14 is arbitrary number. Thus, the outcome will be **will the price move _+ATR14_ upward in the next 3 5m bars?**
- Translate the outcome to math, using Triple-Barrier convention, now I could label 1,0,-1 to the outcome
- this brings the restriction to the input(features). The features must encode **local memory**

### Assumption2: Features Reframed
- Shame, the famous FracDiff is useless here
- a good candidate for my case would be something like
```
Rate of change over last 3–10 bars
ATR ratio (current volatility vs. recent volatility)
RSI or momentum over 5–14 bars
Order flow imbalance (if available)
Distance to recent swing high/low
```
- which are simply technical indicator on **ohlc** chart
- I'll use trial&error, ablation to prove which features works or not

### The Bot will be : scalper predictor
- see [Ablation Design](./ablations/README.md)