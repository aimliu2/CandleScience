# The three realistic exit options:
### Option A — Fixed bar exit. 
Close after N bars regardless of outcome. Simple, no lookahead, easy to backtest. The natural candidate is 3 bars since that is the label horizon the model was trained on. You know the model's precision over that exact window.
### Option B — ATR-based stop + fixed target. 
Set a stop at some fraction of ATR_14 below entry, target at RR × stop distance. The break-even RR of 1.27 tells you the minimum target/stop ratio needed. This is more realistic as an actual trade but introduces two free parameters.
### Option C — Hold until opposite signal. 
Stay in the trade until the model fires a signal in the opposite direction. Given the clustering, this could mean very short holds or occasionally very long ones depending on what fires next.


# Other entry condition
- I'm assuming one position at a time (skip signal if already in trade)
- or stacked position whenever the signal fired ?

# Example plan
- First, the actual P&L distribution under the 3-candle hold rule — mean return, median, std, and the shape of the distribution. Are losses fat-tailed? Are winners capped by the fixed exit?
- Second, the realised RR — average winning trade return divided by average losing trade return. If realised RR > 1.27, the strategy is profitable at threshold=0.10. If it's below 1.27, it isn't, regardless of precision.
- Third, regime breakdown of returns — does the 3-candle hold produce different return profiles in correction versus recovery? Given that correction has symmetric signal quality and recovery is long-only, the P&L profiles will likely differ.