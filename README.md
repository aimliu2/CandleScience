# Candle science
Pipeline for Tradeing Bot Brain (Machine learning)


# Taker : Polymarket
1. [Regime Analysis](src/taker/polymarket/dataScience/8regimeAnalysis/README.md) - Classify what regime are we in.
2. Feature Engineering
    - [Prep Data](src/taker/polymarket/dataScience/1prepData) - Prepare data and labels
    - [Augmented](src/taker/polymarket/dataScience/2augmented) - Add Features groups
    - [Analyse](src/taker/polymarket/dataScience/3analysis) - Analyse data/ feature's quality what to keep/drop/transform
3. [Train-test split](src/taker/polymarket/dataScience/4trainData) - Split Train/Val/Test data. tThe data must be chornologically sorted for time series data
    - [Train/Val/Test](src/taker/polymarket/dataScience/4trainData) - split data, ensure label balance in Train/ Val/ Test. employ rolling-scaler, standard scaler, winsorize as needed
4. [Training Model](src/taker/polymarket/dataScience/5runEpoches) - I don't have GPU. sad :(. Want to try transformer
5. [Execution Analysis](src/taker/polymarket/dataScience/6excutionAnalysis) - Model may have low condifident, but conviction and how it was executed are another story.
    - [Regime Annotation](src/taker/polymarket/dataScience/8regimeAnalysis) - Map regime classifer that was defined in 1. to the test dataset
    - [Conviction sweep analysis](src/taker/polymarket/dataScience/6excutionAnalysis) - May have to run conviction sweep per regime if the test data hasn't cover all regime.
6. [Live-trade analysis](src/taker/polymarket/dataScience/7PnLanalysis) - When you don't validate a profit by the label in live trade i.e. add entry-exit rule 0.7RR
    - [ohlc re-mapped](src/taker/polymarket/dataScience/7PnLanalysis/7remap-feature-ohlcv.py) Map ohlc to create new validation matrix
7. [Analys Edges and acceptable risk](src/taker/polymarket/dataScience/7PnLanalysis/edge-analysis) - No model has F1 score exceed 0.65, how will you survie the trade ?
8. [Simulate trades with acceptable risk](src/taker/polymarket/dataScience/7PnLanalysis/simulation) - Go big or go home :)
9. [Deployment](journal/deployment) - to the live trade !


