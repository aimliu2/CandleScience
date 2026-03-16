# Feature Engineering : Data Analysis

verify the quality of the data;
- Stationary - ADF, KPSS test
- Rolling z-score window - for some regime shifted data and retest with ADF, KPSS
- Local memory test - feature should employ local memory i.e. ACF test
- Mutual Information test : K-NN 3 and 10
- Spearman Test on feature vs label : if Mutual test fail (yield 0)