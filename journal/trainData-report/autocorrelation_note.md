# Autocorrelation Notes

## 95% Confidence Band: `confidence_interval = 1.96 / sqrt(N)`

### Derivation

This comes from the **Bartlett approximation** for the standard error of an autocorrelation coefficient.

**Null Hypothesis:** The time series is white noise — no autocorrelation at any lag.

Under H₀, the autocorrelation at lag *k*, denoted **r(k)**, is approximately normally distributed:

```
r(k) ~ N(0, 1/N)
```

Standard error:

```
SE = 1 / sqrt(N)
```

For a two-tailed test at α = 0.05, use z-score **1.96** (97.5th percentile of the standard normal):

```
CI = ±1.96 × SE = ±1.96 / sqrt(N)
```

---

### Interpreting Lags That Exceed the Band

| Situation | Interpretation |
|---|---|
| `r(k) > 1.96/√N` | Significant **positive** autocorrelation at lag k |
| `r(k) < -1.96/√N` | Significant **negative** autocorrelation at lag k |
| Within band | Consistent with white noise — no significant pattern |

Exceeding the band = **reject H₀** — the correlation at that lag is unlikely to be zero.

---

### Caveats

- **Multiple testing:** With many lags, ~5% will exceed the band by chance. Use Bonferroni or adjust α.
- **Small N:** The normal approximation breaks down.
- **Biased SE:** Bartlett's `1/√N` assumes all other lags are truly zero; if the series has strong autocorrelation, SE is underestimated.
- **99% band:** Use **2.576** instead of 1.96.

---

### Practical Takeaway

A lag exceeding the band suggests **periodic structure or memory** in the data at that lag.
Example: if lag 5 is significant in a daily dataset → possible weekly pattern.
