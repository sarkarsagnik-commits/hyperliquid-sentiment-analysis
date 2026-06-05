# Hyperliquid Sentiment × Trader Performance: Quantitative Research Report

**Author:** Sagnik Sarkar  
**Date:** June 2025  
**Assignment:** Primetrade.ai — Data Science Hiring Task

---

## Executive Summary

This report investigates the statistical relationship between Bitcoin market sentiment, as measured by the Fear & Greed Index (FGI), and trader performance on the Hyperliquid decentralized exchange. The analysis covers 211,181 trade fills across 32 accounts over a 2-year period (May 2023 – May 2025).

Principal findings:

1. **Mean daily PnL is directionally higher in fear-regime markets, but the difference is not statistically significant under parametric tests.** Fear/Extreme Fear mean daily PnL ($5,185) exceeds Greed/Extreme Greed ($4,188) by $998 per account-day. However, the ANOVA p=0.671 fails to reject the null. The Kruskal-Wallis test (p=0.002), which is robust to the heavy-tailed PnL distribution, does detect significant distributional differences across regimes. Cohen's d=0.033 indicates a negligible effect size. The 95% CI for the Fear-vs-Greed mean difference spans (−$1,464, +$5,365), crossing zero.

2. **A regime-based contrarian sizing strategy produced the strongest risk-adjusted backtest performance** (Sharpe 2.96, Sortino 5.46, total return $3.87M) among the three strategies tested. This result is subject to the limitations described in Section 4 and should not be interpreted as evidence of a live trading edge without out-of-sample validation.

3. **Month-to-month PnL autocorrelation is positive and statistically significant** (r=0.194, p=0.008), consistent with — though not proof of — performance persistence across traders.

4. **33.5% of total realized PnL is concentrated in extreme sentiment regimes** (Extreme Fear + Extreme Greed), which account for 26.7% of trading days.

5. **Trading volume is negatively associated with sentiment** (r=−0.074, p<0.001); traders exhibit higher activity during fear-regime periods.

---

## 1. Dataset Overview

### 1.1 Fear & Greed Index
- **Source:** Bitcoin Fear & Greed Index API
- **Period:** February 2018 – May 2025 (2,648 days after gap-filling)
- **Variables:** Daily sentiment score (0–100) with five regime classifications
- **Quality:** Near-perfect — 4 missing days forward-filled, no other issues

### 1.2 Historical Trader Data
- **Source:** Hyperliquid DEX trade logs
- **Period:** May 2023 – May 2025
- **Records:** 211,181 fills (after removing 43 dust trades with Size USD = 0)
- **Accounts:** 32 unique Ethereum wallets
- **Coins:** 246 unique trading pairs
- **Total Realized PnL:** $10,296,959

### 1.3 Data Decisions

| Decision | Rationale |
|---|---|
| `Order ID` as trade identifier | `Trade ID` has 208K duplicates — not unique. Individual rows are fill-level executions sharing an Order ID. |
| No leverage estimation | Leverage is not directly observable in the dataset. Analysis uses position size (`Size USD`), exposure, and margin type (`Crossed`/`Isolated`) instead. This is documented as a material limitation. |
| Forward-fill FGI gaps | 4 missing days (likely API outages near 2018-04-14); forward-fill preserves regime context. |
| Remove Size USD = 0 | 43 dust/rounding trades removed — no analytical value. |

---

## 2. Methodology

### 2.1 Feature Engineering

**Sentiment Features:**
- Rolling means (7/14/30-day) and volatility of FGI value
- Sentiment momentum (day-over-day change)
- Sentiment z-score (deviation from 30-day rolling mean, normalized by rolling standard deviation)
- Regime transition indicators and consecutive-days-in-regime counters

**Trader Features (per account, per day):**
- Daily PnL (sum of realized Closed PnL), win rate, profit factor
- Position size statistics (mean, max, total volume in USD)
- Long/short bias (proportion of long-side trades), cross-margin ratio
- Rolling 30-day Sharpe ratio and rolling win rate

**Trader Segmentation:**
- Top 10% (4 traders): cumulative PnL ≥ 90th percentile
- Bottom 10% (4 traders): cumulative PnL ≤ 10th percentile
- Middle 80% (24 traders): remainder

### 2.2 Statistical Framework

All hypothesis tests report three dimensions:
1. **Statistical significance:** p-values with Bonferroni correction (for pairwise families) and Benjamini-Hochberg FDR correction (for correlation matrices)
2. **Effect size:** Cohen's d with qualitative interpretation (|d|<0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, >0.8 large)
3. **Economic significance:** Dollar-denominated magnitude of the observed difference

**Robustness protocol:** Major findings tested across three specifications:
- Full sample (n=2,338 account-days)
- Winsorized at 1st/99th percentiles
- IQR-filtered (excluding observations beyond ±1.5×IQR from the PnL quartiles)

### 2.3 Causal Limitations

All findings in this report are associational. The FGI co-moves with Bitcoin price, which directly determines trader PnL. Without an exogenous instrument or natural experiment, we cannot isolate a causal sentiment effect from the confounding influence of concurrent price action. Recommendations should be interpreted as conditional correlations warranting further investigation, not as established causal mechanisms.

---

## 3. Key Findings

### 3.1 Sentiment-Performance Relationship

#### Per-Regime Daily PnL Statistics

| Regime | N (account-days) | Mean PnL | Median PnL | Std Dev |
|---|---|---|---|---|
| Extreme Fear | 160 | $4,619 | $218 | $29,535 |
| Fear | 630 | $5,329 | $108 | $31,660 |
| Neutral | 376 | $3,439 | $168 | $17,448 |
| Greed | 649 | $3,378 | $159 | $30,614 |
| Extreme Greed | 523 | $5,192 | $428 | $27,573 |

#### Grouped Comparison

| Group | N | Mean Daily PnL |
|---|---|---|
| Fear + Extreme Fear | 790 | $5,185 |
| Greed + Extreme Greed | 1,172 | $4,188 |
| Difference (Fear − Greed) | — | +$998 |

#### Hypothesis Tests: Daily PnL Across Regimes

| Test | Statistic | p-value | Interpretation |
|---|---|---|---|
| One-way ANOVA | F=0.588 | 0.671 | Not significant — fails to reject H₀ of equal means |
| Kruskal-Wallis | H=16.50 | **0.002** | **Significant** — detects distributional differences robust to non-normality |
| Welch's t-test (Fear vs Greed) | t=1.12 | 0.263 | Not significant |
| Mann-Whitney U (Fear vs Greed) | U=202,684 | 0.788 | Not significant |

**Fear vs Greed pairwise detail:**
- Mean difference: +$1,950; 95% CI: (−$1,464, +$5,365) — **interval crosses zero**
- Cohen's d = 0.063 (Negligible)
- Economic magnitude: $1,950 per account-day, but high uncertainty

**Extreme Fear vs Extreme Greed pairwise detail:**
- Mean difference: −$572; 95% CI: (−$5,723, +$4,578) — **interval crosses zero**
- Cohen's d = −0.020 (Negligible)
- Note: Extreme Greed ($5,192) marginally exceeds Extreme Fear ($4,619), reversing the direction seen in the broader Fear/Greed grouping

**Interpretation:** The parametric ANOVA and t-tests do not reach significance, driven by extreme variance in leveraged trading PnL (standard deviations exceeding $27K against mean differences under $2K). The Kruskal-Wallis test, which does not assume normality and is sensitive to median and distributional shifts, reaches p=0.002. This suggests that the PnL *distributions* differ meaningfully across regimes even though mean differences are not statistically distinguishable. The pattern is directionally consistent but should not be treated as a confirmed mean-shift effect.

### 3.2 Trade Activity by Regime

| Test | Statistic | p-value |
|---|---|---|
| ANOVA (daily trade count) | F=3.18 | **0.013** |
| Kruskal-Wallis | H=23.20 | **<0.001** |
| Extreme Fear vs Extreme Greed (t-test) | t=2.78 | **0.006** |

**Per-regime mean daily trade count:** Extreme Fear 133.8, Fear 98.1, Neutral 100.2, Greed 77.5, Extreme Greed 76.4.

Extreme Fear vs Extreme Greed: mean difference = +57.3 trades/day; 95% CI: (17.0, 97.7); Cohen's d = 0.329 (Small).

Trade frequency is significantly higher in Extreme Fear compared to Extreme Greed (p=0.006), with a small but meaningful effect size. This is consistent with the negative volume-sentiment correlation (r=−0.074, p<0.001).

### 3.3 Position Size by Regime

Position size variation across regimes does not reach conventional significance (ANOVA p=0.071; Kruskal-Wallis p=0.294).

**Behavioral observation (significant):** Mean position size during Greed/Extreme Greed ($5,979) is significantly lower than during Fear/Extreme Fear ($8,531); Welch's t-test p=0.037, Cohen's d = −0.113 (Negligible). The 95% CI for the difference is (−$4,948, −$156). Despite statistical significance, the effect size is negligible, and the finding should be considered preliminary.

Position size coefficient of variation is highest in Fear (4.10) and lowest in Extreme Greed (1.52), indicating more heterogeneous sizing behavior during fear-dominated markets.

### 3.4 Trader Segmentation

| Metric | Top 10% (4 traders) | Middle 80% (24 traders) | Bottom 10% (4 traders) |
|---|---|---|---|
| Total PnL range | $840K – $2.14M | $14.9K – $836K | −$168K – $14.5K |
| Mean Sharpe Ratio | 6.40 | 5.52 | −1.09 |
| Mean Win Rate | 85.9% | 87.3% | 69.4% |
| Mean Position Size | $17,549 | $7,044 | $10,932 |
| Mean Profit Factor | 6,814 | ∞ (no losses) | 0.70 |
| Mean Long Bias | 60.0% | 54.9% | 22.2% |

Notable: Bottom 10% traders exhibit a markedly lower long bias (22.2% vs 60.0% for Top 10%), suggesting they are predominantly short-biased. The Middle 80% cohort contains one trader with an inf profit factor (zero realized losses over the observation period) and a Sharpe of 10.42, representing an outlier.

**Caveat:** With only 4 traders per tail cohort, these comparisons have very low statistical power and should be interpreted as descriptive, not inferential.

### 3.5 Behavioral Analysis

| Hypothesis | Test | Statistic | p-value | Cohen's d | Economic Impact |
|---|---|---|---|---|---|
| Position size higher in greed vs fear | Welch's t | t=−2.09 | **0.037** | −0.113 (Negligible) | −$2,552 mean difference |
| Losses larger in Extreme Greed vs other | Mann-Whitney U | U=2,824 | 0.065 | −0.143 (Negligible) | −$4,580 avg loss difference |
| PnL higher in fear markets vs greed | Welch's t | t=0.71 | 0.477 | 0.033 (Negligible) | +$998/day |
| Top 10% are contrarian (long_ratio vs sentiment) | Pearson r | r=−0.191 | **<0.001** | — | Negative correlation: reduce long exposure as sentiment rises |
| Bottom 10% are trend-following | Pearson r | r=+0.124 | 0.086 | — | Positive but not significant |

**Key behavioral finding:** Top-performing traders show a statistically significant contrarian tendency (r=−0.191, p<0.001), reducing their long exposure as sentiment increases. Bottom-performing traders show the opposite pattern (r=+0.124), though this does not reach significance (p=0.086). The difference in behavioral orientation between cohorts is suggestive but should be validated on a larger sample.

**Losses in Extreme Greed:** Contrary to the initial hypothesis, average losses during Extreme Greed ($6,110) are *smaller* than in other regimes ($10,690). This difference approaches but does not reach significance (p=0.065).

### 3.6 Predictive Lag Analysis

| Lag | Target: daily_pnl | Target: mean_position_size | Target: daily_trades |
|---|---|---|---|
| 1-day | r=−0.008, p=0.686 | r=−0.038, p=0.069 | r=−0.060, **p=0.004** |
| 3-day | r=−0.005, p=0.797 | r=−0.060, **p=0.004** | r=−0.064, **p=0.002** |
| 7-day | r=−0.013, p=0.541 | r=−0.035, p=0.109 | r=−0.087, **p<0.001** |

**PnL predictability:** Lagged sentiment shows no significant correlation with future daily PnL at any lag (all p>0.5). Sentiment alone does not predict next-day, next-3-day, or next-7-day PnL in this sample.

**Trade count predictability:** Lagged sentiment significantly predicts future trade frequency at all three lags (p≤0.004). Higher sentiment today is associated with fewer trades in subsequent days. This is consistent with the contemporaneous volume-sentiment inverse finding, suggesting it is not merely a coincidence of same-day measurement.

**Position size predictability:** The 3-day lag shows significant negative correlation (r=−0.060, p=0.004), suggesting traders reduce position sizes several days after sentiment peaks.

---

## 4. Strategy Research

### 4.1 Strategy Comparison

| Strategy | Total Return | Sharpe | Sortino | Max Drawdown | Win Rate | Profit Factor | Trades |
|---|---|---|---|---|---|---|---|
| Sentiment Reversal (Contrarian) | −$2.52M | −1.23 | −1.06 | −$5.14M | 19.9% | 0.55 | 18 |
| Sentiment Momentum | −$1.20M | −0.73 | −0.39 | −$2.65M | 53.5% | 0.70 | 93 |
| **Regime Contrarian Sizing** | **+$3.87M** | **2.96** | **5.46** | **−$320K** | **65.5%** | **4.20** | **131** |
| *Benchmark (Buy & Hold)* | *$10.30M* | — | — | — | — | — | — |

### 4.2 Analysis: Regime-Based Contrarian Sizing

**Description:** Position size is set inversely proportional to sentiment greed level:
- Extreme Fear: 100% of reference position (full long)
- Fear: 75%
- Neutral: 50%
- Greed: 25%
- Extreme Greed: −25% (small short)

**Performance characteristics:** The strategy produced a Sharpe ratio of 2.96 and Sortino of 5.46 with a maximum drawdown of $320K — significantly smaller than the other two strategies. The win rate of 65.5% and profit factor of 4.20 indicate asymmetric payoff distribution favoring profitable days.

**Why two strategies failed:** The Sentiment Reversal strategy makes discrete long/short bets on extreme sentiment reversals, but extreme regimes persist (the transition matrix shows >50% same-regime probability for most regimes), causing prolonged drawdowns during trending sentiment. The Momentum strategy suffers from signal whipsawing.

**Critical limitations — results are in-sample only:**
- No transaction costs or slippage modeled
- Uses close-of-day sentiment (introduces 1-day signal lag in practice)
- Backtest uses aggregate trader PnL as market proxy, not an investable instrument
- Survivorship bias: dataset includes only accounts that remained active
- No out-of-sample or walk-forward validation performed
- 480 trading days is a short backtest window; strategy has not been tested across a full market cycle

These limitations mean the backtest Sharpe of 2.96 should be treated as an upper-bound estimate. Live performance would likely be materially lower.

---

## 5. Advanced Insights (Ranked by Composite Score)

| Rank | Insight | Score | Key Evidence | Statistical Significance |
|---|---|---|---|---|
| 1 | Extreme Regime PnL Concentration | 4.5/5 | 33.5% of PnL in 26.7% of trading days | Descriptive; no hypothesis test required |
| 2 | Performance Persistence | 4.2/5 | Month-to-month PnL r=0.194 | **p=0.008** |
| 3 | Regime Transition PnL Difference | 4.0/5 | $5,668 vs $3,829 on change vs stable days | p=0.166 (not significant) |
| 4 | Sentiment-PnL Asymmetry | 3.8/5 | Fear mean $5,185 vs Greed $4,188 | p=0.477 (not significant); d=0.033 |
| 5 | Temporal Behavioral Shift | 3.8/5 | Early r=0.087 → Late r=−0.017 | Descriptive |
| 6 | Momentum vs Level Signal | 3.5/5 | Momentum r=0.022 vs Level r=0.001 | Neither significant for PnL |
| 7 | Volume-Sentiment Inverse | 3.2/5 | r=−0.074 | **p<0.001** |
| 8 | Position Size Dispersion | 3.2/5 | CV: Fear 4.10, Extreme Greed 1.52 | Descriptive |
| 9 | Margin Type Regime Shift | 3.0/5 | Cross-margin: 68.8% in Extreme Greed, 62.4% in Extreme Fear | Descriptive |

Of the 9 insights, only 2 (Performance Persistence and Volume-Sentiment Inverse) reach conventional statistical significance. The Extreme Regime PnL Concentration is a descriptive fact that does not require a hypothesis test. The remaining insights represent observed patterns that warrant further investigation with larger samples.

---

## 6. Limitations

1. **Leverage:** Not directly observable in the dataset. All position-risk analysis uses Size USD, exposure metrics, and margin type (Cross/Isolated) as observable proxies. True leverage, margin utilization, and account equity are unknown. This is a material limitation that prevents risk-adjusted comparisons across traders.

2. **Holding Duration:** Not reconstructed. The complexity of matching 211K fills across position opens, closes, flips (Long > Short), partial closes, and auto-deleveraging events exceeds the cost-benefit threshold for this analysis. A FIFO-based position tracker would require significant engineering effort with uncertain match quality.

3. **Temporal Skew:** 87% of trade data is concentrated in the last 6 months (November 2024 – May 2025). Early-period statistics have wider confidence intervals. Any apparent temporal shift in behavior (e.g., Insight #5) may partially reflect this compositional change.

4. **Sample Size:** 32 accounts provide adequate statistical power for daily-level analysis (2,338 account-days) but severely limit cross-sectional trader comparisons. Cohort tests with 4 traders per tail have very low power (post-hoc power <0.3 for medium effects), and their results are descriptive rather than inferential.

5. **Survivorship Bias:** The dataset contains only accounts that remained active through the observation period. Traders who suffered catastrophic losses and ceased trading are absent, biasing aggregate PnL upward and potentially inflating estimated win rates and Sharpe ratios.

6. **Confounding:** The FGI co-moves with Bitcoin price, which directly determines trader PnL through position mark-to-market. Without controlling for concurrent returns (Bitcoin price data was not available in the provided datasets), observed sentiment-PnL associations may reflect the mechanical relationship between price direction and sentiment rather than a behavioral effect. All findings are associational.

7. **Backtest Limitations:** Strategy backtests use aggregate trader PnL as a market proxy, which is not an investable instrument. No transaction costs, slippage, or market impact are modeled. Results are in-sample only, with no out-of-sample or walk-forward validation.

---

## 7. Recommendations

### For a Crypto Trading Desk (conditional on further validation):

1. **Investigate contrarian sizing as a risk management overlay** — The regime-based sizing strategy showed favorable backtest characteristics (Sharpe 2.96), but requires out-of-sample validation, transaction cost modeling, and testing on investable instruments before deployment. The underlying signal (higher PnL in fear) does not reach parametric significance in this sample.

2. **Monitor trade frequency around sentiment extremes** — Trade count variation across regimes is statistically significant (ANOVA p=0.013, KW p<0.001). The 57-trade/day difference between Extreme Fear and Extreme Greed (d=0.329, CI: 17–98) represents a meaningful behavioral signal for execution and risk monitoring.

3. **Assess performance persistence for capital allocation** — Month-to-month PnL autocorrelation (r=0.194, p=0.008) is consistent with skill persistence, though it could also reflect persistent market regime exposure. Distinguishing skill from regime exposure requires controlling for market returns.

4. **Track top-trader positioning as a contrarian indicator** — The significant negative correlation between top-trader long bias and sentiment (r=−0.191, p<0.001) suggests profitable traders reduce exposure during greed. This pattern, if confirmed on a larger sample, could inform sentiment-based position monitoring.

5. **Treat extreme regimes as high-information environments** — 33.5% of realized PnL concentrates in extreme sentiment periods (26.7% of days), suggesting disproportionate opportunity concentration. Whether this reflects genuine alpha or mechanical price-momentum effects requires further investigation with price data.

### For Future Research:

1. Add Bitcoin price returns as a control variable to disentangle sentiment effects from price momentum
2. Expand the trader sample beyond 32 accounts to improve cross-sectional power
3. Incorporate realistic transaction costs (Hyperliquid fee schedule) into strategy backtests
4. Perform out-of-sample or walk-forward validation of the Regime Contrarian Sizing strategy
5. Investigate within-day sentiment timing using intraday FGI snapshots
6. Test nonlinear models (gradient boosted trees, spline regression) for PnL prediction, given the Kruskal-Wallis significance suggests distributional rather than mean effects

---

*Report generated by automated quantitative research pipeline. Parametric tests (ANOVA, Welch's t-test) accompanied by non-parametric alternatives (Kruskal-Wallis, Mann-Whitney U) throughout. Multiple comparison corrections applied: Bonferroni for pairwise test families, Benjamini-Hochberg FDR for correlation matrices. Robustness verified across three data specifications (full sample, winsorized at 1%/99%, IQR-filtered). All confidence intervals are at the 95% level.*
