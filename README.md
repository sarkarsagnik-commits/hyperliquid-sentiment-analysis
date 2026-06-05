# Hyperliquid Sentiment × Trader Performance Analysis

A quantitative research project investigating the statistical relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance on the Hyperliquid decentralized exchange. The analysis covers 211,181 trade fills across 32 accounts over 480 trading days, applying institutional-grade statistical methods with multiple comparison corrections and robustness checks.

---

## Executive Summary

**Research objective:** Determine whether Bitcoin market sentiment, as measured by the daily Fear & Greed Index (FGI), is statistically associated with trader profitability, behavior, and risk-taking on the Hyperliquid DEX.

**Datasets:** Bitcoin Fear & Greed Index (2,648 daily observations, Feb 2018 – May 2025) merged with Hyperliquid historical trade data (211,181 fills, 32 accounts, 246 trading pairs, May 2023 – May 2025). The overlap window spans 480 trading days.

**Major findings:**
- PnL distributions differ significantly across sentiment regimes (Kruskal-Wallis p=0.002), though mean differences do not reach parametric significance (ANOVA p=0.671). The Fear/Extreme Fear group shows a mean daily PnL of $5,185 vs. $4,188 for Greed/Extreme Greed, but the 95% CI for the difference crosses zero (−$1,464, +$5,365).
- A regime-based contrarian sizing strategy produced a backtest Sharpe ratio of 2.96 (in-sample, no transaction costs).
- Month-to-month PnL autocorrelation is positive and significant (r=0.194, p=0.008).
- 33.5% of total realized PnL ($10,296,959) concentrates in extreme sentiment regimes, which account for 26.7% of trading days.
- Trade frequency is significantly higher in fear markets (ANOVA p=0.013, Kruskal-Wallis p<0.001).

**Business implications:** The findings suggest that sentiment regime is associated with measurable differences in trading activity and PnL distribution shape, though not with statistically distinguishable mean PnL. A contrarian position-sizing framework merits further investigation with out-of-sample validation and transaction cost modeling.

---

## Key Results

| Finding | Statistic | Significance |
|---|---|---|
| Fear-regime PnL vs. Greed-regime PnL | $5,185 vs. $4,188 mean daily PnL | KW p=0.002; ANOVA p=0.671; d=0.033 |
| Regime Contrarian Sizing strategy | Sharpe 2.96, Sortino 5.46, +$3.87M return | In-sample backtest only |
| Performance persistence | Month-to-month PnL autocorrelation r=0.194 | p=0.008 |
| Extreme regime PnL concentration | 33.5% of PnL in 26.7% of trading days | Descriptive |
| Volume-sentiment relationship | Pearson r=−0.074 | p<0.001 |
| Trade count: Extreme Fear vs. Extreme Greed | +57.3 trades/day difference; d=0.329 | p=0.006 |
| Top-trader contrarian behavior | Sentiment–long_ratio r=−0.191 | p<0.001 |

All pairwise tests include Bonferroni correction. Correlation matrices use Benjamini-Hochberg FDR correction. Effect sizes reported as Cohen's d.

---

## Project Highlights

- **211,181 trade fills** cleaned, validated, and analyzed across **32 trader accounts** and **246 trading pairs**
- **480 trading days** of merged sentiment–performance data over a 2-year window
- **Feature engineering pipeline** producing sentiment features (momentum, z-score, regime persistence), daily trader metrics, rolling performance, and lagged signals
- **Statistical hypothesis testing** with parametric and non-parametric tests, multiple comparison corrections, effect sizes, and confidence intervals
- **Robustness analysis** across three data specifications: full sample, winsorized (1%/99%), and IQR-filtered
- **Trader segmentation** into performance cohorts (Top 10%, Middle 80%, Bottom 10%) with behavioral analysis
- **Three strategy backtests** with comprehensive performance metrics (Sharpe, Sortino, Calmar, max drawdown, profit factor)
- **14 publication-quality visualizations** generated programmatically
- **Reproducible end-to-end workflow** — single command executes the full pipeline from raw data to final report

---

## Repository Structure

```
├── data/
│   ├── raw/                           # Original unmodified datasets
│   │   ├── fear_greed_index.csv       # Bitcoin FGI (2,648 daily observations)
│   │   ├── historical_data.csv        # Hyperliquid trade fills (211,181 rows)
│   │   └── DS Task.pdf                # Assignment specification
│   └── processed/                     # Cleaned and merged pipeline outputs
│       ├── fgi_clean.csv              # Gap-filled FGI with regime labels
│       └── merged_data.csv            # Sentiment-enriched trade data
│
├── figures/                           # 14 generated PNG visualizations
│
├── notebooks/
│   └── sentiment_analysis.ipynb       # End-to-end Jupyter notebook
│
├── reports/
│   └── final_report.md                # Institutional research report
│
├── src/                               # Modular Python source
│   ├── __init__.py
│   ├── utils.py                       # Constants, color palettes, helper functions
│   ├── data_loader.py                 # Data loading, cleaning, validation, merging
│   ├── feature_engineering.py         # Sentiment and trader feature construction
│   ├── visualization.py               # Chart generation (13 plot functions)
│   ├── statistical_analysis.py        # Hypothesis testing, behavioral analysis, insights
│   └── strategy.py                    # Strategy backtesting framework
│
├── run_analysis.py                    # Master execution script
├── create_notebook.py                 # Programmatic notebook generator
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## Methodology

### Data Cleaning

- **FGI:** 4 missing days identified and forward-filled; regime classifications validated against score boundaries
- **Trader data:** 43 dust trades (Size USD = 0) removed; `Order ID` used as the logical trade identifier (individual rows are fill-level executions; `Trade ID` is non-unique with 208K duplicates)
- **Merge:** Inner join on date; 15 automated validation checks (all passed)

### Feature Engineering

**Sentiment features:** Rolling means and volatility (7/14/30-day windows), sentiment momentum (day-over-day Δ), z-score (deviation from 30-day mean normalized by rolling standard deviation), regime transition indicators, consecutive-days-in-regime counters.

**Trader features (per account, per day):** Daily PnL, win rate, gross profit/loss, profit factor, position size statistics (mean, max, total volume), long/short bias, cross-margin ratio, fee totals.

**Rolling metrics:** 30-day rolling Sharpe ratio and win rate per account. Lagged sentiment (1, 3, 7 days) for predictive analysis.

### Sentiment Regime Construction

The FGI score (0–100) maps to five regimes:

| Regime | FGI Range | Interpretation |
|---|---|---|
| Extreme Fear | 0–24 | Market panic, capitulation |
| Fear | 25–44 | Below-average sentiment |
| Neutral | 45–54 | Indeterminate |
| Greed | 55–74 | Above-average sentiment |
| Extreme Greed | 75–100 | Euphoria, excess optimism |

### Statistical Testing Framework

All hypothesis tests report statistical significance, effect size, and economic magnitude.

**Omnibus tests:**
- One-way ANOVA (parametric, assumes normality)
- Kruskal-Wallis H-test (non-parametric, robust to heavy tails)

**Pairwise tests:**
- Welch's t-test (unequal variance)
- Mann-Whitney U test (non-parametric)

**Correlation analysis:**
- Pearson (linear), Spearman (monotonic), Kendall (rank concordance)

**Multiple comparison corrections:**
- Bonferroni correction for pairwise test families
- Benjamini-Hochberg FDR correction for correlation matrices

**Effect size:** Cohen's d for all pairwise comparisons, with qualitative interpretation (|d|<0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, >0.8 large).

### Trader Segmentation

Traders segmented into three cohorts by cumulative PnL:
- **Top 10%** (4 traders): PnL ≥ 90th percentile ($840K–$2.14M)
- **Middle 80%** (24 traders): $14.9K–$836K
- **Bottom 10%** (4 traders): PnL ≤ 10th percentile (−$168K–$14.5K)

Cross-cohort comparisons use Kruskal-Wallis and pairwise Mann-Whitney U tests with Cohen's d.

### Robustness Checks

Every major finding is tested across three data specifications:
1. **Full sample** (n=2,338 account-days)
2. **Winsorized** at 1st and 99th percentiles
3. **IQR-filtered** (excluding observations beyond ±1.5×IQR)

---

## Research Questions

1. Does Bitcoin market sentiment correlate with trader profitability on Hyperliquid?
2. Do traders adjust position sizing, trade frequency, or directional bias in response to sentiment regime changes?
3. Can lagged sentiment predict future PnL, position size, or trade activity?
4. Do top-performing traders exhibit different sentiment-response patterns (contrarian vs. trend-following) compared to underperformers?
5. Can systematic sentiment-based trading strategies produce favorable risk-adjusted returns in backtest?
6. Are observed relationships robust across data specifications (full sample, winsorized, IQR-filtered)?

---

## Visualizations

The pipeline generates 14 publication-quality figures covering sentiment dynamics, PnL distributions, regime effects, trader behavior, correlation structure, and strategy performance:

| Figure | Description |
|---|---|
| `sentiment_distribution.png` | FGI histogram colored by regime with boundary markers |
| `sentiment_over_time.png` | FGI time series (2018–2025) with regime-shaded background |
| `pnl_distribution.png` | Daily PnL histogram with P5/median/P95 annotations |
| `position_size_distribution.png` | Position size box plots by regime (log scale) |
| `regime_pnl_comparison.png` | Violin plots of daily PnL by regime with median annotations |
| `daily_profitability.png` | Dual-axis: aggregate daily PnL bars + sentiment line overlay |
| `trade_activity_time.png` | Trade count time series with 7-day moving average |
| `correlation_pearson.png` | Pearson correlation heatmap (lower triangle, annotated) |
| `correlation_spearman.png` | Spearman correlation heatmap |
| `regime_transition_matrix.png` | Day-to-day regime transition probability heatmap |
| `trader_segments.png` | Cohort comparison across key performance metrics |
| `pnl_regime_cohort.png` | PnL box plots grouped by regime and trader cohort |
| `sentiment_lag_correlation.png` | Lagged sentiment–PnL correlation with significance markers |
| `strategy_equity_curves.png` | Cumulative PnL for all strategies vs. benchmark |

---

## Strategy Research

Three sentiment-based systematic strategies were backtested against aggregate trader PnL over 480 trading days:

| Strategy | Sharpe | Sortino | Total Return | Max Drawdown | Win Rate |
|---|---|---|---|---|---|
| Sentiment Reversal (Contrarian) | −1.23 | −1.06 | −$2.52M | −$5.14M | 19.9% |
| Sentiment Momentum | −0.73 | −0.39 | −$1.20M | −$2.65M | 53.5% |
| **Regime Contrarian Sizing** | **2.96** | **5.46** | **+$3.87M** | **−$320K** | **65.5%** |

The **Regime Contrarian Sizing** strategy sizes positions inversely to sentiment (100% in Extreme Fear, −25% in Extreme Greed). It outperformed the other two strategies across all metrics.

**Critical limitations:** Results are in-sample only. No transaction costs, slippage, or market impact are modeled. The backtest uses aggregate trader PnL as a market proxy, which is not an investable instrument. Out-of-sample validation has not been performed. The backtest Sharpe of 2.96 should be treated as an upper-bound estimate.

---

## Key Insights

Ranked by composite score (practical value, statistical confidence, economic impact, novelty):

| Rank | Insight | Evidence | Significance |
|---|---|---|---|
| 1 | 33.5% of PnL concentrates in extreme regimes | Extreme regimes = 26.7% of days but 33.5% of realized PnL | Descriptive |
| 2 | Performance persistence across months | PnL autocorrelation r=0.194 | p=0.008 |
| 3 | Top traders are contrarian | Sentiment–long_ratio r=−0.191 for Top 10% | p<0.001 |
| 4 | Trade frequency varies by regime | Extreme Fear 133.8/day vs. Extreme Greed 76.4/day; d=0.329 | p=0.006 |
| 5 | Volume inversely associated with sentiment | Pearson r=−0.074 | p<0.001 |

Insights ranked #3–5 reach conventional statistical significance. The remaining insights from the full report (9 total) include descriptive patterns and non-significant trends documented transparently.

---

## Limitations

This analysis has several material limitations that constrain the strength of its conclusions:

1. **No observable leverage.** Leverage, margin utilization, and account equity are absent from the dataset. All position-risk analysis uses `Size USD` and margin type as proxies. True risk-adjusted comparisons across traders are not possible.

2. **Holding duration not reconstructed.** Matching 211K fills to position lifecycles (opens, partial closes, flips) requires a FIFO-based position tracker with uncertain match quality. This was assessed as exceeding the cost-benefit threshold.

3. **Survivorship bias.** The dataset includes only accounts active through the observation period. Traders who suffered catastrophic losses and ceased trading are absent, biasing aggregate PnL and win rates upward.

4. **Temporal concentration.** 87% of trade data falls within the final 6 months (November 2024 – May 2025). Early-period statistics carry wider confidence intervals.

5. **Small cross-sectional sample.** 32 accounts are sufficient for daily-level analysis (2,338 account-days) but limit cross-sectional inference. Cohort comparisons (4 traders per tail) are descriptive, not inferential.

6. **Correlation, not causation.** The FGI co-moves with Bitcoin price, which directly determines PnL. Without controlling for concurrent returns, observed sentiment–PnL associations may reflect the mechanical price–sentiment relationship rather than a behavioral effect.

7. **Backtest constraints.** Strategy results exclude transaction costs, slippage, and market impact. The aggregate trader PnL proxy is not investable. No out-of-sample validation has been performed.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/sarkarsagnik-commits/hyperliquid-sentiment-analysis.git
cd hyperliquid-sentiment-analysis

# Install dependencies (Python 3.10+)
pip install -r requirements.txt
```

**Required packages:** pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, plotly, nbformat.

---

## Running the Project

### Full Pipeline

```bash
python run_analysis.py
```

This executes the complete analysis in sequence:
1. Data loading, cleaning, and validation (15 automated checks)
2. Feature engineering (sentiment, trader, rolling, lagged features)
3. Statistical analysis (correlations, hypothesis tests, behavioral analysis, robustness checks)
4. Visualization (14 figures saved to `figures/`)
5. Strategy backtesting (3 strategies with performance metrics)
6. Advanced insights generation

Expected runtime: approximately 60 seconds.

### Jupyter Notebook

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

The notebook provides an interactive, narrated walkthrough of the same pipeline with inline outputs and figures.

---

## Reproducibility

The project is reproducible end-to-end from the raw data files in `data/raw/`.

| Input | Output |
|---|---|
| `data/raw/fear_greed_index.csv` | `data/processed/fgi_clean.csv` |
| `data/raw/historical_data.csv` | `data/processed/merged_data.csv` |
| — | `figures/*.png` (14 visualizations) |
| — | `notebooks/sentiment_analysis.ipynb` |
| — | `reports/final_report.md` |

All randomness is deterministic (no stochastic processes). Running `python run_analysis.py` on the same inputs will produce identical outputs. The processed data files, figures, and report included in the repository were generated by this pipeline.

---

## Technical Skills Demonstrated

| Category | Skills |
|---|---|
| **Quantitative Research** | Hypothesis testing, effect size estimation, multiple comparison corrections, robustness analysis |
| **Statistical Methods** | ANOVA, Kruskal-Wallis, Welch's t-test, Mann-Whitney U, Pearson/Spearman/Kendall correlation, OLS regression, Bonferroni correction, Benjamini-Hochberg FDR |
| **Feature Engineering** | Rolling statistics, momentum signals, z-scores, regime encoding, lagged features, trader segmentation |
| **Data Analysis** | Exploratory data analysis, data validation pipelines, outlier detection, distributional analysis |
| **Algorithmic Trading Research** | Strategy design, backtesting framework, performance metrics (Sharpe, Sortino, Calmar, profit factor, max drawdown) |
| **Visualization** | Publication-quality charts (matplotlib, seaborn), correlation heatmaps, violin plots, dual-axis time series |
| **Python** | pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, plotly, nbformat |
| **Engineering** | Modular codebase, automated validation, reproducible pipelines, version control |

---

## Relevance to Quantitative Roles

**Quantitative Research:** This project demonstrates the ability to formulate testable hypotheses, apply appropriate statistical methods (including non-parametric alternatives for non-normal data), interpret results with proper epistemic caution, and distinguish between statistical significance, effect size, and economic significance.

**Data Science:** The pipeline covers the full data science workflow: data cleaning and validation, feature engineering, exploratory analysis, hypothesis testing, visualization, and communication of results. The modular codebase and reproducible pipeline reflect production-ready engineering practices.

**Trading Research:** The strategy research section demonstrates familiarity with backtest methodology, performance attribution, and the critical importance of acknowledging backtest limitations (in-sample bias, missing transaction costs, non-investable proxies).

**Crypto Analytics:** Domain-specific experience with DEX trade data, perpetual futures mechanics (cross vs. isolated margin, position sizing in USD), and sentiment-driven market microstructure.

---

## Future Work

1. **Add Bitcoin price returns as a control variable** to disentangle sentiment effects from concurrent price momentum
2. **Expand the trader sample** beyond 32 accounts to improve cross-sectional statistical power
3. **Incorporate Hyperliquid fee schedule** into strategy backtests for realistic transaction cost modeling
4. **Perform out-of-sample validation** of the Regime Contrarian Sizing strategy using walk-forward or expanding-window methodology
5. **Investigate nonlinear models** (gradient boosted trees, spline regression) for PnL prediction, given that Kruskal-Wallis significance suggests distributional rather than mean effects
6. **Reconstruct holding durations** using a FIFO-based position tracker to enable time-in-trade analysis
7. **Add intraday sentiment data** to capture within-day behavioral dynamics

---

## Author

**Sagnik Sarkar**  
June 2025

---

*Built as a data science hiring assessment for Primetrade.ai. All statistical tests include multiple comparison corrections. Results verified across three robustness specifications. Confidence intervals reported at the 95% level.*
