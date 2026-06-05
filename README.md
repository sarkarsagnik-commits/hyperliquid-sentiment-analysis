# Hyperliquid Sentiment × Trader Performance

Quantitative research investigating the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance on the Hyperliquid DEX.

## Research Question

> How does Bitcoin market sentiment influence trader behavior and performance on Hyperliquid?

## Key Findings

| Finding | Evidence | Actionability |
|---|---|---|
| Traders earn 24% more in fear markets | Fear mean PnL $5,185 vs Greed $4,188 | ★★★★★ |
| Contrarian sizing generates Sharpe 2.96 | Regime-based position sizing strategy | ★★★★★ |
| Performance persists month-to-month | Autocorrelation r=0.194, p=0.008 | ★★★★☆ |
| 33.5% of PnL in extreme regimes | Disproportionate to time spent in regime | ★★★★☆ |
| Volume inversely correlated with sentiment | r=-0.074, p<0.001 | ★★★☆☆ |

## Project Structure

```
├── data/
│   ├── raw/                         # Original datasets
│   │   ├── fear_greed_index.csv     # Bitcoin FGI (2018-2025)
│   │   ├── historical_data.csv      # Hyperliquid trader data
│   │   └── DS Task.pdf              # Assignment specification
│   └── processed/                   # Cleaned & merged outputs
├── notebooks/
│   └── sentiment_analysis.ipynb     # End-to-end Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── utils.py                     # Constants, palettes, helpers
│   ├── data_loader.py               # Data loading & validation
│   ├── feature_engineering.py       # Feature creation pipeline
│   ├── visualization.py             # Publication-quality charts
│   ├── statistical_analysis.py      # Hypothesis testing & insights
│   └── strategy.py                  # Backtesting framework
├── reports/
│   └── final_report.md              # Full research report
├── figures/                         # Generated visualizations (13 plots)
├── run_analysis.py                  # Master execution script
├── create_notebook.py               # Notebook generator
├── requirements.txt                 # Dependencies
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis pipeline
python run_analysis.py

# Or use the Jupyter notebook
jupyter notebook notebooks/sentiment_analysis.ipynb
```

## Datasets

| Dataset | Rows | Columns | Period |
|---|---|---|---|
| Fear & Greed Index | 2,648 | 4 | Feb 2018 – May 2025 |
| Historical Trader Data | 211,181 | 16 | May 2023 – May 2025 |
| Overlap Window | — | — | 732 days |

## Methodology

- **Trade Identification:** `Order ID` as logical trade identifier; rows are fills
- **Leverage:** Not directly observable — analysis uses position size, exposure, and margin type
- **Statistical Rigor:** Bonferroni + Benjamini-Hochberg FDR corrections on all multi-test results
- **Robustness:** Every major finding verified across full sample, winsorized, and IQR-filtered data
- **Effect Sizes:** Cohen's d reported for all pairwise comparisons

## Strategies Backtested

| Strategy | Sharpe | Total Return | Max Drawdown |
|---|---|---|---|
| Sentiment Reversal (Contrarian) | -1.23 | -$2.52M | -$5.14M |
| Sentiment Momentum | -0.73 | -$1.20M | -$2.65M |
| **Regime Contrarian Sizing** | **2.96** | **$3.87M** | **-$320K** |

## Visualizations Generated

13 publication-quality charts saved to `figures/`:
- Sentiment distribution and time series
- PnL distribution and regime comparison
- Position size analysis by regime
- Correlation heatmaps (Pearson, Spearman)
- Trader cohort comparisons
- Regime transition probability matrix
- Strategy equity curves
- Sentiment lag correlation analysis

## Technical Details

- **Python 3.13+**
- **Key Libraries:** pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, plotly
- **32 unique trader accounts**, 246 coins, 480 trading days
- **Total realized PnL across all traders:** $10,296,959

## Author

Sagnik Sarkar — June 2025

---

*Built for the Primetrade.ai Data Science hiring assessment.*
