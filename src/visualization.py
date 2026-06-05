"""
Publication-quality visualization module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from src.utils import (
    REGIME_COLORS, REGIME_ORDER, COHORT_COLORS,
    PALETTE_MAIN, PALETTE_NEGATIVE, PALETTE_ACCENT,
    save_figure, configure_plotting,
)


def plot_sentiment_distribution(fgi):
    """Histogram of FGI value colored by regime."""
    fig, ax = plt.subplots(figsize=(12, 6))
    boundaries = [0, 25, 45, 55, 75, 100]
    for i, regime in enumerate(REGIME_ORDER):
        lo, hi = boundaries[i], boundaries[i + 1]
        subset = fgi[(fgi["value"] >= lo) & (fgi["value"] < (hi + (1 if i == 4 else 0)))]
        ax.hist(subset["value"], bins=range(lo, hi + 1), color=REGIME_COLORS[regime],
                alpha=0.85, label=regime, edgecolor="white", linewidth=0.5)
    for b in [25, 45, 55, 75]:
        ax.axvline(b, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Fear & Greed Index Value")
    ax.set_ylabel("Frequency (Days)")
    ax.set_title("Bitcoin Fear & Greed Index Distribution (2018–2025)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    save_figure(fig, "sentiment_distribution")
    return fig


def plot_pnl_distribution(daily_metrics):
    """Histogram of non-zero daily PnL."""
    pnl = daily_metrics.loc[daily_metrics["daily_pnl"] != 0, "daily_pnl"]
    fig, ax = plt.subplots(figsize=(12, 6))
    pos = pnl[pnl > 0]
    neg = pnl[pnl < 0]
    bins = np.linspace(pnl.quantile(0.01), pnl.quantile(0.99), 80)
    ax.hist(pos, bins=bins, color=PALETTE_ACCENT, alpha=0.8, label="Profit Days")
    ax.hist(neg, bins=bins, color=PALETTE_NEGATIVE, alpha=0.8, label="Loss Days")
    for q, ls, lbl in [(0.05, "--", "P5"), (0.50, "-", "Median"), (0.95, "--", "P95")]:
        v = pnl.quantile(q)
        ax.axvline(v, color="black", linestyle=ls, linewidth=1, alpha=0.7)
        ax.text(v, ax.get_ylim()[1] * 0.9, f" {lbl}: ${v:,.0f}", fontsize=8, rotation=90, va="top")
    ax.set_xlabel("Daily PnL ($)")
    ax.set_ylabel("Frequency")
    ax.set_title("Daily PnL Distribution Across All Traders", fontsize=14, fontweight="bold")
    ax.legend()
    save_figure(fig, "pnl_distribution")
    return fig


def plot_position_size_distribution(merged):
    """Box plots of Size USD by sentiment regime."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data = merged[merged["Size USD"] > 0].copy()
    palette = [REGIME_COLORS[r] for r in REGIME_ORDER]
    sns.boxplot(data=data, x="sentiment_regime", y="Size USD", order=REGIME_ORDER,
                palette=palette, ax=ax, showfliers=False, width=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("Position Size (USD, log scale)")
    ax.set_title("Position Size Distribution by Sentiment Regime", fontsize=14, fontweight="bold")
    save_figure(fig, "position_size_distribution")
    return fig


def plot_trade_activity_over_time(daily_metrics):
    """Daily trade count over time with smoothing."""
    agg = daily_metrics.groupby("trade_date")["daily_trades"].sum().reset_index()
    agg["rolling_7d"] = agg["daily_trades"].rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(agg["trade_date"], agg["daily_trades"], alpha=0.2, color=PALETTE_MAIN)
    ax.plot(agg["trade_date"], agg["rolling_7d"], color=PALETTE_MAIN, linewidth=1.5, label="7-day MA")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Trades")
    ax.set_title("Trading Activity Over Time", fontsize=14, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    save_figure(fig, "trade_activity_time")
    return fig


def plot_sentiment_over_time(fgi):
    """Sentiment value time series with regime shading."""
    fig, ax = plt.subplots(figsize=(16, 5))
    dates = pd.to_datetime(fgi["date"])
    vals = fgi["value"].values
    # Shade background by regime
    boundaries = [(0, 25), (25, 45), (45, 55), (55, 75), (75, 100)]
    for i, regime in enumerate(REGIME_ORDER):
        lo, hi = boundaries[i]
        ax.axhspan(lo, hi, alpha=0.10, color=REGIME_COLORS[regime])
    ax.plot(dates, vals, color="#333333", linewidth=0.6, alpha=0.8)
    ax.fill_between(dates, vals, alpha=0.3, color=PALETTE_MAIN)
    for b in [25, 45, 55, 75]:
        ax.axhline(b, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.set_xlim(dates.min(), dates.max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Date")
    ax.set_ylabel("Fear & Greed Index")
    ax.set_title("Bitcoin Fear & Greed Index Over Time", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_figure(fig, "sentiment_over_time")
    return fig


def plot_daily_profitability(daily_metrics):
    """Dual-axis: aggregate daily PnL bars + sentiment line."""
    agg = daily_metrics.groupby("trade_date").agg(
        daily_pnl=("daily_pnl", "sum"),
        sentiment=("sentiment_value", "first")
    ).reset_index()
    fig, ax1 = plt.subplots(figsize=(14, 6))
    colors = [PALETTE_ACCENT if x >= 0 else PALETTE_NEGATIVE for x in agg["daily_pnl"]]
    ax1.bar(agg["trade_date"], agg["daily_pnl"], color=colors, alpha=0.7, width=1.0)
    ax1.set_ylabel("Aggregate Daily PnL ($)", color=PALETTE_MAIN)
    ax1.set_xlabel("Date")
    ax2 = ax1.twinx()
    ax2.plot(agg["trade_date"], agg["sentiment"], color="#ef6c00", linewidth=1.0, alpha=0.7)
    ax2.set_ylabel("Sentiment Value", color="#ef6c00")
    ax2.set_ylim(0, 100)
    ax1.set_title("Daily Aggregate Profitability vs Market Sentiment", fontsize=14, fontweight="bold")
    fig.autofmt_xdate()
    save_figure(fig, "daily_profitability")
    return fig


def plot_correlation_heatmap(corr_matrix, title, filename):
    """Annotated correlation heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, mask=mask, ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    save_figure(fig, filename)
    return fig


def plot_regime_pnl_comparison(daily_metrics):
    """Violin + strip of daily PnL by regime."""
    fig, ax = plt.subplots(figsize=(12, 6))
    data = daily_metrics[daily_metrics["daily_pnl"] != 0].copy()
    # Clip for visibility
    clip_lo, clip_hi = data["daily_pnl"].quantile(0.02), data["daily_pnl"].quantile(0.98)
    data["daily_pnl_clipped"] = data["daily_pnl"].clip(clip_lo, clip_hi)
    palette = [REGIME_COLORS[r] for r in REGIME_ORDER]
    sns.violinplot(data=data, x="sentiment_regime", y="daily_pnl_clipped",
                   order=REGIME_ORDER, palette=palette, ax=ax, inner="quartile", cut=0)
    # Annotate medians
    for i, regime in enumerate(REGIME_ORDER):
        subset = data[data["sentiment_regime"] == regime]["daily_pnl"]
        med = subset.median()
        ax.text(i, clip_hi * 0.95, f"med=${med:,.0f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("Daily PnL ($, clipped at P2/P98)")
    ax.set_title("Daily PnL by Sentiment Regime", fontsize=14, fontweight="bold")
    save_figure(fig, "regime_pnl_comparison")
    return fig


def plot_trader_segments(trader_summary):
    """Radar-style grouped bar chart comparing cohorts."""
    if "cohort" not in trader_summary.columns:
        return None
    metrics = ["overall_win_rate", "long_bias", "avg_daily_trades", "avg_position_size"]
    labels = ["Win Rate", "Long Bias", "Avg Daily Trades\n(normalized)", "Avg Position Size\n(normalized)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)
    for idx, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[idx]
        for cohort in ["Top 10%", "Middle 80%", "Bottom 10%"]:
            subset = trader_summary[trader_summary["cohort"] == cohort]
            if len(subset) > 0:
                val = subset[metric].mean()
                ax.bar(cohort, val, color=COHORT_COLORS.get(cohort, "gray"), alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Trader Cohort Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_figure(fig, "trader_segments")
    return fig


def plot_regime_transition_matrix(fgi):
    """Heatmap of day-to-day regime transition probabilities."""
    trans = pd.DataFrame({
        "from": fgi["classification"].iloc[:-1].values,
        "to": fgi["classification"].iloc[1:].values,
    })
    matrix = pd.crosstab(trans["from"], trans["to"], normalize="index")
    matrix = matrix.reindex(index=REGIME_ORDER, columns=REGIME_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Next Day Regime")
    ax.set_ylabel("Current Regime")
    ax.set_title("Sentiment Regime Transition Probabilities", fontsize=14, fontweight="bold")
    save_figure(fig, "regime_transition_matrix")
    return fig


def plot_pnl_by_regime_and_cohort(daily_metrics, trader_summary):
    """Grouped box plots: PnL by regime and cohort."""
    if "cohort" not in trader_summary.columns:
        return None
    dm = daily_metrics.merge(
        trader_summary[["cohort"]].reset_index(), on="Account", how="left"
    )
    dm = dm[dm["daily_pnl"] != 0]
    clip_lo, clip_hi = dm["daily_pnl"].quantile(0.05), dm["daily_pnl"].quantile(0.95)
    dm["pnl_clipped"] = dm["daily_pnl"].clip(clip_lo, clip_hi)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=dm, x="sentiment_regime", y="pnl_clipped", hue="cohort",
                order=REGIME_ORDER, hue_order=["Top 10%", "Middle 80%", "Bottom 10%"],
                palette=COHORT_COLORS, ax=ax, showfliers=False, width=0.7)
    ax.set_xlabel("Sentiment Regime")
    ax.set_ylabel("Daily PnL ($, clipped)")
    ax.set_title("PnL by Sentiment Regime and Trader Cohort", fontsize=14, fontweight="bold")
    ax.legend(title="Cohort")
    save_figure(fig, "pnl_regime_cohort")
    return fig


def plot_sentiment_lag_correlation(daily_metrics):
    """Bar chart of lagged sentiment correlation with PnL."""
    results = []
    for lag in [1, 3, 7]:
        col = f"sentiment_lag_{lag}"
        if col in daily_metrics.columns:
            valid = daily_metrics[[col, "daily_pnl"]].dropna()
            if len(valid) > 10:
                from scipy import stats as sp_stats
                r, p = sp_stats.pearsonr(valid[col], valid["daily_pnl"])
                results.append({"lag": f"{lag}-day", "correlation": r, "p_value": p})
    if not results:
        return None
    res_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE_MAIN if p < 0.05 else "#cccccc" for p in res_df["p_value"]]
    bars = ax.bar(res_df["lag"], res_df["correlation"], color=colors, edgecolor="white", width=0.5)
    for bar, row in zip(bars, results):
        sig = "***" if row["p_value"] < 0.001 else ("**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else ""))
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'r={row["correlation"]:.3f}{sig}', ha="center", va="bottom", fontsize=10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Sentiment Lag")
    ax.set_ylabel("Pearson Correlation with Daily PnL")
    ax.set_title("Lagged Sentiment Correlation with Future PnL", fontsize=14, fontweight="bold")
    save_figure(fig, "sentiment_lag_correlation")
    return fig


def generate_all_plots(fgi, daily_metrics, trader_summary, merged, corr_results=None):
    """Generate all publication-quality plots."""
    configure_plotting()
    saved = []
    plot_funcs = [
        ("Sentiment Distribution", lambda: plot_sentiment_distribution(fgi)),
        ("PnL Distribution", lambda: plot_pnl_distribution(daily_metrics)),
        ("Position Size Distribution", lambda: plot_position_size_distribution(merged)),
        ("Trade Activity", lambda: plot_trade_activity_over_time(daily_metrics)),
        ("Sentiment Over Time", lambda: plot_sentiment_over_time(fgi)),
        ("Daily Profitability", lambda: plot_daily_profitability(daily_metrics)),
        ("Regime PnL Comparison", lambda: plot_regime_pnl_comparison(daily_metrics)),
        ("Trader Segments", lambda: plot_trader_segments(trader_summary)),
        ("Regime Transition Matrix", lambda: plot_regime_transition_matrix(fgi)),
        ("PnL by Regime & Cohort", lambda: plot_pnl_by_regime_and_cohort(daily_metrics, trader_summary)),
        ("Sentiment Lag Correlation", lambda: plot_sentiment_lag_correlation(daily_metrics)),
    ]
    if corr_results and "pearson" in corr_results:
        plot_funcs.append(("Pearson Heatmap", lambda: plot_correlation_heatmap(
            corr_results["pearson"], "Pearson Correlation Matrix", "correlation_pearson")))
        plot_funcs.append(("Spearman Heatmap", lambda: plot_correlation_heatmap(
            corr_results["spearman"], "Spearman Correlation Matrix", "correlation_spearman")))

    for name, func in plot_funcs:
        try:
            func()
            saved.append(name)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return saved
