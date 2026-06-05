"""
Feature Engineering — Sentiment, Trader, and Regime Features.
"""

import pandas as pd
import numpy as np
from src.utils import REGIME_ORDER, REGIME_SCORE_MAP


def create_sentiment_features(fgi: pd.DataFrame) -> pd.DataFrame:
    """Add derived sentiment features to FGI data."""
    df = fgi.copy()
    df["sentiment_score"] = df["classification"].map(REGIME_SCORE_MAP)
    df["sentiment_momentum"] = df["value"].diff()
    for w in [7, 14, 30]:
        df[f"sentiment_ma_{w}"] = df["value"].rolling(w, min_periods=1).mean()
        df[f"sentiment_vol_{w}"] = df["value"].rolling(w, min_periods=2).std()
    ma30 = df["sentiment_ma_30"]
    vol30 = df["sentiment_vol_30"]
    df["sentiment_zscore"] = np.where(vol30 > 0, (df["value"] - ma30) / vol30, 0)
    return df


def create_regime_features(fgi: pd.DataFrame) -> pd.DataFrame:
    """Add regime transition and persistence features."""
    df = fgi.copy()
    regime_map = {r: i for i, r in enumerate(REGIME_ORDER)}
    df["regime_numeric"] = df["classification"].map(regime_map)
    df["regime_changed"] = df["classification"] != df["classification"].shift(1)
    # Consecutive days in current regime
    groups = (df["classification"] != df["classification"].shift(1)).cumsum()
    df["consecutive_regime_days"] = df.groupby(groups).cumcount() + 1
    return df


def create_trader_daily_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fill-level data to account-day level metrics."""
    def _agg(g):
        closing = g[g["is_closing"]]
        pnl_nonzero = g.loc[g["Closed PnL"] != 0, "Closed PnL"]
        n = len(g)
        n_closing = len(closing)
        n_win = (g["Closed PnL"] > 0).sum()
        n_lose = (g["Closed PnL"] < 0).sum()
        gross_profit = g.loc[g["Closed PnL"] > 0, "Closed PnL"].sum()
        gross_loss = abs(g.loc[g["Closed PnL"] < 0, "Closed PnL"].sum())
        return pd.Series({
            "daily_pnl": pnl_nonzero.sum() if len(pnl_nonzero) > 0 else 0.0,
            "daily_trades": n,
            "daily_closing_trades": n_closing,
            "daily_winning_trades": n_win,
            "daily_losing_trades": n_lose,
            "daily_gross_profit": gross_profit,
            "daily_gross_loss": gross_loss,
            "daily_win_rate": n_win / n_closing if n_closing > 0 else 0.0,
            "mean_position_size": g["Size USD"].mean(),
            "max_position_size": g["Size USD"].max(),
            "total_volume": g["Size USD"].sum(),
            "long_ratio": g["is_long"].sum() / n if n > 0 else 0.5,
            "cross_margin_ratio": (g["margin_type"] == "Cross").sum() / n if n > 0 else 0.0,
            "daily_fees": g["Fee"].sum(),
            "sentiment_value": g["sentiment_value"].iloc[0],
            "sentiment_regime": g["sentiment_regime"].iloc[0],
        })

    daily = merged.groupby(["Account", "trade_date"]).apply(_agg, include_groups=False).reset_index()
    daily["sentiment_regime"] = pd.Categorical(
        daily["sentiment_regime"], categories=REGIME_ORDER, ordered=True
    )
    daily = daily.sort_values(["Account", "trade_date"]).reset_index(drop=True)
    return daily


def create_trader_summary_metrics(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute per-account summary statistics."""
    results = []
    for acct, g in daily_metrics.groupby("Account"):
        g = g.sort_values("trade_date")
        cum_pnl = g["daily_pnl"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        total_closing = g["daily_closing_trades"].sum()
        gross_profit = g["daily_gross_profit"].sum()
        gross_loss = g["daily_gross_loss"].sum()
        mean_pnl = g["daily_pnl"].mean()
        std_pnl = g["daily_pnl"].std()
        neg_pnl = g.loc[g["daily_pnl"] < 0, "daily_pnl"]
        downside_std = neg_pnl.std() if len(neg_pnl) > 1 else np.nan

        results.append({
            "Account": acct,
            "total_pnl": g["daily_pnl"].sum(),
            "cumulative_max_pnl": cum_pnl.max(),
            "total_trades": int(g["daily_trades"].sum()),
            "total_trading_days": len(g),
            "overall_win_rate": g["daily_winning_trades"].sum() / total_closing if total_closing > 0 else 0,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
            "mean_daily_pnl": mean_pnl,
            "std_daily_pnl": std_pnl,
            "sharpe_ratio": (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0,
            "sortino_ratio": (mean_pnl / downside_std * np.sqrt(252)) if (downside_std and downside_std > 0) else 0,
            "max_drawdown": drawdown.min(),
            "avg_position_size": g["mean_position_size"].mean(),
            "avg_daily_volume": g["total_volume"].mean(),
            "avg_trade_return": g["daily_pnl"].sum() / total_closing if total_closing > 0 else 0,
            "risk_adjusted_return": g["daily_pnl"].sum() / g["total_volume"].sum() if g["total_volume"].sum() > 0 else 0,
            "long_bias": g["long_ratio"].mean(),
            "avg_daily_trades": g["daily_trades"].mean(),
        })
    return pd.DataFrame(results).set_index("Account")


def create_rolling_trader_metrics(daily_metrics: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Add rolling performance metrics per account."""
    df = daily_metrics.copy()
    parts = []
    for _, g in df.groupby("Account"):
        g = g.sort_values("trade_date").copy()
        roll_mean = g["daily_pnl"].rolling(window, min_periods=5).mean()
        roll_std = g["daily_pnl"].rolling(window, min_periods=5).std()
        g["rolling_sharpe"] = np.where(roll_std > 0, roll_mean / roll_std * np.sqrt(252), 0)
        win_roll = g["daily_winning_trades"].rolling(window, min_periods=5).sum()
        close_roll = g["daily_closing_trades"].rolling(window, min_periods=5).sum()
        g["rolling_win_rate"] = np.where(close_roll > 0, win_roll / close_roll, 0)
        g["rolling_pnl"] = g["daily_pnl"].rolling(window, min_periods=1).sum()
        g["rolling_volume"] = g["total_volume"].rolling(window, min_periods=1).sum()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def add_lagged_sentiment(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """Add lagged sentiment values per account."""
    df = daily_metrics.copy()
    parts = []
    for _, g in df.groupby("Account"):
        g = g.sort_values("trade_date").copy()
        for lag in [1, 3, 7]:
            g[f"sentiment_lag_{lag}"] = g["sentiment_value"].shift(lag)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def segment_traders(summary: pd.DataFrame) -> pd.DataFrame:
    """Assign traders to Top 10% / Middle 80% / Bottom 10% cohorts."""
    df = summary.copy()
    p10 = np.percentile(df["total_pnl"], 10)
    p90 = np.percentile(df["total_pnl"], 90)
    df["cohort"] = "Middle 80%"
    df.loc[df["total_pnl"] >= p90, "cohort"] = "Top 10%"
    df.loc[df["total_pnl"] <= p10, "cohort"] = "Bottom 10%"
    return df


def run_feature_engineering(fgi, merged):
    """Orchestrate all feature engineering."""
    print("Creating sentiment features...")
    fgi_feat = create_sentiment_features(fgi)
    fgi_feat = create_regime_features(fgi_feat)

    print("Creating trader daily metrics...")
    daily = create_trader_daily_metrics(merged)

    print("Creating trader summary metrics...")
    summary = create_trader_summary_metrics(daily)
    summary = segment_traders(summary)

    print("Creating rolling metrics...")
    rolling = create_rolling_trader_metrics(daily)
    rolling = add_lagged_sentiment(rolling)

    print(f"  Daily metrics: {daily.shape}")
    print(f"  Trader summary: {summary.shape}")
    print(f"  Cohorts: {summary['cohort'].value_counts().to_dict()}")

    return {
        "fgi_features": fgi_feat,
        "daily_metrics": rolling,  # includes rolling + lagged
        "trader_summary": summary,
    }
