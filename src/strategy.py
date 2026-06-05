"""
Strategy Research — Sentiment-based systematic trading strategies with backtesting.
"""

import pandas as pd
import numpy as np
from src.utils import REGIME_ORDER


def prepare_backtest_data(fgi, daily_metrics):
    """Create daily aggregate data for backtesting."""
    agg = daily_metrics.groupby("trade_date").agg(
        aggregate_pnl=("daily_pnl", "sum"),
        aggregate_volume=("total_volume", "sum"),
        num_traders=("Account", "nunique"),
        num_trades=("daily_trades", "sum"),
    ).reset_index()
    
    fgi_sub = fgi[["date", "value", "classification"]].copy()
    fgi_sub["date"] = pd.to_datetime(fgi_sub["date"])
    fgi_sub = fgi_sub.rename(columns={"value": "sentiment_value", "classification": "regime"})
    
    bt = fgi_sub.merge(agg, left_on="date", right_on="trade_date", how="inner")
    bt = bt.sort_values("date").reset_index(drop=True)
    bt = bt.drop(columns=["trade_date"], errors="ignore")
    return bt


def calculate_backtest_metrics(equity_curve):
    """Calculate comprehensive backtest performance metrics."""
    if len(equity_curve) < 2:
        return {}
    
    returns = equity_curve.diff().fillna(0)
    total_return = equity_curve.iloc[-1] - equity_curve.iloc[0]
    n_days = len(equity_curve)
    ann_factor = 252 / max(n_days, 1)
    ann_return = total_return * ann_factor
    
    daily_ret = returns
    vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    
    neg_ret = daily_ret[daily_ret < 0]
    downside_std = neg_ret.std() if len(neg_ret) > 1 else 1e-10
    sortino = daily_ret.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
    
    running_max = equity_curve.cummax()
    drawdown = equity_curve - running_max
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    pos_days = (daily_ret > 0).sum()
    neg_days = (daily_ret < 0).sum()
    win_rate = pos_days / (pos_days + neg_days) if (pos_days + neg_days) > 0 else 0
    
    gross_profit = daily_ret[daily_ret > 0].sum()
    gross_loss = abs(daily_ret[daily_ret < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_daily_return": daily_ret.mean(),
        "volatility": vol,
        "n_trading_days": n_days,
    }


def sentiment_reversal_strategy(bt_data):
    """Contrarian: long in Extreme Fear, short in Extreme Greed."""
    bt = bt_data.copy()
    bt["position"] = 0
    pos = 0
    trades = 0
    
    for i in range(len(bt)):
        sv = bt.iloc[i]["sentiment_value"]
        if sv <= 24 and pos != 1:
            pos = 1
            trades += 1
        elif sv >= 75 and pos != -1:
            pos = -1
            trades += 1
        elif 45 <= sv <= 54:
            if pos != 0:
                trades += 1
            pos = 0
        bt.iloc[i, bt.columns.get_loc("position")] = pos
    
    bt["strategy_pnl"] = bt["position"].shift(1).fillna(0) * bt["aggregate_pnl"]
    bt["equity"] = bt["strategy_pnl"].cumsum()
    
    metrics = calculate_backtest_metrics(bt["equity"])
    
    return {
        "name": "Sentiment Reversal (Contrarian)",
        "description": "Go long during Extreme Fear, short during Extreme Greed, flat in Neutral",
        "entry_rules": "Long when FGI ≤ 24 (Extreme Fear); Short when FGI ≥ 75 (Extreme Greed)",
        "exit_rules": "Flatten when FGI enters Neutral zone (45-54)",
        "risk_management": "Maximum 1x exposure, no leverage. Stop-loss: regime exit signal.",
        "position_sizing": "Full position (long +1 / short -1 / flat 0)",
        "equity_curve": bt["equity"],
        "dates": bt["date"],
        "num_trades": trades,
        "expected_edge": "Exploits mean-reversion in extreme sentiment. Edge depends on sentiment regime persistence.",
        "limitations": [
            "No transaction costs modeled",
            "Uses close-of-day sentiment (signal lag)",
            "Survivorship bias in trader sample",
            "Backtest period limited to available data overlap",
        ],
        **metrics
    }


def sentiment_momentum_strategy(bt_data):
    """Trade with sentiment momentum."""
    bt = bt_data.copy()
    bt["momentum_7d"] = bt["sentiment_value"].diff(7)
    bt["position"] = 0
    trades = 0
    pos = 0
    
    for i in range(7, len(bt)):
        mom = bt.iloc[i]["momentum_7d"]
        if pd.isna(mom):
            continue
        if mom > 10 and pos != 1:
            pos = 1
            trades += 1
        elif mom < -10 and pos != -1:
            pos = -1
            trades += 1
        elif abs(mom) <= 3:
            if pos != 0:
                trades += 1
            pos = 0
        bt.iloc[i, bt.columns.get_loc("position")] = pos
    
    bt["strategy_pnl"] = bt["position"].shift(1).fillna(0) * bt["aggregate_pnl"]
    bt["equity"] = bt["strategy_pnl"].cumsum()
    
    metrics = calculate_backtest_metrics(bt["equity"])
    
    return {
        "name": "Sentiment Momentum",
        "description": "Increase exposure when sentiment improves rapidly, reduce when declining",
        "entry_rules": "Long when 7d momentum > +10; Short when momentum < -10",
        "exit_rules": "Flatten when |momentum| ≤ 3",
        "risk_management": "Maximum 1x exposure. Momentum signal confirms trend direction.",
        "position_sizing": "Full position based on momentum direction",
        "equity_curve": bt["equity"],
        "dates": bt["date"],
        "num_trades": trades,
        "expected_edge": "Captures sentiment trends. Momentum persistence varies by market regime.",
        "limitations": [
            "No transaction costs",
            "Momentum lookback window (7d) not optimized",
            "Sentiment data is daily — intraday moves missed",
            "Survivorship bias in trader sample",
        ],
        **metrics
    }


def regime_sizing_strategy(bt_data):
    """Contrarian position sizing: larger positions in fear, smaller in greed."""
    bt = bt_data.copy()
    size_map = {
        "Extreme Fear": 1.0,
        "Fear": 0.75,
        "Neutral": 0.50,
        "Greed": 0.25,
        "Extreme Greed": -0.25,  # Small short
    }
    bt["position_size"] = bt["regime"].map(size_map).fillna(0.5)
    bt["strategy_pnl"] = bt["position_size"].shift(1).fillna(0.5) * bt["aggregate_pnl"]
    bt["equity"] = bt["strategy_pnl"].cumsum()
    
    # Count "trades" as regime changes
    trades = (bt["regime"] != bt["regime"].shift(1)).sum()
    
    metrics = calculate_backtest_metrics(bt["equity"])
    
    return {
        "name": "Regime-Based Contrarian Sizing",
        "description": "Size positions inversely to sentiment: large in fear, small/short in greed",
        "entry_rules": "Always in market. Size scales with fear level.",
        "exit_rules": "Continuous — position size adjusts daily based on regime",
        "risk_management": "Position size bounded [−0.25, 1.0]. No leverage beyond 1x.",
        "position_sizing": "Extreme Fear: 100%, Fear: 75%, Neutral: 50%, Greed: 25%, Extreme Greed: −25%",
        "equity_curve": bt["equity"],
        "dates": bt["date"],
        "num_trades": int(trades),
        "expected_edge": "Systematically 'buys fear, sells greed'. Exploits behavioral overreaction.",
        "limitations": [
            "No transaction costs",
            "Constant rebalancing assumed",
            "Backtest uses aggregate trader PnL as market proxy",
            "Does not account for slippage or market impact",
        ],
        **metrics
    }


def run_strategy_research(fgi, daily_metrics):
    """Run all strategy backtests and compare results."""
    print("Preparing backtest data...")
    bt_data = prepare_backtest_data(fgi, daily_metrics)
    print(f"  Backtest period: {bt_data['date'].min()} to {bt_data['date'].max()}")
    print(f"  Trading days: {len(bt_data)}")
    
    strategies = []
    
    print("Running Sentiment Reversal Strategy...")
    try:
        s1 = sentiment_reversal_strategy(bt_data)
        strategies.append(s1)
        print(f"  Sharpe: {s1.get('sharpe_ratio', 0):.2f}, Return: ${s1.get('total_return', 0):,.0f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("Running Sentiment Momentum Strategy...")
    try:
        s2 = sentiment_momentum_strategy(bt_data)
        strategies.append(s2)
        print(f"  Sharpe: {s2.get('sharpe_ratio', 0):.2f}, Return: ${s2.get('total_return', 0):,.0f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("Running Regime Sizing Strategy...")
    try:
        s3 = regime_sizing_strategy(bt_data)
        strategies.append(s3)
        print(f"  Sharpe: {s3.get('sharpe_ratio', 0):.2f}, Return: ${s3.get('total_return', 0):,.0f}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Comparison table
    comp_data = []
    for s in strategies:
        comp_data.append({
            "Strategy": s["name"],
            "Total Return ($)": s.get("total_return", 0),
            "Sharpe Ratio": s.get("sharpe_ratio", 0),
            "Sortino Ratio": s.get("sortino_ratio", 0),
            "Max Drawdown ($)": s.get("max_drawdown", 0),
            "Win Rate": s.get("win_rate", 0),
            "Profit Factor": s.get("profit_factor", 0),
            "Num Trades": s.get("num_trades", 0),
        })
    comparison = pd.DataFrame(comp_data)
    
    # Benchmark: buy and hold aggregate PnL
    bt_data["bh_equity"] = bt_data["aggregate_pnl"].cumsum()
    bh_metrics = calculate_backtest_metrics(bt_data["bh_equity"])
    
    return {
        "strategies": strategies,
        "comparison": comparison,
        "benchmark": bh_metrics,
        "backtest_data": bt_data,
    }
