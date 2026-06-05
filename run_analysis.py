"""
Master execution script — runs the complete analysis pipeline.
This validates all modules and generates all outputs.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("HYPERLIQUID SENTIMENT × TRADER PERFORMANCE")
print("Quantitative Research Pipeline — Full Execution")
print("=" * 70)

# ── Phase 1: Data Loading & Cleaning ─────────────────────────────────────────
print("\n▶ PHASE 1: Data Loading & Cleaning")
from src.data_loader import load_and_prepare
fgi, trader, merged, quality_report = load_and_prepare()

# ── Phase 2: Feature Engineering ─────────────────────────────────────────────
print("\n▶ PHASE 2: Feature Engineering")
from src.feature_engineering import run_feature_engineering
features = run_feature_engineering(fgi, merged)
fgi_feat = features["fgi_features"]
daily_metrics = features["daily_metrics"]
trader_summary = features["trader_summary"]

print(f"\n  Trader Summary:")
print(f"    Total PnL range: ${trader_summary['total_pnl'].min():,.0f} to ${trader_summary['total_pnl'].max():,.0f}")
print(f"    Sharpe range: {trader_summary['sharpe_ratio'].min():.2f} to {trader_summary['sharpe_ratio'].max():.2f}")
print(f"    Win rate range: {trader_summary['overall_win_rate'].min():.1%} to {trader_summary['overall_win_rate'].max():.1%}")

# ── Phase 3: Visualization ───────────────────────────────────────────────────
print("\n▶ PHASE 3: Visualization")
from src.visualization import generate_all_plots

# ── Phase 4-6: Statistical Analysis ──────────────────────────────────────────
print("\n▶ PHASE 4-6: Statistical Analysis")
from src.statistical_analysis import run_statistical_analysis
stat_results = run_statistical_analysis(fgi_feat, daily_metrics, trader_summary, merged)

# Now generate plots (need corr results)
print("\n▶ PHASE 3 (continued): Generating plots...")
saved_plots = generate_all_plots(fgi_feat, daily_metrics, trader_summary, merged,
                                  corr_results=stat_results.get("correlations"))

# ── Phase 7: Strategy Research ───────────────────────────────────────────────
print("\n▶ PHASE 7: Strategy Research")
from src.strategy import run_strategy_research
strategy_results = run_strategy_research(fgi_feat, daily_metrics)

print("\nStrategy Comparison:")
print(strategy_results["comparison"].to_string(index=False))

# ── Phase 8: Advanced Insights ───────────────────────────────────────────────
print("\n▶ PHASE 8: Advanced Insights")
insights = stat_results.get("insights", [])
print(f"  Generated {len(insights)} insights")
for i, ins in enumerate(insights, 1):
    score = ins.get("composite_score", 0)
    print(f"  {i}. [{score:.1f}] {ins['title']}")
    print(f"     {ins['finding'][:100]}")

# ── Save results ─────────────────────────────────────────────────────────────
print("\n▶ Saving strategy equity curves...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.utils import save_figure, configure_plotting, PALETTE_MAIN, PALETTE_ACCENT, PALETTE_NEGATIVE

configure_plotting()

# Strategy equity curves
fig, ax = plt.subplots(figsize=(14, 6))
for s in strategy_results["strategies"]:
    ax.plot(s["dates"], s["equity_curve"], label=s["name"], linewidth=1.5)
bt_data = strategy_results["backtest_data"]
ax.plot(bt_data["date"], bt_data["bh_equity"], label="Buy & Hold (Benchmark)",
        linestyle="--", color="gray", linewidth=1)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative PnL ($)")
ax.set_title("Strategy Equity Curves", fontsize=14, fontweight="bold")
ax.legend()
ax.axhline(0, color="black", linewidth=0.5)
save_figure(fig, "strategy_equity_curves")
print("  ✓ strategy_equity_curves.png")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE ✓")
print("=" * 70)

# Dump key results for notebook/report
import json

# Serialize what we can
summary_data = {
    "quality_report": {
        "fgi_rows": quality_report["fgi"]["clean_rows"],
        "trader_rows": quality_report["trader"]["clean_rows"],
        "merged_rows": quality_report["merged"]["rows"],
        "overlap_days": quality_report["merged"]["overlap_days"],
    },
    "trader_count": len(trader_summary),
    "total_pnl": float(trader_summary["total_pnl"].sum()),
    "insights_count": len(insights),
    "plots_generated": len(saved_plots),
    "robustness": stat_results.get("robustness", {}).get("is_robust", False),
}
print(f"\nSummary: {json.dumps(summary_data, indent=2)}")
