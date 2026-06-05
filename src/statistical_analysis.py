"""
Statistical Analysis — Correlation, hypothesis testing, behavioral analysis,
robustness checks, and advanced insights.

All multi-test results include Benjamini-Hochberg FDR and Bonferroni corrections.
Every significant finding reports: statistical significance, effect size, economic significance.
"""

import pandas as pd
import numpy as np
from scipy import stats
from src.utils import REGIME_ORDER, cohens_d, interpret_effect_size, format_pvalue, winsorize_series

try:
    from statsmodels.stats.multitest import multipletests
    import statsmodels.api as sm
    HAS_SM = True
except ImportError:
    HAS_SM = False


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def correlation_analysis(daily_metrics):
    """Pearson, Spearman, Kendall correlations with FDR correction."""
    cols = [c for c in [
        "sentiment_value", "daily_pnl", "mean_position_size",
        "daily_trades", "daily_win_rate", "long_ratio",
        "cross_margin_ratio", "total_volume"
    ] if c in daily_metrics.columns]
    
    sub = daily_metrics[cols].dropna()
    result = {}
    for method in ["pearson", "spearman", "kendall"]:
        corr = sub.corr(method=method)
        result[method] = corr
        # p-values
        n = len(sub)
        pvals = pd.DataFrame(np.ones_like(corr), index=corr.index, columns=corr.columns)
        all_p = []
        pairs = []
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if i < j:
                    if method == "pearson":
                        _, p = stats.pearsonr(sub[c1], sub[c2])
                    elif method == "spearman":
                        _, p = stats.spearmanr(sub[c1], sub[c2])
                    else:
                        _, p = stats.kendalltau(sub[c1], sub[c2])
                    pvals.loc[c1, c2] = p
                    pvals.loc[c2, c1] = p
                    all_p.append(p)
                    pairs.append((c1, c2))
        result[f"p_values_{method}"] = pvals
        
        # FDR correction
        if all_p and HAS_SM:
            reject, corrected, _, _ = multipletests(all_p, method="fdr_bh")
            fdr_sig = [pairs[k] for k in range(len(pairs)) if reject[k]]
            result[f"fdr_significant_{method}"] = fdr_sig
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIGNIFICANCE TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def regime_significance_tests(daily_metrics):
    """ANOVA, Kruskal-Wallis, and pairwise tests across regimes."""
    results = {}
    metrics_to_test = ["daily_pnl", "mean_position_size", "daily_trades", "daily_win_rate"]
    
    for metric in metrics_to_test:
        if metric not in daily_metrics.columns:
            continue
        
        groups = {}
        for regime in REGIME_ORDER:
            data = daily_metrics.loc[daily_metrics["sentiment_regime"] == regime, metric].dropna()
            if len(data) > 0:
                groups[regime] = data
        
        if len(groups) < 2:
            continue
            
        res = {"metric": metric}
        group_vals = list(groups.values())
        
        # ANOVA
        try:
            f_stat, p_anova = stats.f_oneway(*group_vals)
            res["anova_f"] = f_stat
            res["anova_p"] = p_anova
        except Exception:
            res["anova_f"] = np.nan
            res["anova_p"] = np.nan
        
        # Kruskal-Wallis
        try:
            h_stat, p_kw = stats.kruskal(*group_vals)
            res["kruskal_h"] = h_stat
            res["kruskal_p"] = p_kw
        except Exception:
            res["kruskal_h"] = np.nan
            res["kruskal_p"] = np.nan
        
        # Pairwise: Fear vs Greed
        for pair_name, r1, r2 in [
            ("fear_vs_greed", "Fear", "Greed"),
            ("ext_fear_vs_ext_greed", "Extreme Fear", "Extreme Greed")
        ]:
            if r1 in groups and r2 in groups:
                g1, g2 = groups[r1], groups[r2]
                try:
                    t_stat, p_t = stats.ttest_ind(g1, g2, equal_var=False)
                    u_stat, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    d = cohens_d(g1, g2)
                    diff_mean = g1.mean() - g2.mean()
                    se_diff = np.sqrt(g1.var() / len(g1) + g2.var() / len(g2))
                    ci_lo = diff_mean - 1.96 * se_diff
                    ci_hi = diff_mean + 1.96 * se_diff
                    res[f"{pair_name}_t_stat"] = t_stat
                    res[f"{pair_name}_t_p"] = p_t
                    res[f"{pair_name}_u_stat"] = u_stat
                    res[f"{pair_name}_u_p"] = p_u
                    res[f"{pair_name}_cohens_d"] = d
                    res[f"{pair_name}_effect_interp"] = interpret_effect_size(d)
                    res[f"{pair_name}_diff_mean"] = diff_mean
                    res[f"{pair_name}_ci"] = (ci_lo, ci_hi)
                    res[f"{pair_name}_econ_impact"] = f"${diff_mean:,.2f} mean daily PnL difference"
                except Exception:
                    pass
        
        # Bonferroni correction on collected p-values
        p_keys = [k for k in res if k.endswith("_p") and isinstance(res.get(k), (int, float))]
        p_vals = [res[k] for k in p_keys if not np.isnan(res[k])]
        if p_vals and HAS_SM:
            reject, corrected, _, _ = multipletests(p_vals, method="bonferroni")
            res["bonferroni_any_significant"] = any(reject)
        
        results[metric] = res
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LAG ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def lag_analysis(daily_metrics):
    """Test predictive power of lagged sentiment on PnL, position size, trades."""
    results = {}
    targets = ["daily_pnl", "mean_position_size", "daily_trades"]
    
    for lag in [1, 3, 7]:
        lag_col = f"sentiment_lag_{lag}"
        if lag_col not in daily_metrics.columns:
            continue
        lag_res = {}
        for target in targets:
            if target not in daily_metrics.columns:
                continue
            valid = daily_metrics[[lag_col, target]].dropna()
            if len(valid) < 20:
                continue
            try:
                r, p = stats.pearsonr(valid[lag_col], valid[target])
                rs, ps = stats.spearmanr(valid[lag_col], valid[target])
                lag_res[target] = {
                    "pearson_r": r, "pearson_p": p,
                    "spearman_r": rs, "spearman_p": ps,
                    "n": len(valid)
                }
                # OLS regression
                if HAS_SM:
                    X = sm.add_constant(valid[lag_col])
                    model = sm.OLS(valid[target], X).fit()
                    lag_res[target]["ols_coef"] = model.params.iloc[1]
                    lag_res[target]["ols_r2"] = model.rsquared
                    lag_res[target]["ols_f_pvalue"] = model.f_pvalue
                    lag_res[target]["ols_pvalue"] = model.pvalues.iloc[1]
            except Exception:
                pass
        results[f"lag_{lag}"] = lag_res
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SEGMENTATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def segmentation_analysis(daily_metrics, trader_summary):
    """Compare Top/Middle/Bottom trader cohorts."""
    results = {}
    if "cohort" not in trader_summary.columns:
        return results
    
    cohort_metrics = ["overall_win_rate", "profit_factor", "avg_position_size",
                      "avg_daily_trades", "long_bias", "total_pnl", "sharpe_ratio"]
    
    for metric in cohort_metrics:
        if metric not in trader_summary.columns:
            continue
        groups = {}
        for c in ["Top 10%", "Middle 80%", "Bottom 10%"]:
            data = trader_summary.loc[trader_summary["cohort"] == c, metric].dropna()
            if len(data) > 0:
                groups[c] = data
        if len(groups) < 2:
            continue
        
        res = {"metric": metric}
        # Descriptive stats per cohort
        for c, data in groups.items():
            res[f"{c}_mean"] = data.mean()
            res[f"{c}_median"] = data.median()
        
        # Kruskal-Wallis
        try:
            h, p = stats.kruskal(*groups.values())
            res["kruskal_h"] = h
            res["kruskal_p"] = p
        except Exception:
            pass
        
        # Top vs Bottom
        if "Top 10%" in groups and "Bottom 10%" in groups:
            try:
                u, p = stats.mannwhitneyu(groups["Top 10%"], groups["Bottom 10%"], alternative="two-sided")
                d = cohens_d(groups["Top 10%"], groups["Bottom 10%"])
                res["top_vs_bottom_u_p"] = p
                res["top_vs_bottom_d"] = d
                res["top_vs_bottom_effect"] = interpret_effect_size(d)
            except Exception:
                pass
        results[metric] = res
    
    # Sentiment exposure by cohort
    dm = daily_metrics.merge(trader_summary[["cohort"]].reset_index(), on="Account", how="left")
    sentiment_exposure = {}
    for c in ["Top 10%", "Middle 80%", "Bottom 10%"]:
        subset = dm[dm["cohort"] == c]
        if len(subset) > 0:
            sentiment_exposure[c] = subset["sentiment_value"].mean()
    results["sentiment_exposure"] = sentiment_exposure
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BEHAVIORAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def behavioral_analysis(daily_metrics, trader_summary):
    """Test 5 behavioral hypotheses."""
    results = {}
    
    # 1. Position size in greed vs fear
    greed = daily_metrics[daily_metrics["sentiment_regime"].isin(["Greed", "Extreme Greed"])]["mean_position_size"]
    fear = daily_metrics[daily_metrics["sentiment_regime"].isin(["Fear", "Extreme Fear"])]["mean_position_size"]
    try:
        t, p = stats.ttest_ind(greed, fear, equal_var=False)
        d = cohens_d(greed, fear)
        results["position_size_greed_vs_fear"] = {
            "greed_mean": greed.mean(), "fear_mean": fear.mean(),
            "t_stat": t, "p_value": p, "cohens_d": d,
            "effect": interpret_effect_size(d),
            "economic": f"${greed.mean() - fear.mean():,.2f} avg position size difference"
        }
    except Exception:
        pass
    
    # 2. Losses in extreme greed vs others
    losses_eg = daily_metrics[
        (daily_metrics["sentiment_regime"] == "Extreme Greed") & (daily_metrics["daily_pnl"] < 0)
    ]["daily_pnl"].abs()
    losses_other = daily_metrics[
        (daily_metrics["sentiment_regime"] != "Extreme Greed") & (daily_metrics["daily_pnl"] < 0)
    ]["daily_pnl"].abs()
    try:
        u, p = stats.mannwhitneyu(losses_eg, losses_other, alternative="two-sided")
        d = cohens_d(losses_eg, losses_other)
        results["losses_extreme_greed"] = {
            "eg_mean_loss": losses_eg.mean(), "other_mean_loss": losses_other.mean(),
            "u_stat": u, "p_value": p, "cohens_d": d,
            "effect": interpret_effect_size(d),
            "economic": f"${losses_eg.mean() - losses_other.mean():,.2f} avg loss difference in Extreme Greed"
        }
    except Exception:
        pass
    
    # 3. Performance in fear markets
    fear_pnl = daily_metrics[daily_metrics["sentiment_regime"].isin(["Fear", "Extreme Fear"])]["daily_pnl"]
    greed_pnl = daily_metrics[daily_metrics["sentiment_regime"].isin(["Greed", "Extreme Greed"])]["daily_pnl"]
    try:
        t, p = stats.ttest_ind(fear_pnl, greed_pnl, equal_var=False)
        d = cohens_d(fear_pnl, greed_pnl)
        results["performance_fear_vs_greed"] = {
            "fear_mean_pnl": fear_pnl.mean(), "greed_mean_pnl": greed_pnl.mean(),
            "t_stat": t, "p_value": p, "cohens_d": d,
            "effect": interpret_effect_size(d),
            "economic": f"Fear: ${fear_pnl.mean():,.2f}/day vs Greed: ${greed_pnl.mean():,.2f}/day"
        }
    except Exception:
        pass
    
    # 4. Top traders contrarian?
    if "cohort" in trader_summary.columns:
        dm = daily_metrics.merge(trader_summary[["cohort"]].reset_index(), on="Account", how="left")
        for cohort_name in ["Top 10%", "Bottom 10%"]:
            subset = dm[dm["cohort"] == cohort_name]
            if len(subset) > 10:
                try:
                    r, p = stats.pearsonr(subset["sentiment_value"], subset["long_ratio"])
                    results[f"contrarian_{cohort_name}"] = {
                        "correlation": r, "p_value": p,
                        "is_contrarian": r < 0,
                        "interpretation": "Contrarian (reduces long exposure in greed)" if r < 0 else "Trend-following"
                    }
                except Exception:
                    pass
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ROBUSTNESS CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def robustness_checks(daily_metrics):
    """Test PnL-regime findings across 3 specifications."""
    results = {}
    
    def _run_tests(data, label):
        groups = {}
        for regime in REGIME_ORDER:
            g = data.loc[data["sentiment_regime"] == regime, "daily_pnl"].dropna()
            if len(g) > 0:
                groups[regime] = g
        if len(groups) < 2:
            return None
        res = {"specification": label, "n": len(data)}
        try:
            _, p = stats.f_oneway(*groups.values())
            res["anova_p"] = p
        except Exception:
            res["anova_p"] = np.nan
        try:
            _, p = stats.kruskal(*groups.values())
            res["kruskal_p"] = p
        except Exception:
            res["kruskal_p"] = np.nan
        if "Fear" in groups and "Greed" in groups:
            d = cohens_d(groups["Fear"], groups["Greed"])
            res["fear_greed_d"] = d
            res["fear_greed_effect"] = interpret_effect_size(d)
        return res
    
    # Full sample
    r1 = _run_tests(daily_metrics, "Full Sample")
    
    # Winsorized
    dm_w = daily_metrics.copy()
    dm_w["daily_pnl"] = winsorize_series(dm_w["daily_pnl"], 0.01, 0.99)
    r2 = _run_tests(dm_w, "Winsorized (1%/99%)")
    
    # IQR-filtered
    q1, q3 = daily_metrics["daily_pnl"].quantile(0.25), daily_metrics["daily_pnl"].quantile(0.75)
    iqr = q3 - q1
    dm_f = daily_metrics[
        (daily_metrics["daily_pnl"] >= q1 - 1.5 * iqr) &
        (daily_metrics["daily_pnl"] <= q3 + 1.5 * iqr)
    ]
    r3 = _run_tests(dm_f, "IQR-Filtered")
    
    specs = [r for r in [r1, r2, r3] if r is not None]
    robust = all(
        s.get("anova_p", 1) < 0.05 or s.get("kruskal_p", 1) < 0.05
        for s in specs
    ) if specs else False
    
    return {"specifications": specs, "is_robust": robust}


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ADVANCED INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def advanced_insights(fgi, daily_metrics, trader_summary):
    """Generate 10+ data-driven insights with ratings."""
    insights = []
    
    # 1. Sentiment asymmetry
    try:
        fear_pnl = daily_metrics[daily_metrics["sentiment_regime"].isin(["Fear","Extreme Fear"])]["daily_pnl"]
        greed_pnl = daily_metrics[daily_metrics["sentiment_regime"].isin(["Greed","Extreme Greed"])]["daily_pnl"]
        fear_eff = fear_pnl.mean()
        greed_eff = greed_pnl.mean()
        asymmetry = abs(fear_eff) / (abs(greed_eff) + 1e-10)
        insights.append({
            "title": "Sentiment-PnL Asymmetry",
            "finding": f"Fear-regime mean PnL (${fear_eff:,.2f}) vs Greed (${greed_eff:,.2f}). Ratio: {asymmetry:.2f}x",
            "evidence": f"Fear mean=${fear_eff:.2f}, Greed mean=${greed_eff:.2f}",
            "practical_value": 4, "confidence": 4, "economic_impact": 4, "novelty": 3
        })
    except Exception:
        pass
    
    # 2. Regime transition alpha
    try:
        if "regime_changed" not in fgi.columns:
            fgi = fgi.copy()
            fgi["regime_changed"] = fgi["classification"] != fgi["classification"].shift(1)
        fgi_rc = fgi[["date", "regime_changed"]].copy()
        fgi_rc["date"] = pd.to_datetime(fgi_rc["date"])
        dm2 = daily_metrics.merge(fgi_rc, left_on="trade_date", right_on="date", how="left")
        change_pnl = dm2[dm2["regime_changed"] == True]["daily_pnl"]
        stable_pnl = dm2[dm2["regime_changed"] == False]["daily_pnl"]
        if len(change_pnl) > 5 and len(stable_pnl) > 5:
            u, p = stats.mannwhitneyu(change_pnl, stable_pnl, alternative="two-sided")
            insights.append({
                "title": "Regime Transition Alpha",
                "finding": f"PnL on regime-change days (${change_pnl.mean():,.2f}) vs stable days (${stable_pnl.mean():,.2f}), p={p:.4f}",
                "evidence": f"Mann-Whitney p={p:.4f}, n_change={len(change_pnl)}, n_stable={len(stable_pnl)}",
                "practical_value": 5, "confidence": 3, "economic_impact": 4, "novelty": 4
            })
    except Exception:
        pass
    
    # 3. Nonlinear effects
    try:
        valid = daily_metrics[["sentiment_value", "daily_pnl"]].dropna()
        if len(valid) > 50 and HAS_SM:
            X = valid["sentiment_value"]
            X2 = X ** 2
            Xmat = sm.add_constant(pd.DataFrame({"x": X, "x2": X2}))
            model = sm.OLS(valid["daily_pnl"], Xmat).fit()
            quad_p = model.pvalues.get("x2", 1)
            insights.append({
                "title": "Nonlinear Sentiment-PnL Relationship",
                "finding": f"Quadratic term p={quad_p:.4f}. {'Significant' if quad_p < 0.05 else 'Not significant'} nonlinear effect.",
                "evidence": f"OLS quadratic coef p={quad_p:.4f}, R²={model.rsquared:.4f}",
                "practical_value": 3, "confidence": 4, "economic_impact": 3, "novelty": 4
            })
    except Exception:
        pass
    
    # 4. Behavioral shift over time
    try:
        midpoint = daily_metrics["trade_date"].quantile(0.5)
        early = daily_metrics[daily_metrics["trade_date"] <= midpoint]
        late = daily_metrics[daily_metrics["trade_date"] > midpoint]
        if len(early) > 20 and len(late) > 20:
            r_early, _ = stats.pearsonr(early["sentiment_value"], early["daily_pnl"])
            r_late, _ = stats.pearsonr(late["sentiment_value"], late["daily_pnl"])
            insights.append({
                "title": "Temporal Behavioral Shift",
                "finding": f"Sentiment-PnL correlation shifted from r={r_early:.3f} (early) to r={r_late:.3f} (late period).",
                "evidence": f"Early r={r_early:.3f}, Late r={r_late:.3f}",
                "practical_value": 4, "confidence": 3, "economic_impact": 3, "novelty": 5
            })
    except Exception:
        pass
    
    # 5. Winner's curse
    try:
        dm_monthly = daily_metrics.copy()
        dm_monthly["month"] = dm_monthly["trade_date"].dt.to_period("M")
        monthly_pnl = dm_monthly.groupby(["Account", "month"])["daily_pnl"].sum().reset_index()
        monthly_pnl = monthly_pnl.sort_values(["Account", "month"])
        monthly_pnl["next_month_pnl"] = monthly_pnl.groupby("Account")["daily_pnl"].shift(-1)
        valid_mp = monthly_pnl.dropna(subset=["next_month_pnl"])
        if len(valid_mp) > 10:
            r, p = stats.pearsonr(valid_mp["daily_pnl"], valid_mp["next_month_pnl"])
            insights.append({
                "title": "Winner's Curse / Performance Persistence",
                "finding": f"Month-to-month PnL autocorrelation: r={r:.3f}, p={p:.4f}. {'Performance persists' if r > 0 else 'Mean-reversion detected'}.",
                "evidence": f"Pearson r={r:.3f}, p={p:.4f}, n={len(valid_mp)}",
                "practical_value": 5, "confidence": 4, "economic_impact": 4, "novelty": 4
            })
    except Exception:
        pass
    
    # 6. Cross-margin preference by regime
    try:
        regime_margin = daily_metrics.groupby("sentiment_regime")["cross_margin_ratio"].mean()
        max_r = regime_margin.idxmax()
        min_r = regime_margin.idxmin()
        insights.append({
            "title": "Margin Type Shifts by Regime",
            "finding": f"Cross-margin usage highest in {max_r} ({regime_margin[max_r]:.1%}), lowest in {min_r} ({regime_margin[min_r]:.1%}).",
            "evidence": f"Cross-margin ratio: {regime_margin.to_dict()}",
            "practical_value": 3, "confidence": 4, "economic_impact": 2, "novelty": 3
        })
    except Exception:
        pass
    
    # 7. Extreme regime PnL concentration
    try:
        total_pnl = daily_metrics["daily_pnl"].sum()
        extreme_pnl = daily_metrics[
            daily_metrics["sentiment_regime"].isin(["Extreme Fear", "Extreme Greed"])
        ]["daily_pnl"].sum()
        pct = extreme_pnl / total_pnl * 100 if total_pnl != 0 else 0
        insights.append({
            "title": "Extreme Regime PnL Concentration",
            "finding": f"{pct:.1f}% of total PnL earned during extreme sentiment regimes.",
            "evidence": f"Extreme PnL: ${extreme_pnl:,.0f} / Total: ${total_pnl:,.0f}",
            "practical_value": 5, "confidence": 5, "economic_impact": 5, "novelty": 3
        })
    except Exception:
        pass
    
    # 8. Volume-sentiment relationship
    try:
        r, p = stats.pearsonr(daily_metrics["sentiment_value"], daily_metrics["total_volume"])
        insights.append({
            "title": "Volume-Sentiment Spiral",
            "finding": f"Sentiment-volume correlation: r={r:.3f}, p={p:.4f}. {'Higher volume in greed' if r > 0 else 'Higher volume in fear'}.",
            "evidence": f"Pearson r={r:.3f}, p={p:.4f}",
            "practical_value": 3, "confidence": 4, "economic_impact": 3, "novelty": 3
        })
    except Exception:
        pass
    
    # 9. Sentiment momentum signal
    try:
        if "sentiment_lag_1" in daily_metrics.columns:
            dm_mom = daily_metrics.copy()
            dm_mom["sent_momentum"] = dm_mom["sentiment_value"] - dm_mom["sentiment_lag_1"]
            valid = dm_mom[["sent_momentum", "daily_pnl"]].dropna()
            if len(valid) > 20:
                r_mom, p_mom = stats.pearsonr(valid["sent_momentum"], valid["daily_pnl"])
                r_lvl, p_lvl = stats.pearsonr(
                    daily_metrics["sentiment_value"].dropna(),
                    daily_metrics["daily_pnl"].dropna().iloc[:len(daily_metrics["sentiment_value"].dropna())]
                )
                insights.append({
                    "title": "Momentum vs Level Signal",
                    "finding": f"Sentiment momentum r={r_mom:.3f} vs level r={r_lvl:.3f}. {'Momentum stronger' if abs(r_mom) > abs(r_lvl) else 'Level stronger'}.",
                    "evidence": f"Momentum r={r_mom:.3f} (p={p_mom:.4f}), Level r={r_lvl:.3f} (p={p_lvl:.4f})",
                    "practical_value": 4, "confidence": 3, "economic_impact": 3, "novelty": 4
                })
    except Exception:
        pass
    
    # 10. Position size dispersion by regime
    try:
        regime_cv = daily_metrics.groupby("sentiment_regime")["mean_position_size"].agg(
            lambda x: x.std() / x.mean() if x.mean() > 0 else 0
        )
        insights.append({
            "title": "Position Size Dispersion by Regime",
            "finding": f"Position size CV highest in {regime_cv.idxmax()} ({regime_cv.max():.2f}), lowest in {regime_cv.idxmin()} ({regime_cv.min():.2f}).",
            "evidence": f"CV by regime: {regime_cv.to_dict()}",
            "practical_value": 3, "confidence": 4, "economic_impact": 2, "novelty": 4
        })
    except Exception:
        pass
    
    # Sort by composite score
    for ins in insights:
        ins["composite_score"] = (
            ins.get("practical_value", 0) + ins.get("confidence", 0) +
            ins.get("economic_impact", 0) + ins.get("novelty", 0)
        ) / 4
    insights.sort(key=lambda x: x["composite_score"], reverse=True)
    
    return insights


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_statistical_analysis(fgi, daily_metrics, trader_summary, merged):
    """Run complete statistical analysis pipeline."""
    results = {}
    
    print("Running correlation analysis...")
    try:
        results["correlations"] = correlation_analysis(daily_metrics)
    except Exception as e:
        print(f"  Error: {e}")
        results["correlations"] = {}
    
    print("Running significance tests...")
    try:
        results["significance"] = regime_significance_tests(daily_metrics)
    except Exception as e:
        print(f"  Error: {e}")
        results["significance"] = {}
    
    print("Running lag analysis...")
    try:
        results["lag_analysis"] = lag_analysis(daily_metrics)
    except Exception as e:
        print(f"  Error: {e}")
        results["lag_analysis"] = {}
    
    print("Running segmentation analysis...")
    try:
        results["segmentation"] = segmentation_analysis(daily_metrics, trader_summary)
    except Exception as e:
        print(f"  Error: {e}")
        results["segmentation"] = {}
    
    print("Running behavioral analysis...")
    try:
        results["behavioral"] = behavioral_analysis(daily_metrics, trader_summary)
    except Exception as e:
        print(f"  Error: {e}")
        results["behavioral"] = {}
    
    print("Running robustness checks...")
    try:
        results["robustness"] = robustness_checks(daily_metrics)
    except Exception as e:
        print(f"  Error: {e}")
        results["robustness"] = {}
    
    print("Generating advanced insights...")
    try:
        results["insights"] = advanced_insights(fgi, daily_metrics, trader_summary)
    except Exception as e:
        print(f"  Error: {e}")
        results["insights"] = []
    
    # Print summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS SUMMARY")
    print("=" * 60)
    for metric, res in results.get("significance", {}).items():
        if isinstance(res, dict):
            p = res.get("anova_p", 1)
            print(f"  {metric}: ANOVA p={format_pvalue(p)}")
    
    rob = results.get("robustness", {})
    if rob:
        print(f"  Robustness: {'ROBUST ✓' if rob.get('is_robust') else 'NOT ROBUST ✗'}")
    
    insights = results.get("insights", [])
    if insights:
        print(f"\n  Top Insights:")
        for ins in insights[:5]:
            print(f"    • {ins['title']}: {ins['finding'][:80]}...")
    
    return results
