"""
Utility constants, color palettes, and helper functions.
"""

import os
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

for _d in [DATA_PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Sentiment Regime Constants ───────────────────────────────────────────────
REGIME_BOUNDARIES = {
    "Extreme Fear": (0, 24),
    "Fear": (25, 44),
    "Neutral": (45, 54),
    "Greed": (55, 74),
    "Extreme Greed": (75, 100),
}

REGIME_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]

REGIME_SCORE_MAP = {
    "Extreme Fear": 0,
    "Fear": 25,
    "Neutral": 50,
    "Greed": 75,
    "Extreme Greed": 100,
}

# ── Color Palettes ───────────────────────────────────────────────────────────
REGIME_COLORS = {
    "Extreme Fear": "#c62828",
    "Fear": "#ef6c00",
    "Neutral": "#f9a825",
    "Greed": "#2e7d32",
    "Extreme Greed": "#1b5e20",
}

REGIME_COLORS_LIGHT = {
    "Extreme Fear": "#ef9a9a",
    "Fear": "#ffcc80",
    "Neutral": "#fff59d",
    "Greed": "#a5d6a7",
    "Extreme Greed": "#66bb6a",
}

# Cohort colors
COHORT_COLORS = {
    "Top 10%": "#1565c0",
    "Middle 80%": "#78909c",
    "Bottom 10%": "#c62828",
}

# General palette
PALETTE_MAIN = "#1565c0"
PALETTE_SECONDARY = "#ef6c00"
PALETTE_ACCENT = "#2e7d32"
PALETTE_NEGATIVE = "#c62828"
PALETTE_NEUTRAL = "#78909c"

# ── Matplotlib Configuration ─────────────────────────────────────────────────
def configure_plotting():
    """Set publication-quality matplotlib defaults."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.8,
    })
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ── Helper Functions ─────────────────────────────────────────────────────────
def save_figure(fig, name, tight=True):
    """Save figure to figures directory."""
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight" if tight else None, facecolor="white")
    plt.close(fig)
    return path


def format_pvalue(p):
    """Format p-value for display."""
    if p < 0.001:
        return "< 0.001***"
    elif p < 0.01:
        return f"{p:.4f}**"
    elif p < 0.05:
        return f"{p:.4f}*"
    else:
        return f"{p:.4f}"


def format_currency(x, prefix="$"):
    """Format number as currency."""
    if abs(x) >= 1_000_000:
        return f"{prefix}{x/1_000_000:,.1f}M"
    elif abs(x) >= 1_000:
        return f"{prefix}{x/1_000:,.1f}K"
    else:
        return f"{prefix}{x:,.2f}"


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    import numpy as np
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = pooled_std ** 0.5
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "Negligible"
    elif d < 0.5:
        return "Small"
    elif d < 0.8:
        return "Medium"
    else:
        return "Large"


def winsorize_series(s, lower=0.01, upper=0.99):
    """Winsorize a pandas Series at given quantiles."""
    import numpy as np
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)
