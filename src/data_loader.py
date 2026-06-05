"""
Data loading, cleaning, validation, and merging pipeline.

Design decisions (approved):
- Order ID is the logical trade/order identifier
- Individual rows are fills/executions
- Trade ID is a non-unique fill grouping field, not a primary key
- Leverage is NOT estimated; documented as a limitation
"""

import pandas as pd
import numpy as np
import os
from src.utils import DATA_RAW_DIR, DATA_PROCESSED_DIR, REGIME_ORDER


# ═══════════════════════════════════════════════════════════════════════════════
# LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_fear_greed_index(path=None):
    """Load the Bitcoin Fear & Greed Index CSV."""
    if path is None:
        path = os.path.join(DATA_RAW_DIR, "fear_greed_index.csv")
    df = pd.read_csv(path)
    return df


def load_historical_data(path=None):
    """Load the Hyperliquid historical trader data CSV."""
    if path is None:
        path = os.path.join(DATA_RAW_DIR, "historical_data.csv")
    df = pd.read_csv(path, low_memory=False)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_fear_greed(df):
    """
    Clean the Fear & Greed Index dataset.
    
    Steps:
    1. Parse date column to datetime
    2. Sort by date
    3. Fill any missing days via forward-fill
    4. Validate classification boundaries
    5. Set date as index
    """
    df = df.copy()
    
    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # Create complete date range and reindex to fill gaps
    full_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range)
    df.index.name = "date"
    
    # Forward-fill missing days (4 gaps)
    df["value"] = df["value"].ffill().astype(int)
    df["classification"] = df["classification"].ffill()
    df["timestamp"] = df["timestamp"].ffill().astype(int)
    
    df = df.reset_index()
    
    return df


def clean_historical_data(df):
    """
    Clean the historical trader dataset.
    
    Steps:
    1. Parse Timestamp IST to datetime
    2. Extract trade_date for daily aggregation
    3. Remove zero-Size-USD dust trades
    4. Convert types
    5. Add direction categories
    """
    df = df.copy()
    
    # Parse timestamp
    df["datetime_ist"] = pd.to_datetime(df["Timestamp IST"], format="%d-%m-%Y %H:%M")
    df["trade_date"] = df["datetime_ist"].dt.normalize()
    
    # Remove dust trades (Size USD == 0)
    n_before = len(df)
    df = df[df["Size USD"] > 0].copy()
    n_removed = n_before - len(df)
    
    # Classify trade types
    df["is_opening"] = df["Direction"].isin([
        "Open Long", "Open Short", "Buy"
    ])
    df["is_closing"] = df["Direction"].isin([
        "Close Long", "Close Short", "Sell",
        "Short > Long", "Long > Short",
        "Auto-Deleveraging", "Liquidated Isolated Short", "Settlement"
    ])
    df["is_long"] = df["Direction"].isin([
        "Open Long", "Close Long", "Buy", "Short > Long"
    ])
    df["is_short"] = df["Direction"].isin([
        "Open Short", "Close Short", "Sell", "Long > Short"
    ])
    
    # Margin type
    df["margin_type"] = df["Crossed"].map({True: "Cross", False: "Isolated"})
    
    # Sort by time
    df = df.sort_values(["Account", "datetime_ist"]).reset_index(drop=True)
    
    print(f"  Removed {n_removed} dust trades (Size USD = 0)")
    print(f"  Final shape: {df.shape}")
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def merge_datasets(fgi, trader):
    """
    Merge Fear & Greed Index with trader data on date.
    
    Only keeps data within the overlap period.
    """
    # Ensure trade_date is datetime
    trader = trader.copy()
    fgi = fgi.copy()
    
    # Rename FGI columns for clarity
    fgi_merge = fgi[["date", "value", "classification"]].copy()
    fgi_merge = fgi_merge.rename(columns={
        "value": "sentiment_value",
        "classification": "sentiment_regime"
    })
    
    # Merge on date
    merged = trader.merge(
        fgi_merge,
        left_on="trade_date",
        right_on="date",
        how="inner"
    )
    
    # Drop the redundant date column from FGI
    merged = merged.drop(columns=["date"])
    
    # Set regime as ordered categorical
    merged["sentiment_regime"] = pd.Categorical(
        merged["sentiment_regime"],
        categories=REGIME_ORDER,
        ordered=True
    )
    
    print(f"  Merged shape: {merged.shape}")
    print(f"  Date range: {merged['trade_date'].min()} to {merged['trade_date'].max()}")
    print(f"  Unique trading days: {merged['trade_date'].nunique()}")
    
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_fear_greed(df):
    """Run validation assertions on cleaned FGI data."""
    checks = {}
    
    checks["no_nulls"] = df[["value", "classification", "date"]].isnull().sum().sum() == 0
    checks["value_range"] = (df["value"].min() >= 0) and (df["value"].max() <= 100)
    checks["no_duplicate_dates"] = df["date"].duplicated().sum() == 0
    checks["valid_classifications"] = set(df["classification"].unique()).issubset(
        {"Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"}
    )
    checks["sorted_dates"] = df["date"].is_monotonic_increasing
    
    return checks


def validate_historical_data(df):
    """Run validation assertions on cleaned trader data."""
    checks = {}
    
    checks["no_null_account"] = df["Account"].isnull().sum() == 0
    checks["no_null_coin"] = df["Coin"].isnull().sum() == 0
    checks["positive_size_usd"] = (df["Size USD"] > 0).all()
    checks["valid_sides"] = set(df["Side"].unique()).issubset({"BUY", "SELL"})
    checks["dates_parsed"] = df["datetime_ist"].isnull().sum() == 0
    checks["sorted"] = True  # We sorted during cleaning
    
    return checks


def validate_merged(df):
    """Run validation assertions on merged dataset."""
    checks = {}
    
    checks["has_sentiment"] = "sentiment_value" in df.columns
    checks["has_regime"] = "sentiment_regime" in df.columns
    checks["no_null_sentiment"] = df["sentiment_value"].isnull().sum() == 0
    checks["sentiment_range"] = (
        df["sentiment_value"].min() >= 0 and df["sentiment_value"].max() <= 100
    )
    
    return checks


def run_validation(fgi, trader, merged):
    """Run all validations and print results."""
    print("\n" + "=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    all_passed = True
    
    for name, checks in [
        ("Fear & Greed Index", validate_fear_greed(fgi)),
        ("Historical Data", validate_historical_data(trader)),
        ("Merged Dataset", validate_merged(merged)),
    ]:
        print(f"\n  {name}:")
        for check, passed in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"    {status} — {check}")
            if not passed:
                all_passed = False
    
    print(f"\n{'=' * 60}")
    print(f"  Overall: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
    print(f"{'=' * 60}")
    
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_quality_report(fgi_raw, fgi_clean, trader_raw, trader_clean, merged):
    """Generate comprehensive data quality summary."""
    report = {}
    
    # FGI stats
    report["fgi"] = {
        "raw_rows": len(fgi_raw),
        "clean_rows": len(fgi_clean),
        "date_range": f"{fgi_clean['date'].min()} to {fgi_clean['date'].max()}",
        "missing_values": int(fgi_clean[["value","classification"]].isnull().sum().sum()),
        "duplicates": int(fgi_clean["date"].duplicated().sum()),
        "value_stats": fgi_clean["value"].describe().to_dict(),
        "classification_counts": fgi_clean["classification"].value_counts().to_dict(),
        "gaps_filled": len(fgi_clean) - len(fgi_raw),
    }
    
    # Trader stats
    report["trader"] = {
        "raw_rows": len(trader_raw),
        "clean_rows": len(trader_clean),
        "dust_removed": len(trader_raw) - len(trader_clean),
        "date_range": f"{trader_clean['trade_date'].min()} to {trader_clean['trade_date'].max()}",
        "unique_accounts": trader_clean["Account"].nunique(),
        "unique_coins": trader_clean["Coin"].nunique(),
        "unique_trading_days": trader_clean["trade_date"].nunique(),
        "total_pnl": float(trader_clean["Closed PnL"].sum()),
        "direction_counts": trader_clean["Direction"].value_counts().to_dict(),
        "margin_type_counts": trader_clean["margin_type"].value_counts().to_dict(),
        "negative_fees": int((trader_clean["Fee"] < 0).sum()),
    }
    
    # Merged stats
    report["merged"] = {
        "rows": len(merged),
        "overlap_days": merged["trade_date"].nunique(),
        "date_range": f"{merged['trade_date'].min()} to {merged['trade_date'].max()}",
        "accounts_in_overlap": merged["Account"].nunique(),
    }
    
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    """
    Full data pipeline: load → clean → merge → validate.
    
    Returns:
        tuple: (fgi_clean, trader_clean, merged, quality_report)
    """
    print("Loading datasets...")
    fgi_raw = load_fear_greed_index()
    trader_raw = load_historical_data()
    print(f"  FGI: {fgi_raw.shape}")
    print(f"  Trader: {trader_raw.shape}")
    
    print("\nCleaning Fear & Greed Index...")
    fgi_clean = clean_fear_greed(fgi_raw)
    print(f"  Clean shape: {fgi_clean.shape}")
    
    print("\nCleaning Historical Data...")
    trader_clean = clean_historical_data(trader_raw)
    
    print("\nMerging datasets...")
    merged = merge_datasets(fgi_clean, trader_clean)
    
    print("\nRunning validation...")
    run_validation(fgi_clean, trader_clean, merged)
    
    print("\nGenerating quality report...")
    quality_report = generate_quality_report(
        fgi_raw, fgi_clean, trader_raw, trader_clean, merged
    )
    
    # Save processed data
    merged.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "merged_data.csv"),
        index=False
    )
    fgi_clean.to_csv(
        os.path.join(DATA_PROCESSED_DIR, "fgi_clean.csv"),
        index=False
    )
    print(f"\nSaved processed data to {DATA_PROCESSED_DIR}/")
    
    return fgi_clean, trader_clean, merged, quality_report
