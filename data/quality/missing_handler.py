"""
Missing Data Handler for Market Time Series
Forward-fill missing values with safety limits
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def analyze_missing_data(df: pd.DataFrame) -> dict:
    """
    Analyze missing data patterns in DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary with missing data statistics
    """
    total_rows = len(df)
    missing_stats = {}
    
    for col in df.columns:
        if col == 'Date':
            continue
        
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        if missing_count > 0:
            missing_stats[col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2),
                'first_missing': df[df[col].isnull()]['Date'].min() if 'Date' in df.columns else None,
                'last_missing': df[df[col].isnull()]['Date'].max() if 'Date' in df.columns else None
            }
    
    return missing_stats


def find_consecutive_missing(series: pd.Series) -> pd.DataFrame:
    """
    Find runs of consecutive missing values.
    
    Args:
        series: Pandas Series to analyze
        
    Returns:
        DataFrame with consecutive missing runs
    """
    is_missing = series.isnull()
    
    # Find starts and ends of missing runs
    missing_starts = is_missing & ~is_missing.shift(1, fill_value=False)
    missing_ends = is_missing & ~is_missing.shift(-1, fill_value=False)
    
    starts = series.index[missing_starts].tolist()
    ends = series.index[missing_ends].tolist()
    
    if len(starts) != len(ends):
        # Handle edge case where series ends with missing values
        if len(starts) > len(ends):
            ends.append(series.index[-1])
    
    runs = []
    for start, end in zip(starts, ends):
        length = end - start + 1
        runs.append({'start_idx': start, 'end_idx': end, 'length': length})
    
    return pd.DataFrame(runs)


def forward_fill_with_limit(
    df: pd.DataFrame,
    max_consecutive: int = 3,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Forward-fill missing values with a maximum consecutive limit.
    
    Args:
        df: DataFrame to fill
        max_consecutive: Maximum consecutive days to forward-fill
        columns: Specific columns to fill (None = all numeric columns)
        
    Returns:
        DataFrame with filled values
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            # Forward fill with limit
            df[col] = df[col].fillna(method='ffill', limit=max_consecutive)
    
    return df


def handle_missing_data(
    df: pd.DataFrame,
    max_ffill: int = 3,
    drop_threshold: float = 0.05,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Comprehensive missing data handling pipeline.
    
    Args:
        df: DataFrame with potential missing values
        max_ffill: Maximum consecutive forward-fills allowed
        drop_threshold: Drop column if missing % exceeds this
        output_path: Optional path to save cleaned data
        
    Returns:
        Cleaned DataFrame
    """
    print("Analyzing missing data...")
    print("="*60)
    
    # Initial analysis
    missing_stats = analyze_missing_data(df)
    
    if not missing_stats:
        print("✓ No missing data found!")
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✓ Saved to {output_path}")
        return df
    
    print(f"Found missing data in {len(missing_stats)} columns:")
    for col, stats in missing_stats.items():
        print(f"  {col}: {stats['count']} missing ({stats['percentage']}%)")
    
    # Check for columns with excessive missing data
    drop_cols = []
    for col, stats in missing_stats.items():
        if stats['percentage'] > drop_threshold * 100:
            drop_cols.append(col)
            print(f"⚠ {col} has {stats['percentage']}% missing (above {drop_threshold*100}% threshold)")
    
    if drop_cols:
        print(f"\nDropping columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    
    # Forward-fill with limit
    print(f"\nForward-filling with max_consecutive={max_ffill}...")
    df = forward_fill_with_limit(df, max_consecutive=max_ffill)
    
    # Check remaining missing
    remaining_stats = analyze_missing_data(df)
    
    if remaining_stats:
        print(f"\n⚠ {len(remaining_stats)} columns still have missing data after ffill:")
        for col, stats in remaining_stats.items():
            print(f"  {col}: {stats['count']} missing ({stats['percentage']}%)")
            
            # Find consecutive missing runs
            runs = find_consecutive_missing(df[col])
            if len(runs) > 0:
                max_run = runs['length'].max()
                print(f"    → Longest consecutive run: {max_run} days")
        
        # Drop rows with any remaining missing values
        rows_before = len(df)
        df = df.dropna()
        rows_after = len(df)
        rows_dropped = rows_before - rows_after
        
        if rows_dropped > 0:
            print(f"\n✓ Dropped {rows_dropped} rows with remaining missing values")
    else:
        print("\n✓ All missing values filled successfully!")
    
    print("="*60)
    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"✓ Saved cleaned data to {output_path}")
    
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Handle missing data in market time series")
    parser.add_argument(
        "--input",
        default="data/processed/market_data_historical.csv",
        help="Input CSV file"
    )
    parser.add_argument(
        "--output",
        default="data/processed/market_data_cleaned.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--max-ffill",
        type=int,
        default=3,
        help="Maximum consecutive forward-fills (default: 3)"
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.05,
        help="Drop column if missing % exceeds this (default: 0.05 = 5%%)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    # Handle missing data
    cleaned_df = handle_missing_data(
        df,
        max_ffill=args.max_ffill,
        drop_threshold=args.drop_threshold,
        output_path=args.output
    )
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
