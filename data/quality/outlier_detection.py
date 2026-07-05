"""
Outlier Detection for Market Data
Identifies anomalous returns, volume spikes, and data quality issues
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def detect_return_outliers(
    df: pd.DataFrame,
    threshold: float = 5.0,
    window: int = 252
) -> pd.DataFrame:
    """
    Detect extreme returns using rolling z-score method.
    
    Args:
        df: DataFrame with Close price
        threshold: Z-score threshold (default: 5 = extreme outliers)
        window: Rolling window for mean/std calculation
        
    Returns:
        DataFrame with outlier flags
    """
    returns = df['Close'].pct_change()
    
    # Rolling z-score
    rolling_mean = returns.rolling(window=window, min_periods=20).mean()
    rolling_std = returns.rolling(window=window, min_periods=20).std()
    z_score = (returns - rolling_mean) / rolling_std
    
    # Flag outliers
    outliers = abs(z_score) > threshold
    
    return pd.DataFrame({
        'Date': df['Date'],
        'return': returns,
        'z_score': z_score,
        'is_outlier': outliers,
        'severity': abs(z_score)
    })


def detect_volume_spikes(
    df: pd.DataFrame,
    threshold: float = 10.0,
    window: int = 60
) -> pd.DataFrame:
    """
    Detect unusual volume spikes.
    
    Args:
        df: DataFrame with Volume
        threshold: Multiple of average volume to flag
        window: Rolling window for average calculation
        
    Returns:
        DataFrame with volume spike flags
    """
    volume = df['Volume']
    
    # Rolling average volume
    avg_volume = volume.rolling(window=window, min_periods=20).mean()
    volume_ratio = volume / avg_volume
    
    # Flag spikes
    spikes = volume_ratio > threshold
    
    return pd.DataFrame({
        'Date': df['Date'],
        'volume': volume,
        'avg_volume': avg_volume,
        'volume_ratio': volume_ratio,
        'is_spike': spikes
    })


def detect_price_gaps(
    df: pd.DataFrame,
    threshold: float = 0.10
) -> pd.DataFrame:
    """
    Detect large overnight price gaps.
    
    Args:
        df: DataFrame with Open and Close prices
        threshold: Gap percentage to flag (default: 10%)
        
    Returns:
        DataFrame with gap flags
    """
    prev_close = df['Close'].shift(1)
    gap = (df['Open'] - prev_close) / prev_close
    
    # Flag large gaps
    large_gaps = abs(gap) > threshold
    
    return pd.DataFrame({
        'Date': df['Date'],
        'prev_close': prev_close,
        'open': df['Open'],
        'gap_pct': gap * 100,
        'is_large_gap': large_gaps
    })


def detect_stuck_prices(
    df: pd.DataFrame,
    max_consecutive: int = 3
) -> pd.DataFrame:
    """
    Detect periods where prices are stuck (identical closes).
    
    Args:
        df: DataFrame with Close prices
        max_consecutive: Max allowed consecutive identical prices
        
    Returns:
        DataFrame with stuck price flags
    """
    close = df['Close']
    
    # Count consecutive identical prices
    price_change = close.diff()
    is_stuck = price_change == 0
    
    # Find runs of stuck prices
    consecutive_count = is_stuck.groupby((~is_stuck).cumsum()).cumsum()
    
    # Flag if exceeds threshold
    flagged = consecutive_count > max_consecutive
    
    return pd.DataFrame({
        'Date': df['Date'],
        'close': close,
        'price_unchanged': is_stuck,
        'consecutive_days': consecutive_count,
        'is_stuck': flagged
    })


def generate_outlier_report(
    df: pd.DataFrame,
    output_path: str = None
) -> Dict:
    """
    Generate comprehensive outlier detection report.
    
    Args:
        df: DataFrame with OHLCV data
        output_path: Optional path to save report
        
    Returns:
        Dictionary with all outlier analyses
    """
    print("Running outlier detection...")
    print("="*60)
    
    # Detect various anomalies
    return_outliers = detect_return_outliers(df)
    volume_spikes = detect_volume_spikes(df)
    price_gaps = detect_price_gaps(df)
    stuck_prices = detect_stuck_prices(df)
    
    # Count issues
    n_return_outliers = return_outliers['is_outlier'].sum()
    n_volume_spikes = volume_spikes['is_spike'].sum()
    n_large_gaps = price_gaps['is_large_gap'].sum()
    n_stuck_prices = stuck_prices['is_stuck'].sum()
    
    print(f"Return outliers (|z| > 5): {n_return_outliers}")
    print(f"Volume spikes (>10x avg): {n_volume_spikes}")
    print(f"Large gaps (>10%): {n_large_gaps}")
    print(f"Stuck prices (>3 days): {n_stuck_prices}")
    print("="*60)
    
    # Show worst cases
    if n_return_outliers > 0:
        print("\nTop 5 return outliers:")
        top_returns = return_outliers.nlargest(5, 'severity')[['Date', 'return', 'z_score']]
        print(top_returns.to_string(index=False))
    
    if n_volume_spikes > 0:
        print("\nTop 5 volume spikes:")
        top_volumes = volume_spikes.nlargest(5, 'volume_ratio')[['Date', 'volume', 'volume_ratio']]
        print(top_volumes.to_string(index=False))
    
    if n_large_gaps > 0:
        print("\nTop 5 price gaps:")
        top_gaps = (
            price_gaps.assign(abs_gap=price_gaps['gap_pct'].abs())
            .nlargest(5, 'abs_gap')[['Date', 'gap_pct']]
        )
        print(top_gaps.to_string(index=False))
    
    report = {
        'summary': {
            'total_rows': len(df),
            'return_outliers': int(n_return_outliers),
            'volume_spikes': int(n_volume_spikes),
            'large_gaps': int(n_large_gaps),
            'stuck_prices': int(n_stuck_prices)
        },
        'return_outliers': return_outliers,
        'volume_spikes': volume_spikes,
        'price_gaps': price_gaps,
        'stuck_prices': stuck_prices
    }
    
    if output_path:
        # Save summary
        summary_df = pd.DataFrame([report['summary']])
        summary_df.to_csv(output_path.replace('.csv', '_summary.csv'), index=False)
        
        # Save detailed outliers
        if n_return_outliers > 0:
            flagged = return_outliers[return_outliers['is_outlier']]
            flagged.to_csv(output_path.replace('.csv', '_returns.csv'), index=False)
        
        print(f"\n✓ Report saved to {output_path}")
    
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect outliers in market data")
    parser.add_argument(
        "--input",
        default="data/processed/market_data_historical.csv",
        help="Input CSV file with market data"
    )
    parser.add_argument(
        "--output",
        default="data/quality/outlier_report.csv",
        help="Output path for outlier report"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Z-score threshold for return outliers (default: 5.0)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Generate report
    report = generate_outlier_report(df, args.output)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
