"""
Feature Normalization for ML Regime Classifier
Applies rolling z-score normalization to ensure stationarity
SLOW features: 252D window (12 months)
FAST features: 60D window (3 months)
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def rolling_z_score(
    series: pd.Series,
    window: int,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Compute rolling z-score normalization.
    
    Formula: z = (x - μ) / σ
    where μ and σ are computed over rolling window.
    
    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum observations for valid result
        
    Returns:
        Z-score normalized series
    """
    if min_periods is None:
        min_periods = window
    
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    z_score = (series - rolling_mean) / rolling_std
    
    return z_score


def normalize_slow_features(
    df: pd.DataFrame,
    slow_features: List[str],
    window: int = 252,
    suffix: str = '_norm'
) -> pd.DataFrame:
    """
    Normalize SLOW features using 252D rolling window.
    
    Args:
        df: DataFrame with SLOW features
        slow_features: List of SLOW feature column names
        window: Rolling window size (default 252D = 1 year)
        suffix: Suffix to add to normalized column names
        
    Returns:
        DataFrame with normalized SLOW features
    """
    result = df.copy()
    
    print(f"Normalizing {len(slow_features)} SLOW features (window={window}D)...")
    
    for col in slow_features:
        if col not in df.columns:
            print(f"  Warning: {col} not found in DataFrame")
            continue
        
        norm_col = f"{col}{suffix}"
        result[norm_col] = rolling_z_score(df[col], window=window)
    
    print(f"✓ Created {len(slow_features)} normalized SLOW features")
    
    return result


def normalize_fast_features(
    df: pd.DataFrame,
    fast_features: List[str],
    window: int = 60,
    suffix: str = '_norm'
) -> pd.DataFrame:
    """
    Normalize FAST features using 60D rolling window.
    
    Args:
        df: DataFrame with FAST features
        fast_features: List of FAST feature column names
        window: Rolling window size (default 60D = 3 months)
        suffix: Suffix to add to normalized column names
        
    Returns:
        DataFrame with normalized FAST features
    """
    result = df.copy()
    
    print(f"Normalizing {len(fast_features)} FAST features (window={window}D)...")
    
    for col in fast_features:
        if col not in df.columns:
            print(f"  Warning: {col} not found in DataFrame")
            continue
        
        norm_col = f"{col}{suffix}"
        result[norm_col] = rolling_z_score(df[col], window=window)
    
    print(f"✓ Created {len(fast_features)} normalized FAST features")
    
    return result


def clip_outliers(
    df: pd.DataFrame,
    columns: List[str],
    n_std: float = 5.0
) -> pd.DataFrame:
    """
    Clip extreme outliers beyond n standard deviations.
    Applied AFTER normalization to handle extreme events.
    
    Args:
        df: DataFrame with normalized features
        columns: Columns to clip
        n_std: Number of standard deviations for clipping threshold
        
    Returns:
        DataFrame with clipped values
    """
    result = df.copy()
    
    print(f"Clipping outliers beyond {n_std} std for {len(columns)} features...")
    
    clipped_count = 0
    for col in columns:
        if col not in df.columns:
            continue
        
        # Count values beyond threshold
        outliers = (np.abs(result[col]) > n_std).sum()
        if outliers > 0:
            clipped_count += outliers
            # Clip to [-n_std, +n_std]
            result[col] = result[col].clip(lower=-n_std, upper=n_std)
    
    print(f"✓ Clipped {clipped_count} outlier values")
    
    return result


def check_stationarity(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Check stationarity of normalized features.
    Reports mean, std, min, max for each feature.
    
    Args:
        df: DataFrame with normalized features
        columns: Columns to check
        
    Returns:
        Summary DataFrame
    """
    summary = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        valid_data = df[col].dropna()
        
        if len(valid_data) == 0:
            continue
        
        summary.append({
            'feature': col,
            'mean': valid_data.mean(),
            'std': valid_data.std(),
            'min': valid_data.min(),
            'max': valid_data.max(),
            'nan_pct': 100 * df[col].isna().sum() / len(df)
        })
    
    return pd.DataFrame(summary)


def normalize_features(
    df: pd.DataFrame,
    slow_features: List[str],
    fast_features: List[str],
    slow_window: int = 252,
    fast_window: int = 60,
    clip_std: float = 5.0
) -> pd.DataFrame:
    """
    Complete normalization pipeline for SLOW and FAST features.
    
    Args:
        df: DataFrame with raw features
        slow_features: List of SLOW feature names
        fast_features: List of FAST feature names
        slow_window: Window for SLOW normalization (default 252D)
        fast_window: Window for FAST normalization (default 60D)
        clip_std: Standard deviations for outlier clipping
        
    Returns:
        DataFrame with normalized features
    """
    print("="*60)
    print("Feature Normalization Pipeline")
    print("="*60)
    
    result = df.copy()
    
    # Normalize SLOW features
    result = normalize_slow_features(result, slow_features, window=slow_window)
    
    # Normalize FAST features
    result = normalize_fast_features(result, fast_features, window=fast_window)
    
    # Get normalized column names
    slow_norm_cols = [f"{col}_norm" for col in slow_features]
    fast_norm_cols = [f"{col}_norm" for col in fast_features]
    all_norm_cols = slow_norm_cols + fast_norm_cols
    
    # Clip outliers
    result = clip_outliers(result, all_norm_cols, n_std=clip_std)
    
    # Check stationarity
    print("\nStationarity check (normalized features):")
    summary = check_stationarity(result, all_norm_cols)
    print(summary.to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ Normalization complete")
    print("="*60)
    
    return result


if __name__ == "__main__":
    # Test normalization pipeline
    print("="*60)
    print("Normalization Module - Test")
    print("="*60)
    
    # Load features (assumes slow_features.py has been run)
    try:
        df = pd.read_csv('features/slow_features_matrix.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"✓ Loaded SLOW features: {len(df)} rows")
    except FileNotFoundError:
        print("✗ Run slow_features.py first")
        exit(1)
    
    # Load fast features
    try:
        fast_df = pd.read_csv('features/fast_features_matrix.csv')
        fast_df['Date'] = pd.to_datetime(fast_df['Date'])
        print(f"✓ Loaded FAST features: {len(fast_df)} rows")
        
        # Merge (should be identical dates)
        df = df.merge(fast_df[['Date'] + [c for c in fast_df.columns if c.startswith(('ret_', 'rv_', 'vix_', 'hl_', 'gap', 'volume_', 'amihud', 'downside', 'skewness_60', 'kurtosis_60'))]], on='Date', how='inner')
        print(f"✓ Merged: {len(df)} rows")
    except FileNotFoundError:
        print("✗ Run fast_features.py first")
        exit(1)
    
    # Define feature lists
    from features.slow_features import get_slow_feature_names
    from features.fast_features import get_fast_feature_names
    
    slow_features = get_slow_feature_names()
    fast_features = get_fast_feature_names()
    
    # Normalize
    result = normalize_features(
        df,
        slow_features=slow_features,
        fast_features=fast_features,
        slow_window=252,
        fast_window=60,
        clip_std=5.0
    )
    
    # Save output
    output_path = 'features/normalized_features_matrix.csv'
    result.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
