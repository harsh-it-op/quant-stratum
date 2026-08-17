"""
FAST Features for ML Regime Classifier
Designed for 2×3 HMM: Fast Layer (Calm/Choppy/Stress)
Window sizes: 1D, 5D, 20D (1 month), 60D (3 months)
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_returns(
    prices: pd.Series,
    periods: list = [1, 5, 20, 60]
) -> pd.DataFrame:
    """
    Compute returns over multiple horizons.
    
    Args:
        prices: Price series
        periods: List of lookback periods
        
    Returns:
        DataFrame with return columns
    """
    result = pd.DataFrame(index=prices.index)
    
    for p in periods:
        result[f'ret_{p}'] = np.log(prices / prices.shift(p))
    
    return result


def rolling_realized_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True
) -> pd.Series:
    """
    Rolling realized volatility.
    
    Args:
        returns: Log returns
        window: Rolling window size
        annualize: Whether to annualize (multiply by sqrt(252))
        
    Returns:
        Rolling volatility series
    """
    rv = returns.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv


def rolling_downside_volatility(
    returns: pd.Series,
    window: int = 20,
    threshold: float = 0.0,
    annualize: bool = True
) -> pd.Series:
    """
    Downside volatility (semi-deviation below threshold).
    
    Args:
        returns: Log returns
        window: Rolling window size
        threshold: Threshold return (default 0)
        annualize: Whether to annualize
        
    Returns:
        Downside volatility series
    """
    def downside_std(x):
        # LPM2 / Sortino-style semi-deviation: RMSE around the threshold (default 0),
        # NOT std around the empirical mean of the below-threshold subset. The latter
        # under-reports during persistent grind-down stress where below-zero returns
        # cluster together (low variance around their own mean) but are individually
        # large in magnitude.
        downside = x[x < threshold]
        if len(downside) < 2:
            return np.nan
        return float(np.sqrt(((downside - threshold) ** 2).mean()))

    dv = returns.rolling(window).apply(downside_std, raw=False)
    if annualize:
        dv = dv * np.sqrt(252)
    return dv


def rolling_vol_of_vol(
    returns: pd.Series,
    inner_window: int = 5,
    outer_window: int = 60
) -> pd.Series:
    """
    Volatility of volatility: std of rolling volatility.
    Captures short-term regime transitions.
    
    Args:
        returns: Log returns
        inner_window: Window for volatility calculation
        outer_window: Window for vol-of-vol
        
    Returns:
        Vol-of-vol series
    """
    # First compute rolling vol
    rv = returns.rolling(inner_window).std() * np.sqrt(252)
    
    # Then compute std of that vol
    vol_of_vol = rv.rolling(outer_window).std()
    
    return vol_of_vol


def rolling_skewness(
    returns: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Rolling skewness of returns.
    Negative skew = left tail risk (Stress regime indicator).
    
    Args:
        returns: Log returns
        window: Rolling window size
        
    Returns:
        Skewness series
    """
    return returns.rolling(window).skew()


def rolling_kurtosis(
    returns: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Rolling excess kurtosis of returns.
    High kurtosis = fat tails (Stress regime indicator).
    
    Args:
        returns: Log returns
        window: Rolling window size
        
    Returns:
        Excess kurtosis series
    """
    return returns.rolling(window).kurt()


def high_low_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series
) -> pd.Series:
    """
    High-Low range as fraction of close.
    Measures intraday volatility.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        
    Returns:
        HL range series
    """
    return (high - low) / close


def gap_measure(
    open_price: pd.Series,
    close_prev: pd.Series
) -> pd.Series:
    """
    Overnight gap as fraction of previous close.
    
    Args:
        open_price: Open prices
        close_prev: Previous close prices
        
    Returns:
        Gap series
    """
    return (open_price - close_prev) / close_prev


def volume_z_score(
    volume: pd.Series,
    window: int = 60
) -> pd.Series:
    """
    Z-score of volume (standardized).
    High values indicate unusual trading activity.
    
    Args:
        volume: Volume series
        window: Rolling window for mean/std
        
    Returns:
        Volume z-score series
    """
    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std()
    
    return (volume - vol_mean) / vol_std


def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    prices: pd.Series | None = None,
    window: int = 20,
) -> pd.Series:
    """
    Amihud (2002) illiquidity measure: |return| / dollar_volume, rolling-averaged.

    Dollar volume (price * volume) keeps the denominator scale-stationary across
    the 15-year NIFTY 500 history (index level ~5x'd over that period). Using
    raw share volume would let the level effect leak into the raw series; the
    60D rolling z-score normalization downstream removes most of that drift but
    using the textbook dollar-volume form is cheap and avoids the failure mode
    entirely.

    Args:
        returns: Log returns
        volume: Volume series (shares/contracts traded)
        prices: Close-price series. If None, falls back to raw-volume form
            (legacy behavior — only for callers that don't pass a price).
        window: Rolling window
    """
    if prices is not None:
        denom = (prices * volume) + 1.0
    else:
        denom = volume + 1
    illiq = np.abs(returns) / denom
    return illiq.rolling(window).mean()


def vix_level_features(
    vix: pd.Series
) -> pd.DataFrame:
    """
    VIX-based features for FAST regimes.
    
    Args:
        vix: VIX closing prices
        
    Returns:
        DataFrame with VIX features
    """
    result = pd.DataFrame(index=vix.index)
    
    # VIX level (raw)
    result['vix_level'] = vix
    
    # VIX change (1-day)
    result['vix_change'] = vix.pct_change()
    
    # VIX percentile (60D) — fraction of the 60D window <= today's VIX.
    # Higher VIX -> closer to 1.0 (rare/elevated); lower VIX -> closer to 0.0.
    result['vix_percentile_60'] = vix.rolling(60).apply(
        lambda x: (x.iloc[-1] >= x).mean() if len(x) > 0 else np.nan
    )
    
    # VIX shock (2-std move in 5 days)
    vix_ma = vix.rolling(5).mean()
    vix_std = vix.rolling(60).std()
    result['vix_shock'] = (vix - vix_ma) / vix_std
    
    # VIX momentum (20D change)
    result['vix_momentum_20'] = (vix - vix.shift(20)) / vix.shift(20)
    
    return result


def build_fast_features(
    df: pd.DataFrame,
    price_col: str = 'Close',
    open_col: str = 'Open',
    high_col: str = 'High',
    low_col: str = 'Low',
    volume_col: str = 'Volume',
    vix_col: str = 'VIX_Close'
) -> pd.DataFrame:
    """
    Build all FAST features for regime detection.
    
    Args:
        df: DataFrame with OHLCV and VIX data
        price_col: Column name for closing price
        open_col: Column name for open price
        high_col: Column name for high price
        low_col: Column name for low price
        volume_col: Column name for volume
        vix_col: Column name for VIX
        
    Returns:
        DataFrame with FAST features
    """
    result = df.copy()
    
    # Compute log returns
    result['log_ret'] = np.log(result[price_col] / result[price_col].shift(1))
    
    print("Computing FAST features...")
    
    # Returns over multiple horizons
    returns_df = compute_returns(result[price_col], periods=[1, 5, 20, 60])
    result = pd.concat([result, returns_df], axis=1)
    
    # Volatility features
    result['rv_20'] = rolling_realized_volatility(result['log_ret'], window=20)
    result['rv_60'] = rolling_realized_volatility(result['log_ret'], window=60)
    result['downside_vol_20'] = rolling_downside_volatility(result['log_ret'], window=20)
    result['downside_vol_60'] = rolling_downside_volatility(result['log_ret'], window=60)
    result['vol_of_vol_60'] = rolling_vol_of_vol(result['log_ret'], inner_window=5, outer_window=60)
    
    # Distribution features
    result['skewness_60'] = rolling_skewness(result['log_ret'], window=60)
    result['kurtosis_60'] = rolling_kurtosis(result['log_ret'], window=60)
    
    # Intraday features
    result['hl_range'] = high_low_range(result[high_col], result[low_col], result[price_col])
    result['gap'] = gap_measure(result[open_col], result[price_col].shift(1))
    
    # Volume features
    result['volume_z_60'] = volume_z_score(result[volume_col], window=60)
    result['amihud_20'] = amihud_illiquidity(
        result['log_ret'], result[volume_col], prices=result[price_col], window=20
    )
    
    # VIX features
    vix_features = vix_level_features(result[vix_col])
    result = pd.concat([result, vix_features], axis=1)
    
    print(f"✓ Created {len([c for c in result.columns if c not in df.columns])} FAST features")
    print(f"  Returns: 4 (1D, 5D, 20D, 60D)")
    print(f"  Volatility: 5 (rv_20/60, downside_20/60, vol_of_vol_60)")
    print(f"  Distribution: 2 (skewness, kurtosis)")
    print(f"  Intraday: 2 (HL range, gap)")
    print(f"  Volume: 2 (z-score, Amihud)")
    print(f"  VIX: 5 (level, change, percentile, shock, momentum)")
    
    return result


def get_fast_feature_names() -> list:
    """Return list of FAST feature column names."""
    return [
        'ret_1', 'ret_5', 'ret_20', 'ret_60',
        'rv_20', 'rv_60', 'downside_vol_20', 'downside_vol_60', 'vol_of_vol_60',
        'skewness_60', 'kurtosis_60',
        'hl_range', 'gap',
        'volume_z_60', 'amihud_20',
        'vix_level', 'vix_change', 'vix_percentile_60', 'vix_shock', 'vix_momentum_20'
    ]


if __name__ == "__main__":
    # Test on historical data
    print("="*60)
    print("FAST Features Module - Test")
    print("="*60)
    
    df = pd.read_csv('data/processed/market_data_historical.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Input: {len(df)} rows")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Build FAST features
    result = build_fast_features(df)
    
    # Check NaN counts
    fast_cols = get_fast_feature_names()
    print("\nNaN counts:")
    for col in fast_cols:
        nan_count = result[col].isna().sum()
        nan_pct = 100 * nan_count / len(result)
        print(f"  {col}: {nan_count} ({nan_pct:.1f}%)")
    
    # Show sample
    print("\nSample features (last 5 rows):")
    print(result[['Date'] + fast_cols].tail())
    
    # Save output
    output_path = 'features/fast_features_matrix.csv'
    result.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
