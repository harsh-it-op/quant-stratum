"""
NSE Data Fetcher for ML Regime Classifier
Fetches NIFTY 50, India VIX, and related market data from Yahoo Finance
Targets: 15+ years of historical data
"""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf


def fetch_nifty_data(
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch NIFTY 50 OHLCV data from Yahoo Finance.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with NIFTY OHLCV data
    """
    print(f"Fetching NIFTY 50 data from {start_date} to {end_date}...")
    
    # Yahoo Finance ticker for NIFTY 500 (CRSLDX tracks Nifty 500 effectively on YF)
    ticker = "^CRSLDX"
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError("No data returned from Yahoo Finance")
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    
    # Rename columns to match our schema
    df = df.rename(columns={
        'Date': 'Date',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume',
        'Adj Close': 'Adj_Close'
    })
    
    # Keep only needed columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"[OK] Fetched {len(df)} days of NIFTY data")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved to {output_path}")
    
    return df


def fetch_india_vix(
    start_date: str,
    end_date: str,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch India VIX data from Yahoo Finance.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with India VIX data
    """
    print(f"Fetching India VIX data from {start_date} to {end_date}...")
    
    # Yahoo Finance ticker for India VIX
    ticker = "^INDIAVIX"
    
    # Download data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        raise ValueError("No VIX data returned from Yahoo Finance")
    
    # Reset index
    df = df.reset_index()
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if col[1] == '' else col[0] for col in df.columns]
    
    # Keep only Date and Close
    df = df[['Date', 'Close']].rename(columns={'Close': 'VIX_Close'})
    
    print(f"[OK] Fetched {len(df)} days of India VIX data")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[OK] Saved to {output_path}")
    
    return df


def merge_nifty_vix(
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Merge NIFTY and VIX data on Date.
    
    Args:
        nifty_df: DataFrame with NIFTY data
        vix_df: DataFrame with VIX data
        output_path: Optional path to save merged CSV
        
    Returns:
        Merged DataFrame
    """
    print("Merging NIFTY and VIX data...")
    
    # Convert Date to datetime if not already
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    vix_df['Date'] = pd.to_datetime(vix_df['Date'])
    
    # Merge on Date (inner join to keep only dates with both)
    merged = pd.merge(nifty_df, vix_df, on='Date', how='inner')
    
    # Sort by date
    merged = merged.sort_values('Date').reset_index(drop=True)
    
    print(f"[OK] Merged dataset: {len(merged)} rows")
    print(f"  Date range: {merged['Date'].min()} to {merged['Date'].max()}")
    print(f"  Missing NIFTY: {len(nifty_df) - len(merged)} days")
    print(f"  Missing VIX: {len(vix_df) - len(merged)} days")
    
    if output_path:
        merged.to_csv(output_path, index=False)
        print(f"[OK] Saved merged data to {output_path}")
    
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch NIFTY 50 and India VIX data for regime detection"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=15,
        help="Number of years of historical data to fetch (default: 15)"
    )
    parser.add_argument(
        "--output",
        default="../processed/market_data_historical.csv",
        help="Output path for merged data"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD), overrides --years"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today"
    )
    
    args = parser.parse_args()
    
    # Some environments set a dead local proxy (127.0.0.1:9) that breaks outbound fetches.
    for proxy_key in ['HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy']:
        val = os.environ.get(proxy_key)
        if val and '127.0.0.1:9' in val:
            os.environ.pop(proxy_key, None)

    # Ensure yfinance uses a writable local cache path.
    cache_root = (Path(__file__).resolve().parent / '.cache' / 'yfinance').resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_root))
    if hasattr(yf, 'set_cache_location'):
        try:
            yf.set_cache_location(str(cache_root))
        except Exception:
            pass

    # Calculate date range
    end_date = args.end_date or datetime.now().strftime('%Y-%m-%d')
    
    if args.start_date:
        start_date = args.start_date
    else:
        start_dt = datetime.now() - timedelta(days=args.years * 365 + 30)  # +30 for safety
        start_date = start_dt.strftime('%Y-%m-%d')
    
    print("="*60)
    print("NSE Data Fetcher - ML Regime Classifier")
    print("="*60)
    print(f"Target: {args.years} years of data")
    print(f"Date range: {start_date} to {end_date}")
    print("="*60)
    
    try:
        # Fetch NIFTY data
        nifty_df = fetch_nifty_data(start_date, end_date)
        
        # Fetch VIX data
        vix_df = fetch_india_vix(start_date, end_date)
        
        # Merge datasets
        merged_df = merge_nifty_vix(nifty_df, vix_df, args.output)
        
        print("\n" + "="*60)
        print("[OK] Data fetch complete!")
        print("="*60)
        print(f"Total rows: {len(merged_df)}")
        print(f"Columns: {list(merged_df.columns)}")
        print(f"\nSample data:")
        print(merged_df.head(3))
        print("...")
        print(merged_df.tail(3))
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
