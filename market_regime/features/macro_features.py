"""
Macro Fundamentals features for the SLOW axis.

These features add truly orthogonal (non-price) information to the slow HMM:
  - USD/INR (INR=X)      -- EM FX stress proxy
  - US 10Y yield (^TNX)  -- global risk-free benchmark
  - Brent crude (BZ=F)   -- India oil-import sensitivity
  - DXY (DX-Y.NYB)       -- global USD liquidity

In the offline proof-of-concept (scripts/q1_macro_fundamentals.py) adding these
doubled the slow-axis Sharpe spread (0.46 -> 0.99) on the corrected forward-
return-aligned OOS evaluation, so they are now part of the production slow
feature set.

The features are RAW (un-normalized) and rely on the standard rolling z-score
normalization in features/normalization.py (window=252, clip=5 sigma) to
deliver the final '_norm' columns the HMM consumes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MACRO_TICKERS = {
    "USD_INR": "INR=X",
    "US10Y":   "^TNX",
    "BRENT":   "BZ=F",
    "DXY":     "DX-Y.NYB",
}


def get_macro_feature_names() -> list:
    """Return raw (pre-normalization) macro feature column names."""
    return [
        "usdinr_pct_252",
        "usdinr_mom_60",
        "usdinr_vol_60",
        "us10y_change_60",
        "us10y_vol_60",
        "brent_ret_60",
        "brent_vol_60",
        "dxy_ret_60",
    ]


def fetch_macro_series(start: str, end: str) -> pd.DataFrame:
    """Fetch macro tickers from yfinance.  Returns DataFrame indexed by Date."""
    import yfinance as yf
    frames = {}
    for name, ticker in MACRO_TICKERS.items():
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        frames[name] = df["Close"]
    out = pd.DataFrame(frames).ffill().dropna(how="all")
    out.index.name = "Date"
    return out


def load_macro_series(csv_path: Path) -> pd.DataFrame:
    """Load cached macro time series from CSV (Date index + ticker columns)."""
    df = pd.read_csv(csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def compute_macro_features(macros_df: pd.DataFrame) -> pd.DataFrame:
    """Compute raw macro features from a DataFrame of macro price series."""
    out = pd.DataFrame(index=macros_df.index)

    if "USD_INR" in macros_df.columns:
        s = macros_df["USD_INR"]
        out["usdinr_pct_252"] = s.rolling(252).rank(pct=True)
        out["usdinr_mom_60"]  = (s / s.shift(60) - 1.0)
        out["usdinr_vol_60"]  = np.log(s / s.shift(1)).rolling(60).std() * np.sqrt(252)

    if "US10Y" in macros_df.columns:
        y = macros_df["US10Y"]
        out["us10y_change_60"] = y - y.shift(60)
        out["us10y_vol_60"]    = y.diff().rolling(60).std()

    if "BRENT" in macros_df.columns:
        b = macros_df["BRENT"]
        out["brent_ret_60"] = (b / b.shift(60) - 1.0)
        out["brent_vol_60"] = np.log(b / b.shift(1)).rolling(60).std() * np.sqrt(252)

    if "DXY" in macros_df.columns:
        d = macros_df["DXY"]
        out["dxy_ret_60"] = (d / d.shift(60) - 1.0)

    return out


def build_macro_features(
    market_dates: pd.DatetimeIndex,
    macro_csv_path: Optional[Path] = None,
    refresh: bool = False,
    period_start: str = "2010-01-01",
    period_end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build macro features aligned to a list of market trading dates.

    Args:
        market_dates: Trading days from the primary market (e.g. NIFTY 500).
        macro_csv_path: Cached macro price series; if missing or refresh=True,
            fetches from yfinance and (over)writes the cache.
        refresh: Force fresh fetch even if cache exists.
        period_start / period_end: yfinance fetch range when fetching.

    Returns:
        DataFrame indexed by Date containing the 8 raw macro features.
    """
    market_dates = pd.DatetimeIndex(pd.to_datetime(market_dates))
    if period_end is None:
        end_dt = market_dates.max() + pd.Timedelta(days=5)
        period_end = end_dt.strftime("%Y-%m-%d")

    macros = None
    if macro_csv_path is not None and Path(macro_csv_path).exists() and not refresh:
        macros = load_macro_series(Path(macro_csv_path))

    if macros is None or macros.empty:
        macros = fetch_macro_series(period_start, period_end)
        if macro_csv_path is not None:
            Path(macro_csv_path).parent.mkdir(parents=True, exist_ok=True)
            macros.to_csv(macro_csv_path)

    # Compute features on each instrument's NATIVE trading calendar — computing
    # returns/vols on an ffill'd NSE-aligned series would inject zero-return days
    # on non-trading days of the macro instrument and bias rolling vol downward.
    # Forward-fill the computed features (not the raw prices) onto NSE days.
    macros = macros[~macros.index.duplicated(keep="last")].sort_index()
    # To prevent timezone look-ahead bias (US markets close 10 hours after NSE closes),
    # we must shift the native macro features by 1 day BEFORE aligning to the NSE calendar.
    # Otherwise, the NSE regime for Day T would theoretically use the US close for Day T,
    # which hasn't happened yet when the NSE market closes at 15:30 IST.
    feats_native = compute_macro_features(macros).shift(1)
    feats = feats_native.reindex(feats_native.index.union(market_dates)).ffill().reindex(market_dates)
    return feats
