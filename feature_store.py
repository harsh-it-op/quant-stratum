"""
Feature Store for ML Regime Classifier
Deterministic pipeline: Data → SLOW → FAST → Normalize → Validate
Ensures reproducibility and prevents data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import sys

# Add features module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.slow_features import build_slow_features, get_slow_feature_names
from features.fast_features import build_fast_features, get_fast_feature_names
from features.normalization import normalize_features


class FeatureStore:
    """
    Feature engineering pipeline for regime detection.
    Ensures deterministic, reproducible feature generation.
    """
    
    def __init__(
        self,
        data_path: str = 'data/processed/market_data_historical.csv',
        output_dir: str = 'features',
        include_macro: bool = True,
        macro_csv_path: Optional[str] = None,
        refresh_macro: bool = False,
    ):
        """
        Initialize FeatureStore.

        Args:
            data_path: Path to raw market data
            output_dir: Directory for feature outputs
            include_macro: Whether to include macro fundamentals (USD/INR, US10Y,
                Brent, DXY) in the slow feature set.  Defaults to True.
            macro_csv_path: Path to cached macro time series; defaults to
                data/macro/raw_macro_series.csv next to data_path.
            refresh_macro: If True, refetch macro series from yfinance even if
                a cached CSV is available.
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.raw_data = None
        self.slow_features = None
        self.fast_features = None
        self.normalized_features = None
        self.final_features = None

        self.include_macro = include_macro
        if macro_csv_path is None:
            self.macro_csv_path = Path(data_path).resolve().parent.parent / "macro" / "raw_macro_series.csv"
        else:
            self.macro_csv_path = Path(macro_csv_path)
        self.refresh_macro = refresh_macro

        self.slow_feature_names = get_slow_feature_names(include_macro=include_macro)
        self.fast_feature_names = get_fast_feature_names()
    
    def load_data(self) -> pd.DataFrame:
        """Load raw market data."""
        print("="*60)
        print("STEP 1: Loading Data")
        print("="*60)
        
        self.raw_data = pd.read_csv(self.data_path)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])
        
        print(f"✓ Loaded {len(self.raw_data)} rows")
        print(f"  Columns: {list(self.raw_data.columns)}")
        print(f"  Date range: {self.raw_data['Date'].min()} to {self.raw_data['Date'].max()}")
        
        return self.raw_data
    
    def build_slow(self) -> pd.DataFrame:
        """Build SLOW features (long-term regime indicators)."""
        print("\n" + "="*60)
        print("STEP 2: Building SLOW Features (6M/12M windows)")
        print("="*60)
        
        self.slow_features = build_slow_features(
            self.raw_data,
            include_macro=self.include_macro,
            macro_csv_path=self.macro_csv_path,
            refresh_macro=self.refresh_macro,
        )
        
        # Save intermediate output
        slow_path = self.output_dir / 'slow_features_matrix.csv'
        self.slow_features.to_csv(slow_path, index=False)
        print(f"✓ Saved SLOW features to {slow_path}")
        
        return self.slow_features
    
    def build_fast(self) -> pd.DataFrame:
        """Build FAST features (short-term regime indicators)."""
        print("\n" + "="*60)
        print("STEP 3: Building FAST Features (1D-3M windows)")
        print("="*60)
        
        self.fast_features = build_fast_features(self.raw_data)
        
        # Save intermediate output
        fast_path = self.output_dir / 'fast_features_matrix.csv'
        self.fast_features.to_csv(fast_path, index=False)
        print(f"✓ Saved FAST features to {fast_path}")
        
        return self.fast_features
    
    def merge_features(self) -> pd.DataFrame:
        """Merge SLOW and FAST features."""
        print("\n" + "="*60)
        print("STEP 4: Merging Features")
        print("="*60)
        
        # Get base columns (Date, OHLCV, VIX)
        base_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'VIX_Close']
        
        # Merge on Date
        merged = self.slow_features[base_cols + self.slow_feature_names].merge(
            self.fast_features[['Date'] + self.fast_feature_names],
            on='Date',
            how='inner'
        )
        
        print(f"✓ Merged: {len(merged)} rows")
        print(f"  SLOW features: {len(self.slow_feature_names)}")
        print(f"  FAST features: {len(self.fast_feature_names)}")
        print(f"  Total features: {len(self.slow_feature_names) + len(self.fast_feature_names)}")
        
        return merged
    
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to features."""
        print("\n" + "="*60)
        print("STEP 5: Normalizing Features")
        print("="*60)
        
        self.normalized_features = normalize_features(
            df,
            slow_features=self.slow_feature_names,
            fast_features=self.fast_feature_names,
            slow_window=252,
            fast_window=60,
            clip_std=5.0
        )
        
        return self.normalized_features
    
    def validate(self) -> Dict:
        """Validate feature matrix."""
        print("\n" + "="*60)
        print("STEP 6: Validation")
        print("="*60)
        
        validation_results = {}
        
        # Check for NaN patterns
        all_norm_cols = [f"{col}_norm" for col in self.slow_feature_names + self.fast_feature_names]
        
        total_rows = len(self.normalized_features)
        nan_counts = self.normalized_features[all_norm_cols].isna().sum()
        
        print("\nNaN Analysis:")
        print(f"  Total rows: {total_rows}")
        print(f"  Max NaN count: {nan_counts.max()} ({100*nan_counts.max()/total_rows:.1f}%)")
        print(f"  Min NaN count: {nan_counts.min()} ({100*nan_counts.min()/total_rows:.1f}%)")
        
        # First valid row (after all rolling windows filled)
        first_valid_idx = self.normalized_features[all_norm_cols].dropna().index[0]
        first_valid_date = self.normalized_features.loc[first_valid_idx, 'Date']
        
        print(f"\nFirst complete row:")
        print(f"  Index: {first_valid_idx}")
        print(f"  Date: {first_valid_date}")
        print(f"  Rows dropped: {first_valid_idx} ({100*first_valid_idx/total_rows:.1f}%)")
        
        validation_results['total_rows'] = total_rows
        validation_results['first_valid_idx'] = first_valid_idx
        validation_results['first_valid_date'] = str(first_valid_date)
        validation_results['rows_with_complete_features'] = total_rows - first_valid_idx
        validation_results['dropout_pct'] = 100 * first_valid_idx / total_rows
        
        # Check for inf values
        inf_counts = np.isinf(self.normalized_features[all_norm_cols]).sum()
        print(f"\nInf values: {inf_counts.sum()}")
        
        validation_results['inf_count'] = int(inf_counts.sum())
        
        # Final usable dataset
        usable_rows = total_rows - first_valid_idx
        print(f"\n✓ Final usable dataset: {usable_rows} rows ({100*usable_rows/total_rows:.1f}%)")
        
        return validation_results
    
    def build_final_matrix(self) -> pd.DataFrame:
        """Create final feature matrix (drop NaN rows)."""
        print("\n" + "="*60)
        print("STEP 7: Building Final Matrix")
        print("="*60)
        
        # Get normalized feature columns
        all_norm_cols = [f"{col}_norm" for col in self.slow_feature_names + self.fast_feature_names]
        
        # Keep Date + normalized features only
        final_cols = ['Date'] + all_norm_cols
        
        # Drop rows with any NaN in normalized features
        self.final_features = self.normalized_features[final_cols].dropna()
        
        print(f"✓ Final matrix: {len(self.final_features)} rows × {len(final_cols)} columns")
        print(f"  Date range: {self.final_features['Date'].min()} to {self.final_features['Date'].max()}")
        print(f"  Features: {len(all_norm_cols)}")
        
        return self.final_features
    
    def save(self, output_path: Optional[str] = None) -> str:
        """Save final feature matrix."""
        if output_path is None:
            output_path = self.output_dir / 'final_features_matrix.csv'
        
        self.final_features.to_csv(output_path, index=False)
        print(f"\n✓ Saved final matrix to {output_path}")
        
        return str(output_path)
    
    def run(self) -> pd.DataFrame:
        """
        Run complete feature engineering pipeline.
        
        Returns:
            Final feature matrix ready for HMM training
        """
        print("="*60)
        print("FEATURE STORE PIPELINE")
        print("ML Regime Classifier - 2×3 HMM")
        print("="*60)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2-3: Build features
        self.build_slow()
        self.build_fast()
        
        # Step 4: Merge
        merged = self.merge_features()
        
        # Step 5: Normalize
        self.normalize(merged)
        
        # Step 6: Validate
        validation = self.validate()
        
        # Step 7: Build final matrix
        self.build_final_matrix()
        
        # Save
        self.save()
        
        print("\n" + "="*60)
        print("✓ PIPELINE COMPLETE")
        print("="*60)
        print(f"Output: features/final_features_matrix.csv")
        print(f"Rows: {len(self.final_features)}")
        print(f"Features: {len([c for c in self.final_features.columns if c != 'Date'])}")
        print(f"Dropout: {validation['dropout_pct']:.1f}%")
        
        return self.final_features


def main():
    """Run feature store pipeline from command line."""
    store = FeatureStore(
        data_path='data/processed/market_data_historical.csv',
        output_dir='features'
    )
    
    store.run()
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
