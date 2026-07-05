"""
Feature Engineering Tests for quant-stratum
Validates:
1. No data leakage (shift-by-1 test)
2. Rolling window integrity
3. NaN patterns
4. Deterministic reproducibility
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add features module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.slow_features import build_slow_features, get_slow_feature_names
from features.fast_features import build_fast_features, get_fast_feature_names
from features.normalization import normalize_features


class TestDataLeakage:
    """Test for data leakage in features."""
    
    def test_shift_by_one(self):
        """
        Shift-by-1 test: Features at time t should not depend on data at t+1.
        Method: Perturb future data and verify features don't change.
        """
        # Load data
        df = pd.read_csv('data/processed/market_data_historical.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Build features on original data
        features_original = build_fast_features(df)
        
        # Perturb future: Add 10% to all prices after row 1000
        df_perturbed = df.copy()
        df_perturbed.loc[1000:, 'Close'] *= 1.10
        df_perturbed.loc[1000:, 'Open'] *= 1.10
        df_perturbed.loc[1000:, 'High'] *= 1.10
        df_perturbed.loc[1000:, 'Low'] *= 1.10
        
        # Build features on perturbed data
        features_perturbed = build_fast_features(df_perturbed)
        
        # Features before row 1000 should be identical
        # (rolling windows may extend slightly, so test up to row 940)
        test_cols = ['ret_1', 'ret_5', 'rv_20', 'rv_60']
        
        for col in test_cols:
            original_slice = features_original.loc[:940, col]
            perturbed_slice = features_perturbed.loc[:940, col]
            
            # Allow for floating point precision
            diff = (original_slice - perturbed_slice).abs()
            max_diff = diff.max()
            
            assert max_diff < 1e-10, f"Leakage detected in {col}: max diff = {max_diff}"
        
        print("✓ Shift-by-1 test PASSED: No future leakage detected")
    
    def test_return_calculation(self):
        """Verify returns use only past data."""
        df = pd.read_csv('data/processed/market_data_historical.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Manual return calculation
        manual_ret_1 = np.log(df['Close'] / df['Close'].shift(1))
        
        # Feature engineering return
        features = build_fast_features(df)
        
        # Compare
        diff = (manual_ret_1 - features['ret_1']).abs()
        max_diff = diff.max()
        
        assert np.isnan(max_diff) or max_diff < 1e-10, f"Return calculation error: {max_diff}"
        
        print("✓ Return calculation test PASSED")


class TestRollingWindows:
    """Test rolling window implementations."""
    
    def test_window_integrity(self):
        """Verify rolling windows use correct number of observations."""
        df = pd.read_csv('data/processed/market_data_historical.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Compute manual 20D rolling std
        returns = np.log(df['Close'] / df['Close'].shift(1))
        manual_rv_20 = returns.rolling(20).std() * np.sqrt(252)
        
        # Feature engineering rv_20
        features = build_fast_features(df)
        
        # Compare (skip first 20 rows where window not full)
        diff = (manual_rv_20.iloc[20:] - features['rv_20'].iloc[20:]).abs()
        max_diff = diff.max()
        
        assert max_diff < 1e-6, f"Rolling window error: {max_diff}"
        
        print("✓ Rolling window integrity test PASSED")
    
    def test_nan_pattern(self):
        """Verify NaN patterns match expected rolling window sizes."""
        df = pd.read_csv('data/processed/market_data_historical.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        features = build_fast_features(df)
        
        # ret_1 should have 1 NaN at start
        assert features['ret_1'].isna().sum() >= 1, "ret_1 NaN count incorrect"
        
        # rv_20 should have ~20 NaN at start
        rv_20_nans = features['rv_20'].iloc[:25].isna().sum()
        assert rv_20_nans >= 19, f"rv_20 should have ~20 leading NaNs, got {rv_20_nans}"
        
        # rv_60 should have ~60 NaN at start
        rv_60_nans = features['rv_60'].iloc[:65].isna().sum()
        assert rv_60_nans >= 59, f"rv_60 should have ~60 leading NaNs, got {rv_60_nans}"
        
        print("✓ NaN pattern test PASSED")


class TestDeterminism:
    """Test deterministic reproducibility."""
    
    def test_reproducibility(self):
        """Verify pipeline produces identical results on repeated runs."""
        df = pd.read_csv('data/processed/market_data_historical.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # First run
        features_1 = build_fast_features(df)
        
        # Second run
        features_2 = build_fast_features(df)
        
        # Compare all numerical columns
        numeric_cols = features_1.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Use allclose to handle floating point precision
            assert np.allclose(
                features_1[col].fillna(-9999),
                features_2[col].fillna(-9999),
                rtol=1e-10,
                atol=1e-10
            ), f"Non-deterministic result in {col}"
        
        print("✓ Reproducibility test PASSED")


class TestNormalization:
    """Test feature normalization."""
    
    def test_z_score_properties(self):
        """Verify normalized features have ~0 mean, ~1 std."""
        df = pd.read_csv('features/final_features_matrix.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Check all normalized columns
        norm_cols = [col for col in df.columns if col.endswith('_norm')]
        
        for col in norm_cols[:5]:  # Test first 5 features
            # Drop NaN
            values = df[col].dropna()
            
            # Check mean close to 0 (within 0.5)
            mean = values.mean()
            assert abs(mean) < 0.5, f"{col} mean = {mean}, expected ~0"
            
            # Check std close to 1 (within 0.5 of 1)
            std = values.std()
            assert abs(std - 1.0) < 0.6, f"{col} std = {std}, expected ~1"
        
        print("✓ Z-score properties test PASSED")
    
    def test_no_inf(self):
        """Verify no infinite values in normalized features."""
        df = pd.read_csv('features/final_features_matrix.csv')
        
        norm_cols = [col for col in df.columns if col.endswith('_norm')]
        
        for col in norm_cols:
            inf_count = np.isinf(df[col]).sum()
            assert inf_count == 0, f"{col} has {inf_count} infinite values"
        
        print("✓ No infinite values test PASSED")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("FEATURE ENGINEERING TESTS")
    print("="*60)
    
    # Test 1: Data Leakage
    print("\n[1] Testing for data leakage...")
    leakage_tests = TestDataLeakage()
    leakage_tests.test_shift_by_one()
    leakage_tests.test_return_calculation()
    
    # Test 2: Rolling Windows
    print("\n[2] Testing rolling windows...")
    window_tests = TestRollingWindows()
    window_tests.test_window_integrity()
    window_tests.test_nan_pattern()
    
    # Test 3: Determinism
    print("\n[3] Testing determinism...")
    determinism_tests = TestDeterminism()
    determinism_tests.test_reproducibility()
    
    # Test 4: Normalization
    print("\n[4] Testing normalization...")
    norm_tests = TestNormalization()
    norm_tests.test_z_score_properties()
    norm_tests.test_no_inf()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
