import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Define Data Paths
RAW_FILE = "data/NIFTY 500 01 Jan 2010 - 20 Mar 2026.csv"
CLEAN_FILE = "data/processed/nifty500_behavior_data.csv"
OUTPUT_FILE = "output/behavior_regime/behavior_regime_predictions.csv"

def process_raw_data() -> pd.DataFrame:
    print(f"Loading raw data from {RAW_FILE}...")
    df = pd.read_csv(RAW_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float)
            
    df = df.sort_values("Date").set_index("Date")
    df = df[["Open", "High", "Low", "Close"]]
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_FILE)
    print(f"Cleaned data saved to {CLEAN_FILE} ({len(df)} rows)")
    return df

def rolling_trend_metrics(series, window):
    t = pd.Series(np.arange(len(series)), index=series.index)
    r = series.rolling(window).corr(t)
    r2 = r**2
    tstat = r * np.sqrt(window - 2) / np.sqrt(np.maximum(1 - r2, 1e-8))
    return tstat, r2

def create_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing behavior features for Fast and Slow models...")
    df = df.copy()
    ret = df["Close"].pct_change().fillna(0)
    
    # 1. Trend t-stats & R2
    df["trend_tstat_60"], df["rolling_r2_60"] = rolling_trend_metrics(df["Close"], 60)
    df["trend_tstat_90"], df["rolling_r2_90"] = rolling_trend_metrics(df["Close"], 90)
    df["trend_tstat_120"], df["rolling_r2_120"] = rolling_trend_metrics(df["Close"], 120)
    df["trend_tstat_180"], _ = rolling_trend_metrics(df["Close"], 180)
    
    # 2. Autocorrelations
    df["autocorr_1"] = ret.rolling(20).corr(ret.shift(1))
    df["autocorr_5"] = ret.rolling(20).corr(ret.shift(5))
    df["autocorr_10"] = ret.rolling(40).corr(ret.shift(10))
    
    # 3. Sign Flip Rates
    sign_flips = (ret * ret.shift(1) < 0).astype(int)
    # df["sign_flip_rate_20"] = sign_flips.rolling(20).mean() # REMOVED for stability
    df["sign_flip_rate_60"] = sign_flips.rolling(60).mean()
    df["sign_flip_rate_120"] = sign_flips.rolling(120).mean()
    
    # 4. Returns & Momentum
    df["momentum_persistence_20"] = (ret > 0).astype(int).rolling(20).mean()
    df["momentum_persistence_60"] = (ret > 0).astype(int).rolling(60).mean()
    
    df["ret_20"] = df["Close"].pct_change(20)
    df["ret_60"] = df["Close"].pct_change(60)
    df["ret_120"] = df["Close"].pct_change(120)
    
    # 5. Volatility Shape (skewness, kurtosis)
    df["skewness_60"] = ret.rolling(60).skew()
    df["kurtosis_60"] = ret.rolling(60).kurt()
    df["skewness_120"] = ret.rolling(120).skew()
    df["kurtosis_120"] = ret.rolling(120).kurt()
    
    features = df.dropna()
    print(f"Features computed. Final sample size: {len(features)} days")
    return features


FAST_FEATURES = [
    "trend_tstat_60", "trend_tstat_120", "autocorr_5", "autocorr_10",
    "sign_flip_rate_60", "rolling_r2_60", "rolling_r2_90", "ret_20", "ret_60", 
    "momentum_persistence_20"
]

SLOW_FEATURES = [
    "trend_tstat_120", "trend_tstat_180", "autocorr_5", "autocorr_10",
    "sign_flip_rate_60", "sign_flip_rate_120", "rolling_r2_90", "rolling_r2_120",
    "ret_60", "ret_120", "momentum_persistence_60", "skewness_120", "kurtosis_120"
]

def strict_state_labeling(model, feature_names, scaler):
    means = pd.DataFrame(model.means_, columns=feature_names)
    
    tstat_cols = [c for c in feature_names if "trend_tstat" in c]
    r2_cols = [c for c in feature_names if "rolling_r2" in c]
    autocorr_cols = [c for c in feature_names if "autocorr" in c]
    flip_cols = [c for c in feature_names if "sign_flip" in c]
    kurt_cols = [c for c in feature_names if "kurtosis" in c]
    
    scores_trend = pd.Series(0.0, index=means.index)
    scores_mr = pd.Series(0.0, index=means.index)
    scores_noisy = pd.Series(0.0, index=means.index)
    
    for i in means.index:
        row = means.loc[i]
        
        trend_score = row[tstat_cols].abs().mean() + row[r2_cols].mean() - row[flip_cols].mean() + row[autocorr_cols].mean()
        scores_trend[i] = trend_score
        
        mr_score = row[flip_cols].mean() - row[autocorr_cols].mean() - row[tstat_cols].abs().mean()
        scores_mr[i] = mr_score
        
        noisy_score = row[kurt_cols].mean() - row[r2_cols].mean() - row[autocorr_cols].abs().mean()
        scores_noisy[i] = noisy_score

    labels = {}
    unassigned = [0, 1, 2]
    
    trending_state = scores_trend.idxmax()
    labels[trending_state] = "Trending"
    unassigned.remove(trending_state)
    
    mr_candidates = scores_mr[unassigned]
    mr_state = mr_candidates.idxmax()
    labels[mr_state] = "Mean-Reverting"
    unassigned.remove(mr_state)
    
    noisy_state = unassigned[0]
    labels[noisy_state] = "Noisy"
    
    return labels

def enforce_min_duration(states, min_duration):
    """Collapse runs shorter than `min_duration` into a neighboring run.

    Picks the LONGER of the two neighbors to absorb the short run rather than
    always backfilling to the previous state. The always-backfill variant
    introduced a direction-dependent bias: a short A-B-A pattern collapsed to
    A-A-A, but the same short B between two C runs collapsed to C-C-C, destroying
    transition signal asymmetrically. The longer-neighbor rule is
    direction-symmetric.
    """
    if len(states) == 0:
        return states

    out_states = np.array(states)

    for pass_idx in range(3):
        shifts = np.where(out_states[:-1] != out_states[1:])[0] + 1
        shifts = np.concatenate(([0], shifts, [len(out_states)]))
        for i in range(1, len(shifts) - 1):
            start = shifts[i]
            end = shifts[i + 1]
            length = end - start
            if length >= min_duration:
                continue
            prev_state = out_states[start - 1] if start > 0 else None
            next_state = out_states[end] if end < len(out_states) else None
            if prev_state is None and next_state is None:
                continue
            if prev_state is None:
                out_states[start:end] = next_state
                continue
            if next_state is None:
                out_states[start:end] = prev_state
                continue
            prev_run_len = start - shifts[i - 1]
            next_run_len = shifts[i + 2] - end if i + 2 < len(shifts) else 0
            out_states[start:end] = prev_state if prev_run_len >= next_run_len else next_state
    return list(out_states)

def train_rolling_pipeline(features: pd.DataFrame, feature_cols: list, model_name: str, train_window: int, step_size: int, ema_span: int, min_duration: int):
    print(f"--- Training {model_name} (Window={train_window}, Step={step_size}) ---")
    
    n_days = len(features)
    predicted_probs_df = pd.DataFrame(index=features.index, columns=[0, 1, 2], dtype=float)
    label_mappings = pd.DataFrame(index=features.index, columns=[0, 1, 2], dtype=object)

    for start_idx in range(0, n_days - train_window, step_size):
        train_slice = features.iloc[start_idx : start_idx + train_window]
        test_slice = features.iloc[start_idx + train_window : start_idx + train_window + step_size]
        
        if len(test_slice) == 0: break
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_slice[feature_cols])
        X_test = scaler.transform(test_slice[feature_cols])
        
        model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
        model.fit(X_train)
        
        state_map = strict_state_labeling(model, feature_cols, scaler)
        
        probs = model.predict_proba(X_test)
        
        for i, dt in enumerate(test_slice.index):
            for state_idx in range(3):
                predicted_probs_df.loc[dt, state_idx] = probs[i, state_idx]
                label_mappings.loc[dt, state_idx] = state_map[state_idx]

    predicted_probs_df = predicted_probs_df.dropna()
    label_mappings = label_mappings.loc[predicted_probs_df.index]
    
    smoothed_probs = predicted_probs_df.ewm(span=ema_span).mean()

    # Apply noise floor to prevent 100% confidence (keeps probabilities realistic)
    noise_floor = 0.02 / 3  # ~0.67% minimum per state
    smoothed_probs = smoothed_probs * 0.98 + noise_floor
    
    # Confidence metrics on smoothed posteriors
    sorted_probs = np.sort(smoothed_probs.values, axis=1)
    confidence = sorted_probs[:, -1]
    prob_gap = sorted_probs[:, -1] - sorted_probs[:, -2]
    
    final_states_idx = smoothed_probs.values.argmax(axis=1)
    final_states_idx = enforce_min_duration(final_states_idx, min_duration)
    
    final_labels = []
    for i, dt in enumerate(predicted_probs_df.index):
        state_idx = final_states_idx[i]
        final_labels.append(label_mappings.loc[dt, state_idx])
        
    res = pd.DataFrame(index=predicted_probs_df.index)
    res[f"{model_name}_state"] = final_labels
    res[f"{model_name}_confidence"] = np.round(confidence, 4)
    res[f"{model_name}_prob_gap"] = np.round(prob_gap, 4)
    return res

def hybrid_interpretation(row):
    fast = row["behavior_fast_state"]
    slow = row["behavior_slow_state"]
    fast_conf = row["behavior_fast_confidence"]
    fast_gap = row["behavior_fast_prob_gap"]
    
    # --- CONFIDENCE GATE ---
    # If the fast model is uncertain, strictly defer to the slow model's structural thesis
    if fast_conf < 0.60 or fast_gap < 0.15:
        return f"Structural {slow} (tactical override ignored due to low confidence)"
        
    # --- HYBRID MATRIX ---
    if fast == "Trending" and slow == "Trending": return "strong trend confirmation"
    if fast == "Mean-Reverting" and slow == "Trending": return "pullback inside broader uptrend"
    if fast == "Noisy" and slow == "Noisy": return "low-conviction environment"
    if fast == "Trending" and slow == "Noisy": return "short tactical move, weak structural support"
    if fast == "Mean-Reverting" and slow == "Mean-Reverting": return "structural chop environment"
    if fast == "Noisy" and slow == "Trending": return "pause in uptrend"
    
    return f"{slow} structure with {fast} tactical short-term"

if __name__ == "__main__":
    df = process_raw_data()
    features = create_behavior_features(df)
    
    res_fast = train_rolling_pipeline(
        features, FAST_FEATURES, "behavior_fast", 
        train_window=504, step_size=21, ema_span=7, min_duration=7
    )
    
    res_slow = train_rolling_pipeline(
        features, SLOW_FEATURES, "behavior_slow", 
        train_window=756, step_size=42, ema_span=7, min_duration=10
    )
    
    final_res = res_fast.join(res_slow, how="inner")
    
    final_res["hybrid_action"] = final_res.apply(hybrid_interpretation, axis=1)
    
    Path("output/behavior_regime").mkdir(parents=True, exist_ok=True)
    final_res.to_csv(OUTPUT_FILE)
    print(f"\nSaved hybrid predictions to {OUTPUT_FILE}")
    print("\n--- HYBRID MODEL SUCCESS ---")
    print(final_res["hybrid_action"].value_counts())
