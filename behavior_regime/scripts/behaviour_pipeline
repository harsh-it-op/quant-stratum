#!/usr/bin/env python3
import json
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import joblib
import shutil

warnings.filterwarnings("ignore")

FAST_EMA_SPAN = 5
SLOW_EMA_SPAN = 6
FAST_MIN_DURATION = 5
SLOW_MIN_DURATION = 3
SLOW_MAX_STREAK_DAYS = 17
SLOW_SWITCH_MARGIN = 0.08

SCRIPT_DIR = Path(__file__).resolve().parent
BEHAVIOR_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BEHAVIOR_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "nifty500_behavior_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "output" / "behavior_regime"
OUTPUT_PATH = OUTPUT_DIR / "behavior_regime_predictions.csv"
MODELS_DIR = BEHAVIOR_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs" / "behavior_regime"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import the feature functions from behavior_regimes.py
sys.path.insert(0, str(BEHAVIOR_DIR))
from scripts.behavior_regimes import (
    create_behavior_features, FAST_FEATURES, SLOW_FEATURES, 
    strict_state_labeling, enforce_min_duration, hybrid_interpretation
)


def _cap_transition_diagonal(model: GaussianHMM, max_diag: float = 0.95) -> dict:
    """Cap excessive self-transition probabilities and renormalize each row."""
    trans = np.array(model.transmat_, dtype=float)
    original_diag = np.diag(trans).copy()
    n = trans.shape[0]

    for i in range(n):
        row = trans[i].copy()
        diag_val = row[i]
        if diag_val <= max_diag:
            continue

        excess = diag_val - max_diag
        row[i] = max_diag
        other_idx = [j for j in range(n) if j != i]
        other_sum = row[other_idx].sum()

        if other_sum <= 1e-12:
            row[other_idx] = excess / max(len(other_idx), 1)
        else:
            row[other_idx] = row[other_idx] + excess * (row[other_idx] / other_sum)

        row_sum = row.sum()
        if row_sum <= 1e-12:
            row[:] = 1.0 / n
        else:
            row = row / row_sum
        trans[i] = row

    model.transmat_ = trans
    return {
        "diag_before": [float(x) for x in original_diag],
        "diag_after": [float(x) for x in np.diag(trans)],
        "max_diag_threshold": float(max_diag),
    }


def _behavior_economic_gate(
    features_df: pd.DataFrame,
    slow_model: GaussianHMM,
    slow_scaler: StandardScaler,
    slow_map: dict,
) -> dict:
    """Check whether slow labels are economically coherent on in-sample forward returns."""
    slice_df = features_df.iloc[-756:].copy()
    close = slice_df["Close"].astype(float)
    fwd_ret_20 = close.pct_change(20).shift(-20)
    daily_ret = close.pct_change()

    valid = fwd_ret_20.notna() & daily_ret.notna()
    if valid.sum() < 126:
        return {
            "passed": False,
            "reason": "insufficient_data",
            "details": {"valid_rows": int(valid.sum())},
        }

    X_slow = slice_df.loc[valid, SLOW_FEATURES]
    X_slow = X_slow.replace([np.inf, -np.inf], np.nan).dropna()
    if len(X_slow) < 126:
        return {
            "passed": False,
            "reason": "insufficient_feature_rows",
            "details": {"feature_rows": int(len(X_slow))},
        }

    aligned_idx = X_slow.index
    X_slow_scaled = slow_scaler.transform(X_slow.values)
    states_idx = slow_model.predict(X_slow_scaled)
    state_labels = pd.Series([slow_map[s] for s in states_idx], index=aligned_idx)

    eval_df = pd.DataFrame(
        {
            "state": state_labels,
            "fwd_ret_20": fwd_ret_20.reindex(aligned_idx),
            "daily_ret": daily_ret.reindex(aligned_idx),
        }
    ).dropna()

    if len(eval_df) < 126:
        return {
            "passed": False,
            "reason": "insufficient_eval_rows",
            "details": {"eval_rows": int(len(eval_df))},
        }

    grouped = eval_df.groupby("state")
    avg_fwd = grouped["fwd_ret_20"].mean().to_dict()
    sharpe = (grouped["daily_ret"].mean() / grouped["daily_ret"].std().replace(0, np.nan) * np.sqrt(252)).fillna(0.0).to_dict()
    occupancy = eval_df["state"].value_counts(normalize=True).to_dict()

    mr_occ = float(occupancy.get("Mean-Reverting", 0.0))
    mr_sharpe = float(sharpe.get("Mean-Reverting", 0.0))
    noisy_sharpe = float(sharpe.get("Noisy", 0.0))

    # Reject only when structural state dominates AND is economically much worse than noisy.
    mismatch = mr_occ >= 0.60 and (noisy_sharpe - mr_sharpe) >= 0.75
    return {
        "passed": not mismatch,
        "reason": "ok" if not mismatch else "economic_misalignment",
        "details": {
            "occupancy": {k: float(v) for k, v in occupancy.items()},
            "state_sharpe": {k: float(v) for k, v in sharpe.items()},
            "state_avg_fwd_ret_20": {k: float(v) for k, v in avg_fwd.items()},
        },
    }

def fetch_nifty500():
    print("Checking for NIFTY 500 missing dates...")
    end_dt = datetime.now()
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        last_date = df.index[-1]
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = (end_dt - timedelta(days=15*365)).strftime('%Y-%m-%d')
        df = pd.DataFrame()
        
    end_date_str = (end_dt + timedelta(days=1)).strftime('%Y-%m-%d')
    
    if pd.Timestamp(start_date) < pd.Timestamp(end_date_str):
        print(f"Fetching ^CRSLDX from {start_date} to {end_date_str}")
        new_data = yf.download("^CRSLDX", start=start_date, end=end_date_str, progress=False)
        if not new_data.empty:
            new_data = new_data.reset_index()
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = [col[0] if col[1] == '' else col[0] for col in new_data.columns]
            new_data = new_data.rename(columns={'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close'})
            new_data = new_data[['Date', 'Open', 'High', 'Low', 'Close']]
            new_data.set_index('Date', inplace=True)
            
            df = pd.concat([df, new_data])
            df = df[~df.index.duplicated(keep='last')].sort_index()
            df.to_csv(DATA_PATH)
            return len(new_data)
    return 0

def retrain_models(features_df, run_date):
    print("Retraining behavior models...")

    # For fast
    fast_slice = features_df.iloc[-504:]  # Last 504 days
    if len(fast_slice) < 504: return False
    
    fast_scaler = StandardScaler()
    X_fast = fast_scaler.fit_transform(fast_slice[FAST_FEATURES])
    fast_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    fast_model.fit(X_fast)
    fast_map = strict_state_labeling(fast_model, FAST_FEATURES, fast_scaler)
    
    # For slow
    slow_slice = features_df.iloc[-756:]  # Last 756 days
    slow_scaler = StandardScaler()
    X_slow = slow_scaler.fit_transform(slow_slice[SLOW_FEATURES])
    slow_model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    slow_model.fit(X_slow)
    slow_transition_diag = _cap_transition_diagonal(slow_model, max_diag=0.95)
    slow_map = strict_state_labeling(slow_model, SLOW_FEATURES, slow_scaler)

    # Version the model
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"behavior_regime_components_{timestamp_str}.pkl"
    versioned_path = MODELS_DIR / versioned_filename
    base_path = MODELS_DIR / "behavior_regime_components.pkl"

    payload = {
        "fast_model": fast_model, "fast_scaler": fast_scaler, "fast_map": fast_map,
        "slow_model": slow_model, "slow_scaler": slow_scaler, "slow_map": slow_map,
        "last_trained": run_date.strftime("%Y-%m-%d"),
        "version": timestamp_str,
        "slow_transition_diag": slow_transition_diag,
    }
    
    joblib.dump(payload, versioned_path)
    
    # Copy latest to base .pkl for easier inference fetching
    if len(set(fast_map.values())) < 3 or len(set(slow_map.values())) < 3:
        print('Validation Failed: Degenerate states detected. Discarding this model version and keeping the old one.')
        return False

    econ_gate = _behavior_economic_gate(features_df, slow_model, slow_scaler, slow_map)
    if not econ_gate["passed"]:
        print(f"Validation Failed: Economic alignment gate rejected model ({econ_gate['reason']}). Keeping old model.")
        return False

    shutil.copy2(versioned_path, base_path)
    print(f"Model saved to {versioned_path} and copied to base.")
    return True


def _apply_max_streak_guard(state_idx: np.ndarray, probs: np.ndarray, max_streak: int, switch_margin: float) -> np.ndarray:
    """Force occasional slow-state re-evaluation when one state persists too long."""
    if len(state_idx) == 0:
        return state_idx

    out = np.array(state_idx, dtype=int)
    run_len = 1

    for i in range(1, len(out)):
        if out[i] == out[i - 1]:
            run_len += 1
        else:
            run_len = 1

        if run_len <= max_streak:
            continue

        row = probs[i]
        ranked = np.argsort(row)
        best = int(ranked[-1])
        second = int(ranked[-2])
        gap = float(row[best] - row[second])

        # Hard escape after prolonged lock-in: if one state dominates for too long,
        # force a switch to the second-likeliest state to restore structural adaptivity.
        if best == out[i] and (gap <= switch_margin or run_len > (max_streak + 10)):
            block_end = min(i + SLOW_MIN_DURATION, len(out))
            out[i:block_end] = second
            run_len = max(1, block_end - i)

    return out

def predict_regimes():
    # Load all data to compute features properly
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    features = create_behavior_features(df)
    
    models = joblib.load(MODELS_DIR / "behavior_regime_components.pkl")
    
    X_fast = models["fast_scaler"].transform(features[FAST_FEATURES])
    fast_probs = models["fast_model"].predict_proba(X_fast)
    
    X_slow = models["slow_scaler"].transform(features[SLOW_FEATURES])
    slow_probs = models["slow_model"].predict_proba(X_slow)
    
    fast_df = pd.DataFrame(fast_probs, index=features.index)
    slow_df = pd.DataFrame(slow_probs, index=features.index)
    
    fast_smooth = fast_df.ewm(span=FAST_EMA_SPAN).mean().values
    slow_smooth = slow_df.ewm(span=SLOW_EMA_SPAN).mean().values
    
    res = pd.DataFrame(index=features.index)
    
    def process_model_probs(probs, model_map, name, min_duration):
        sorted_probs = np.sort(probs, axis=1)
        res[f"behavior_{name}_confidence"] = np.round(sorted_probs[:, -1], 4)
        res[f"behavior_{name}_prob_gap"] = np.round(sorted_probs[:, -1] - sorted_probs[:, -2], 4)
        
        final_idx = probs.argmax(axis=1)
        final_idx = enforce_min_duration(final_idx, min_duration)
        res[f"behavior_{name}_state"] = [model_map[idx] for idx in final_idx]

    process_model_probs(fast_smooth, models["fast_map"], "fast", FAST_MIN_DURATION)

    slow_idx = slow_smooth.argmax(axis=1)
    slow_idx = np.array(enforce_min_duration(slow_idx, SLOW_MIN_DURATION), dtype=int)
    slow_idx = _apply_max_streak_guard(
        slow_idx,
        slow_smooth,
        max_streak=SLOW_MAX_STREAK_DAYS,
        switch_margin=SLOW_SWITCH_MARGIN,
    )
    slow_idx = np.array(enforce_min_duration(slow_idx, SLOW_MIN_DURATION), dtype=int)

    slow_sorted = np.sort(slow_smooth, axis=1)
    res["behavior_slow_confidence"] = np.round(slow_sorted[:, -1], 4)
    res["behavior_slow_prob_gap"] = np.round(slow_sorted[:, -1] - slow_sorted[:, -2], 4)
    res["behavior_slow_state"] = [models["slow_map"][idx] for idx in slow_idx]
    
    res["hybrid_action"] = res.apply(hybrid_interpretation, axis=1)
    
    # Save predictions
    res.to_csv(OUTPUT_PATH)
    return res


def _avg_regime_duration(state_series: pd.Series) -> dict:
    state_series = state_series.astype(str)
    blocks = (state_series != state_series.shift(1)).cumsum()
    durations = state_series.groupby(blocks).size()
    block_states = state_series.groupby(blocks).first()
    out = {}
    for state_name in sorted(state_series.dropna().unique()):
        vals = durations[block_states == state_name]
        out[state_name] = float(vals.mean()) if len(vals) > 0 else 0.0
    return out


def _state_metrics(merged: pd.DataFrame, state_col: str, model_name: str) -> pd.DataFrame:
    dist = merged[state_col].value_counts(dropna=True)
    pct = (dist / max(len(merged), 1)) * 100.0
    avg_dur = _avg_regime_duration(merged[state_col])
    grouped = merged.groupby(state_col)

    table = grouped.agg(
        fwd_20d_ret_mean=("fwd_ret_20", "mean"),
        fwd_20d_abs_ret=("abs_fwd_ret_20", "mean"),
        daily_volatility=("daily_ret", "std"),
        positive_20d_hit_rate=("fwd_ret_positive", "mean"),
    ).reset_index()
    table.rename(columns={state_col: "state"}, inplace=True)
    table.insert(0, "model", model_name)
    table["count"] = table["state"].map(dist).fillna(0).astype(int)
    table["distribution_pct"] = table["state"].map(pct).fillna(0.0)
    table["avg_duration_days"] = table["state"].map(avg_dur).fillna(0.0)
    table["annual_volatility"] = table["daily_volatility"] * np.sqrt(252)
    return table[
        [
            "model",
            "state",
            "count",
            "distribution_pct",
            "avg_duration_days",
            "fwd_20d_ret_mean",
            "fwd_20d_abs_ret",
            "daily_volatility",
            "annual_volatility",
            "positive_20d_hit_rate",
        ]
    ]


def write_behavior_reports(predictions: pd.DataFrame) -> dict:
    market = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    merged = market[["Close"]].join(predictions, how="inner").copy()
    merged["daily_ret"] = merged["Close"].pct_change()
    merged["fwd_ret_20"] = merged["Close"].pct_change(20).shift(-20)
    merged["abs_fwd_ret_20"] = merged["fwd_ret_20"].abs()
    merged["fwd_ret_positive"] = (merged["fwd_ret_20"] > 0).astype(float)

    # Keep rows where forward return exists for meaningful forward metrics.
    metric_df = merged.dropna(subset=["fwd_ret_20"]).copy()
    if len(metric_df) == 0:
        raise ValueError("Not enough data to compute forward behavior metrics.")

    fast_metrics = _state_metrics(metric_df, "behavior_fast_state", "fast")
    slow_metrics = _state_metrics(metric_df, "behavior_slow_state", "slow")
    economic_metrics = pd.concat([fast_metrics, slow_metrics], ignore_index=True)

    economic_metrics_path = OUTPUT_DIR / "economic_metrics.csv"
    event_validation_path = OUTPUT_DIR / "event_validation.csv"
    stability_metrics_path = OUTPUT_DIR / "stability_metrics.json"
    evaluation_summary_path = OUTPUT_DIR / "evaluation_summary.json"

    economic_metrics.to_csv(economic_metrics_path, index=False)
    event_validation = economic_metrics[
        ["model", "state", "count", "positive_20d_hit_rate", "fwd_20d_ret_mean", "fwd_20d_abs_ret"]
    ].copy()
    event_validation.to_csv(event_validation_path, index=False)

    fast_state = metric_df["behavior_fast_state"].astype(str)
    slow_state = metric_df["behavior_slow_state"].astype(str)
    fast_flip = float((fast_state != fast_state.shift(1)).sum() / max(len(fast_state) - 1, 1) * 252.0)
    slow_flip = float((slow_state != slow_state.shift(1)).sum() / max(len(slow_state) - 1, 1) * 252.0)
    mismatch = float((fast_state != slow_state).mean())

    stability_metrics = {
        "rows_evaluated": int(len(metric_df)),
        "fast_flip_rate_annualized": fast_flip,
        "slow_flip_rate_annualized": slow_flip,
        "fast_avg_confidence": float(metric_df["behavior_fast_confidence"].mean()),
        "slow_avg_confidence": float(metric_df["behavior_slow_confidence"].mean()),
        "fast_avg_prob_gap": float(metric_df["behavior_fast_prob_gap"].mean()),
        "slow_avg_prob_gap": float(metric_df["behavior_slow_prob_gap"].mean()),
        "fast_vs_slow_mismatch_rate": mismatch,
    }
    with open(stability_metrics_path, "w", encoding="utf-8") as f:
        json.dump(stability_metrics, f, indent=2)

    evaluation_summary = {
        "generated_at": datetime.now().isoformat(),
        "date_range": {
            "start": str(metric_df.index.min().date()),
            "end": str(metric_df.index.max().date()),
        },
        "rows_evaluated": int(len(metric_df)),
        "relevant_files": {
            "economic_metrics": str(economic_metrics_path),
            "event_validation": str(event_validation_path),
            "stability_metrics": str(stability_metrics_path),
        },
        "top_states": {
            "fast": fast_state.value_counts().head(3).to_dict(),
            "slow": slow_state.value_counts().head(3).to_dict(),
        },
    }
    with open(evaluation_summary_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    metrics_payload = {
        "economic_metrics_csv": str(economic_metrics_path),
        "event_validation_csv": str(event_validation_path),
        "stability_metrics_json": str(stability_metrics_path),
        "evaluation_summary_json": str(evaluation_summary_path),
        "rows_evaluated": int(len(metric_df)),
    }

    metrics_log = {
        "timestamp": datetime.now().isoformat(),
        "status": "SUCCESS",
        "artifacts": metrics_payload,
    }
    with open(LOGS_DIR / "behavior_metrics_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics_log) + "\n")

    return metrics_payload

def main():
    ts = datetime.now().isoformat()
    try:
        new_rows = fetch_nifty500()
        
        # Determine if we need to retrain (monthly)
        retrain = False
        today_date = datetime.now().date()
        components_path = MODELS_DIR / "behavior_regime_components.pkl"
        
        if not components_path.exists():
            retrain = True
        else:
            models = joblib.load(components_path)
            last_trained = pd.Timestamp(models["last_trained"]).date()
            if last_trained.month != today_date.month or last_trained.year != today_date.year:
                retrain = True
                
        if retrain:
            df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
            features = create_behavior_features(df)
            retrain_models(features, today_date)
            
        res = predict_regimes()
        metrics_artifacts = write_behavior_reports(res)
        missing_dates_count = new_rows
        
        payload = {
            "timestamp": ts,
            "status": "SUCCESS",
            "datasets": {
                "latest_data_date": str(res.index[-1].date()),
                "missing_dates_count": missing_dates_count,
            },
            "metrics_artifacts": metrics_artifacts,
            "retrained": retrain
        }
    except Exception as e:
        import traceback
        payload = {
            "timestamp": ts,
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print("ERROR:", e)
        
    log_file = LOGS_DIR / "behavior_automation_runs.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(payload) + "\n")
        
if __name__ == "__main__":
    main()
