import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
BEHAVIOR_OUTPUT_FILE = BASE_DIR / "output" / "behavior_regime" / "behavior_regime_predictions.csv"
BEHAVIOR_DATA_FILE = BASE_DIR / "data" / "processed" / "nifty500_behavior_data.csv"

def evaluate_model(merged, col_name, model_label):
    print(f"\n================ {model_label.upper()} ===================")
    
    # 1. Regime Counts
    counts = merged[col_name].value_counts()
    dist = (counts / len(merged) * 100).round(2).astype(str) + "%"
    
    # 2. State Persistence
    merged["regime_shift"] = (merged[col_name] != merged[col_name].shift(1))
    merged["regime_block"] = merged["regime_shift"].cumsum()
    
    durations = merged.groupby([col_name, "regime_block"]).size()
    avg_duration = durations.groupby(col_name).mean()
    
    # 3. Financial Characteristics
    stats = merged.groupby(col_name).agg({
        "daily_ret": ["mean", "std"],
        "fwd_ret_20": "mean",
        "abs_fwd_ret_20": "mean"
    })
    
    stats.columns = ["Daily_Ret_Mean", "Daily_Volatility", "Fwd_20d_Ret_Mean", "Fwd_20d_Abs_Ret"]
    stats["Annual_Volatility"] = stats["Daily_Volatility"] * np.sqrt(252)
    
    for c in ["Daily_Ret_Mean", "Fwd_20d_Ret_Mean", "Fwd_20d_Abs_Ret", "Annual_Volatility"]:
        stats[c] = (stats[c] * 100).round(2).astype(str) + "%"
        
    print(f"\n1. Regime Distribution:\n{dist}")
    print(f"\n2. Average Duration (Days):\n{avg_duration.round(1)}")
    print(f"\n3. Returns & Volatility:\n{stats[['Fwd_20d_Ret_Mean', 'Fwd_20d_Abs_Ret', 'Annual_Volatility']]}")


def evaluate_behavior_models():
    preds = pd.read_csv(BEHAVIOR_OUTPUT_FILE)
    preds["Date"] = pd.to_datetime(preds["Date"])
    preds = preds.set_index("Date")
    
    df = pd.read_csv(BEHAVIOR_DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    
    merged = df.join(preds, how="inner")
    
    merged["fwd_ret_20"] = merged["Close"].pct_change(20).shift(-20)
    merged["abs_fwd_ret_20"] = merged["fwd_ret_20"].abs()
    merged["daily_ret"] = merged["Close"].pct_change()
    
    evaluate_model(merged.copy(), "behavior_fast_state", "Fast Model (Tactical)")
    evaluate_model(merged.copy(), "behavior_slow_state", "Slow Model (Structural)")

if __name__ == "__main__":
    evaluate_behavior_models()
