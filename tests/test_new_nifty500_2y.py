import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / 'market_regime' / 'scripts'))
from quarterly_retrain import evaluate_week8_gate, infer_regimes_with_models, compute_economic_metrics

features_df = pd.read_csv('features/final_features_matrix.csv', parse_dates=['Date'], index_col='Date')
market_df = pd.read_csv('data/processed/market_data_historical.csv', parse_dates=['Date'], index_col='Date')

report = pd.read_json("logs/retrain_report_20260319_quarterly_rejected.json")
import json
from quarterly_retrain import RetrainController

c = RetrainController({})
new_models = c._build_new_models(features_df, features_df.index, joblib.load('models/hmm_regime_models.joblib'))

val_max = pd.Timestamp('2026-03-19')
eval_idx = features_df.index[features_df.index <= val_max][-504:]  # Last 2 years
eval_features = features_df.loc[eval_idx]
new_preds = infer_regimes_with_models(eval_features, new_models)
market_slice = market_df.loc[(market_df.index >= eval_idx.min()) & (market_df.index <= val_max)]

new_eval = evaluate_week8_gate(new_preds, market_slice)
print(json.dumps(new_eval['metrics']['checks'], indent=2))
print("===")
print(compute_economic_metrics(new_preds, market_slice))
