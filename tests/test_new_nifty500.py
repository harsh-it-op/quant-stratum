import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / 'market_regime' / 'scripts'))
from quarterly_retrain import evaluate_week8_gate, infer_regimes_with_models, compute_economic_metrics

features_df = pd.read_csv('features/final_features_matrix.csv', parse_dates=['Date'], index_col='Date')
market_df = pd.read_csv('data/processed/market_data_historical.csv', parse_dates=['Date'], index_col='Date')

report = pd.read_json("logs/retrain_report_20260319_quarterly_rejected.json")

# Wait, we don't have the new model because it was discarded!
# Let's train a model directly and see its metrics
from quarterly_retrain import RetrainController
import json

c = RetrainController({})
new_models = c._build_new_models(features_df, features_df.index, joblib.load('models/hmm_regime_models.joblib'))
preds = infer_regimes_with_models(features_df, new_models)
metrics = evaluate_week8_gate(preds, market_df)
print(json.dumps(metrics['metrics']['checks'], indent=2))
print("===")
print(compute_economic_metrics(preds, market_df))
