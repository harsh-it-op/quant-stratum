import pandas as pd
import joblib
from pathlib import Path
import sys

sys.path.append('C:/Users/ahaan/CODE/Research Papers/ML Regime Classifier/scripts')
from quarterly_retrain import evaluate_week8_gate, infer_regimes_with_models, compute_economic_metrics

features_df = pd.read_csv('features/final_features_matrix.csv', parse_dates=['Date'], index_col='Date')
market_df = pd.read_csv('data/processed/market_data_historical.csv', parse_dates=['Date'], index_col='Date')

current_models = joblib.load('models/hmm_regime_models.joblib')

val_max = pd.Timestamp('2026-03-19')
eval_idx = features_df.index[features_df.index <= val_max][-504:]  # Last 2 years
eval_features = features_df.loc[eval_idx]

current_preds = infer_regimes_with_models(eval_features, current_models)
market_slice = market_df.loc[(market_df.index >= eval_idx.min()) & (market_df.index <= val_max)]

current_eval = evaluate_week8_gate(current_preds, market_slice)
print("Current Model Score (2Y Eval):", current_eval['overall_pass'])
print(current_eval['metrics']['checks'])
