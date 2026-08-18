#!/usr/bin/env python3
"""
Standalone comprehensive evaluator.

Reads predictions + market data and emits:
- output/evaluation_summary.json
- output/stability_metrics.json
- output/economic_metrics.csv
- output/event_validation.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_base_dir() -> Path:
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "output").exists() and (script_root / "data").exists():
        return script_root
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent]:
        if (candidate / "output").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not resolve project root with output/ and data/ folders")


BASE_DIR = resolve_base_dir()
ROOT_DIR = BASE_DIR.parent
OUTPUT_DIR = ROOT_DIR / "output" / "market_regime"
FEATURES_DIR = BASE_DIR / "features"
DATA_DIR = BASE_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs" / "market_regime"
for p in [OUTPUT_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


import sys

sys.path.insert(0, str(BASE_DIR))
from scripts.quarterly_retrain import compute_economic_metrics, evaluate_week8_gate  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate regime predictions")
    parser.add_argument(
        "--predictions-file",
        type=str,
        default=str(OUTPUT_DIR / "walkforward_predictions.csv"),
        help="Predictions CSV path",
    )
    parser.add_argument(
        "--market-file",
        type=str,
        default=str(DATA_DIR / "market_data_historical.csv"),
        help="Market data CSV path",
    )
    return parser.parse_args()


def append_log(payload):
    with open(LOGS_DIR / "evaluate_model_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def load_predictions(pred_path: Path):
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    return pd.read_csv(pred_path, parse_dates=["Date"], index_col="Date").sort_index()


def compute_flip_rate(series: pd.Series):
    vals = series.astype(str).values
    if len(vals) < 2:
        return 0.0
    flips = int((vals[1:] != vals[:-1]).sum())
    return float(flips / max(len(vals) - 1, 1) * 252)


def compute_median_duration(series: pd.Series):
    vals = series.astype(str).values
    if len(vals) == 0:
        return None
    durations = []
    run_len = 1
    for i in range(1, len(vals)):
        if vals[i] == vals[i - 1]:
            run_len += 1
        else:
            durations.append(run_len)
            run_len = 1
    durations.append(run_len)
    return float(np.median(durations)) if len(durations) > 0 else None


def compute_entropy(series: pd.Series):
    vc = series.astype(str).value_counts(normalize=True)
    if len(vc) <= 1:
        return 0.0
    p = vc.values
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent / np.log(len(vc)))


def compute_stability_metrics(predictions: pd.DataFrame):
    out = {}
    for col, prefix in [("combined_state", "combined"), ("macro_state", "macro"), ("fast_state", "fast")]:
        if col in predictions.columns:
            out[f"{prefix}_flip_rate_per_year"] = compute_flip_rate(predictions[col])
            out[f"{prefix}_median_duration_days"] = compute_median_duration(predictions[col])
            out[f"{prefix}_entropy"] = compute_entropy(predictions[col])
    if "combined_state" in predictions.columns:
        occ = predictions["combined_state"].value_counts(normalize=True)
        out["combined_occupancy"] = {str(k): float(v) for k, v in occ.to_dict().items()}
        out["combined_min_occupancy"] = float(occ.min()) if len(occ) > 0 else None
    if "p_fragile" in predictions.columns:
        p = pd.to_numeric(predictions["p_fragile"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
        out["macro_informativeness_midband_pct"] = float(((p > 0.2) & (p < 0.8)).mean() * 100)
    return out


def event_pass(name: str, fragile_pct: float, stress_pct: float):
    n = str(name).lower()
    if "covid-19 crash" in n:
        return (stress_pct >= 30.0) or (fragile_pct >= 60.0)
    if "2022 rate shock" in n:
        return fragile_pct >= 70.0
    return None


def compute_event_validation(predictions: pd.DataFrame):
    events = [
        ("COVID-19 Crash", "2020-02-20", "2020-04-30"),
        ("2022 Rate Shock", "2022-01-01", "2022-12-31"),
        ("Russia-Ukraine Shock", "2022-02-24", "2022-04-30"),
        ("SVB Banking Shock", "2023-03-01", "2023-04-15"),
    ]
    rows = []
    for name, start, end in events:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        sub = predictions.loc[(predictions.index >= s) & (predictions.index <= e)].copy()
        if len(sub) == 0:
            continue
        if "macro_state" in sub.columns:
            fragile_pct = float((sub["macro_state"].astype(str) == "Fragile").mean() * 100)
        else:
            fragile_pct = float(sub["combined_state"].astype(str).str.contains("Fragile", na=False).mean() * 100)
        if "fast_state" in sub.columns:
            stress_pct = float((sub["fast_state"].astype(str) == "Stress").mean() * 100)
        else:
            stress_pct = float(sub["combined_state"].astype(str).str.contains("Stress", na=False).mean() * 100)

        vc = sub["combined_state"].astype(str).value_counts(normalize=True) if "combined_state" in sub.columns else pd.Series(dtype=float)
        rows.append(
            {
                "event": name,
                "start_date": str(s.date()),
                "end_date": str(e.date()),
                "rows": int(len(sub)),
                "fragile_pct": fragile_pct,
                "stress_pct": stress_pct,
                "top_regime": str(vc.index[0]) if len(vc) > 0 else None,
                "top_regime_pct": float(vc.iloc[0] * 100) if len(vc) > 0 else None,
                "rule_pass": event_pass(name, fragile_pct, stress_pct),
            }
        )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    start = datetime.now()
    try:
        pred_path = Path(args.predictions_file)
        if not pred_path.exists():
            fallback = FEATURES_DIR / "regime_timeline_history.csv"
            if fallback.exists():
                pred_path = fallback
        predictions = load_predictions(pred_path)
        market = pd.read_csv(args.market_file, parse_dates=["Date"], index_col="Date").sort_index()

        stability = compute_stability_metrics(predictions)
        gate = evaluate_week8_gate(predictions, market)
        economic = compute_economic_metrics(predictions, market)
        events = compute_event_validation(predictions)

        stability_path = OUTPUT_DIR / "stability_metrics.json"
        economic_path = OUTPUT_DIR / "economic_metrics.csv"
        events_path = OUTPUT_DIR / "event_validation.csv"
        summary_path = OUTPUT_DIR / "evaluation_summary.json"

        stability_path.write_text(json.dumps(stability, indent=2), encoding="utf-8")
        economic.to_csv(economic_path, index=False)
        events.to_csv(events_path, index=False)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": "SUCCESS",
            "predictions_file": str(pred_path),
            "market_file": str(args.market_file),
            "rows_evaluated": int(len(predictions)),
            "date_start": str(pd.Timestamp(predictions.index.min()).date()) if len(predictions) else None,
            "date_end": str(pd.Timestamp(predictions.index.max()).date()) if len(predictions) else None,
            "week8_gate": gate,
            "stability": stability,
            "economic_rows": int(len(economic)),
            "event_rows": int(len(events)),
            "outputs": {
                "stability_metrics_json": str(stability_path),
                "economic_metrics_csv": str(economic_path),
                "event_validation_csv": str(events_path),
                "evaluation_summary_json": str(summary_path),
            },
            "elapsed_seconds": float((datetime.now() - start).total_seconds()),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        append_log(summary)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": "FAILED",
            "error": str(exc),
            "elapsed_seconds": float((datetime.now() - start).total_seconds()),
        }
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
