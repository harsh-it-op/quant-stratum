#!/usr/bin/env python3
"""
Walk-forward validation as a standalone script.

Outputs:
- output/walkforward_predictions.csv
- output/walkforward_metrics.json
- output/walkforward_stability_metrics.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_base_dir() -> Path:
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "models").exists() and (script_root / "features").exists():
        return script_root
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent]:
        if (candidate / "models").exists() and (candidate / "features").exists():
            return candidate
    raise FileNotFoundError("Could not resolve project root with models/ and features/ folders")


BASE_DIR = resolve_base_dir()
ROOT_DIR = BASE_DIR.parent
FEATURES_DIR = BASE_DIR / "features"
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output" / "market_regime"
LOGS_DIR = ROOT_DIR / "logs" / "market_regime"
for p in [OUTPUT_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


import sys

sys.path.insert(0, str(BASE_DIR))
from scripts.quarterly_retrain import (  # noqa: E402
    evaluate_week8_gate,
    fast_state_mapping,
    infer_regimes_with_models,
    invert_state_map,
    macro_state_mapping,
    train_hmm_model,
    validate_model_bundle,
)
from scripts.compute_adaptive_alpha import (  # noqa: E402
    ALPHA_SLOW_BASE,
    ALPHA_FAST_BASE,
    ALPHA_MAX_SLOW,
    ALPHA_MAX_FAST,
    GAMMA,
    MACRO_ENTER,
    MACRO_EXIT,
    STRESS_ENTER,
    STRESS_EXIT,
    CHOPPY_ENTER,
    CHOPPY_EXIT,
    MACRO_MIN_DUR,
    FAST_MIN_DUR,
    COOLDOWN_MACRO,
    COOLDOWN_FAST,
    STRESS_OVERRIDE_IMMEDIATE,
    _update_stable,
)
import joblib  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Walk-forward validation")
    parser.add_argument("--features-file", type=str, default=str(FEATURES_DIR / "final_features_matrix.csv"))
    parser.add_argument("--market-file", type=str, default=str(DATA_DIR / "market_data_historical.csv"))
    parser.add_argument("--model-file", type=str, default=str(MODELS_DIR / "hmm_regime_models.joblib"))
    parser.add_argument("--train-days", type=int, default=756, help="Rolling train window in trading days")
    parser.add_argument("--test-days", type=int, default=126, help="Rolling test window in trading days")
    parser.add_argument("--step-days", type=int, default=126, help="Step size between splits")
    parser.add_argument("--min-fast-context-samples", type=int, default=100)
    parser.add_argument("--context-quantile", type=float, default=0.60)
    parser.add_argument("--hmm-n-init", type=int, default=10)
    parser.add_argument("--hmm-n-iter", type=int, default=100)
    parser.add_argument("--hmm-tol", type=float, default=1e-4)
    parser.add_argument("--reg-covar", type=float, default=1e-5)
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="Deterministic seed for HMM restarts. When set, the walk-forward "
        "pipeline is fully reproducible (used to freeze the paper's numbers). "
        "Leave unset for production's random restarts.",
    )
    parser.add_argument(
        "--pred-out",
        type=str,
        default=None,
        help="Optional override path for the predictions CSV (used by the "
        "seed-robustness sweep so per-seed runs do not clobber the frozen anchor).",
    )
    return parser.parse_args()


def append_log(payload):
    with open(LOGS_DIR / "walkforward_validation_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def create_walkforward_splits(index: pd.DatetimeIndex, train_days: int, test_days: int, step_days: int):
    idx = pd.DatetimeIndex(index).sort_values().unique()
    splits = []
    if len(idx) < train_days + test_days:
        return splits
    end_limit = len(idx) - test_days + 1
    split_id = 1
    for cursor in range(train_days, end_limit, step_days):
        train_idx = idx[cursor - train_days : cursor]
        test_idx = idx[cursor : cursor + test_days]
        if len(train_idx) < train_days or len(test_idx) == 0:
            continue
        splits.append({"split_id": split_id, "train_idx": train_idx, "test_idx": test_idx})
        split_id += 1
    return splits


def build_models_for_train(train_frame: pd.DataFrame, template_models: dict, args, split_id: int = 0):
    slow_cols = list(template_models["slow_feature_names"])
    fast_cols = list(template_models["fast_feature_names"])

    X_slow = train_frame[slow_cols].values
    X_fast = train_frame[fast_cols].values
    n_slow_states = int(template_models["slow_hmm"].n_components)
    n_fast_states = int(template_models["fast_hmm_durable"].n_components)

    # Per-split, per-model deterministic seed offsets so each fit gets distinct
    # restarts while the whole run stays reproducible. None -> production random.
    bs = args.base_seed
    seed_slow = (bs + split_id * 100 + 0) if bs is not None else None
    seed_fast_global = (bs + split_id * 100 + 25) if bs is not None else None
    seed_fast_dur = (bs + split_id * 100 + 50) if bs is not None else None
    seed_fast_fra = (bs + split_id * 100 + 75) if bs is not None else None

    slow_hmm = train_hmm_model(
        X_slow,
        n_states=n_slow_states,
        n_iter=args.hmm_n_iter,
        n_init=args.hmm_n_init,
        min_samples=200,
        tol=args.hmm_tol,
        reg_covar=args.reg_covar,
        base_seed=seed_slow,
    )
    if slow_hmm is None:
        raise RuntimeError("Failed to train slow HMM for split")

    slow_map = macro_state_mapping(slow_hmm, slow_cols)
    fragile_states = invert_state_map(slow_map).get(1, [])
    p_frag = slow_hmm.predict_proba(X_slow)[:, fragile_states].sum(axis=1) if len(fragile_states) > 0 else np.zeros(len(X_slow))
    p_frag = np.clip(p_frag, 1e-4, 1 - 1e-4)

    q = float(args.context_quantile)
    durable_mask = p_frag <= float(np.quantile(p_frag, 1 - q))
    fragile_mask = p_frag >= float(np.quantile(p_frag, q))
    min_context = int(args.min_fast_context_samples)

    fast_global = train_hmm_model(
        X_fast,
        n_states=n_fast_states,
        n_iter=args.hmm_n_iter,
        n_init=args.hmm_n_init,
        min_samples=min_context,
        tol=args.hmm_tol,
        reg_covar=args.reg_covar,
        base_seed=seed_fast_global,
    )
    if fast_global is None:
        raise RuntimeError("Failed to train global fast HMM for split")

    fast_durable = (
        train_hmm_model(
            X_fast[durable_mask],
            n_states=n_fast_states,
            n_iter=args.hmm_n_iter,
            n_init=args.hmm_n_init,
            min_samples=min_context,
            tol=args.hmm_tol,
            reg_covar=args.reg_covar,
            base_seed=seed_fast_dur,
        )
        if durable_mask.sum() >= min_context
        else None
    )
    fast_fragile = (
        train_hmm_model(
            X_fast[fragile_mask],
            n_states=n_fast_states,
            n_iter=args.hmm_n_iter,
            n_init=args.hmm_n_init,
            min_samples=min_context,
            tol=args.hmm_tol,
            reg_covar=args.reg_covar,
            base_seed=seed_fast_fra,
        )
        if fragile_mask.sum() >= min_context
        else None
    )
    if fast_durable is None:
        fast_durable = fast_global
    if fast_fragile is None:
        fast_fragile = fast_global

    return {
        "slow_hmm": slow_hmm,
        "fast_hmm_durable": fast_durable,
        "fast_hmm_fragile": fast_fragile,
        "slow_state_map": slow_map,
        "dur_state_map": fast_state_mapping(fast_durable, fast_cols),
        "fra_state_map": fast_state_mapping(fast_fragile, fast_cols),
        "slow_feature_names": slow_cols,
        "fast_feature_names": fast_cols,
        "combined_state_names": [
            "Durable–Calm",
            "Durable–Choppy",
            "Durable–Stress",
            "Fragile–Calm",
            "Fragile–Choppy",
            "Fragile–Stress",
        ],
    }


def compute_stability_summary(predictions: pd.DataFrame):
    out = {
        "rows": int(len(predictions)),
        "combined_flip_rate_per_year": None,
        "macro_flip_rate_per_year": None,
        "fast_flip_rate_per_year": None,
        "min_combined_occupancy": None,
        "combined_occupancy": {},
    }
    if len(predictions) < 2:
        return out

    for col, key in [
        ("combined_state", "combined_flip_rate_per_year"),
        ("macro_state", "macro_flip_rate_per_year"),
        ("fast_state", "fast_flip_rate_per_year"),
    ]:
        if col in predictions.columns:
            vals = predictions[col].astype(str).values
            flips = int((vals[1:] != vals[:-1]).sum())
            out[key] = float(flips / max(len(vals) - 1, 1) * 252)

    if "combined_state" in predictions.columns:
        occ = predictions["combined_state"].value_counts(normalize=True)
        out["combined_occupancy"] = {str(k): float(v) for k, v in occ.to_dict().items()}
        out["min_combined_occupancy"] = float(occ.min()) if len(occ) > 0 else None
    return out


def compute_split_diagnostics(preds: pd.DataFrame, split_id: int, gate_pass: bool, gate_reason: str):
    macro_vals = preds["macro_state"].astype(str).values if "macro_state" in preds.columns else np.array([])
    fast_vals = preds["fast_state"].astype(str).values if "fast_state" in preds.columns else np.array([])
    combined_vals = preds["combined_state"].astype(str).values if "combined_state" in preds.columns else np.array([])

    macro_flips = int((macro_vals[1:] != macro_vals[:-1]).sum()) if len(macro_vals) > 1 else 0
    fast_flips = int((fast_vals[1:] != fast_vals[:-1]).sum()) if len(fast_vals) > 1 else 0
    combined_flips = int((combined_vals[1:] != combined_vals[:-1]).sum()) if len(combined_vals) > 1 else 0

    dominant_regime = None
    dominant_share = None
    single_regime_window = False
    if len(combined_vals) > 0:
        occ = pd.Series(combined_vals).value_counts(normalize=True)
        dominant_regime = str(occ.index[0])
        dominant_share = float(occ.iloc[0])
        single_regime_window = bool(dominant_share >= 0.95)

    instability_reasons = []
    if combined_flips == 0:
        instability_reasons.append("zero_switch_split")
    if single_regime_window:
        instability_reasons.append("single_dominant_regime")
    if not gate_pass:
        instability_reasons.append("week8_gate_failed")

    return {
        "split_id": int(split_id),
        "status": "SUCCESS",
        "rows_predicted": int(len(preds)),
        "macro_flips": int(macro_flips),
        "fast_flips": int(fast_flips),
        "combined_flips": int(combined_flips),
        "zero_switch": bool(combined_flips == 0),
        "dominant_regime": dominant_regime,
        "dominant_share": dominant_share,
        "single_regime_window": bool(single_regime_window),
        "gate_pass": bool(gate_pass),
        "gate_reason": gate_reason,
        "instability_reasons": instability_reasons,
    }


def apply_smoothing_stack(preds: pd.DataFrame) -> pd.DataFrame:
    """Apply the production adaptive-alpha EMA + hysteresis + min-duration
    smoothing to the raw daily posteriors, mirroring
    compute_adaptive_alpha.run_adaptive_alpha and the production
    daily_inference_runtime cascade.

    The walk-forward CSV previously stored RAW argmax labels, which is why the
    cascade's stability advantage (a smoothing artifact) and the paper's headline
    Sharpe never reproduced from this file. Smoothing here makes the saved
    combined_state match what production actually trades. Raw posteriors are
    preserved (suffixed _raw) so q1_corrected.py can still backtest the
    unsmoothed argmax for the raw-vs-smoothed comparison.

    Runs sequentially over the full concatenated walk-forward timeline; test
    windows are non-overlapping (step == test), so dates are unique and the
    EMA/min-duration state carries across split boundaries exactly as a live
    daily stream would.
    """
    df = preds.sort_index().copy()
    n = len(df)
    if n == 0:
        return df

    ema_a = {"p_fragile": 0.5, "p_calm": 1 / 3, "p_choppy": 1 / 3, "p_stress": 1 / 3}
    macro_sm = {"stable": None, "candidate": None, "count": 0, "cooldown": 0, "last_date": None}
    fast_sm = {"stable": None, "candidate": None, "count": 0, "cooldown": 0, "last_date": None}

    p_frag = df["p_fragile"].astype(float).values
    p_calm = df["p_calm"].astype(float).values
    p_chop = df["p_choppy"].astype(float).values
    p_str = df["p_stress"].astype(float).values
    dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in df.index]

    sm_frag = np.zeros(n)
    sm_calm = np.zeros(n)
    sm_chop = np.zeros(n)
    sm_str = np.zeros(n)
    out_macro = [""] * n
    out_fast = [""] * n
    out_combined = [""] * n

    for i in range(n):
        div_frag = abs(p_frag[i] - ema_a["p_fragile"])
        a_slow = ALPHA_SLOW_BASE + (ALPHA_MAX_SLOW - ALPHA_SLOW_BASE) * div_frag ** GAMMA
        ema_a["p_fragile"] = a_slow * p_frag[i] + (1 - a_slow) * ema_a["p_fragile"]

        for raw_val, key in [(p_calm[i], "p_calm"), (p_chop[i], "p_choppy"), (p_str[i], "p_stress")]:
            div = abs(raw_val - ema_a[key])
            a_fast = ALPHA_FAST_BASE + (ALPHA_MAX_FAST - ALPHA_FAST_BASE) * div ** GAMMA
            ema_a[key] = a_fast * raw_val + (1 - a_fast) * ema_a[key]
        _s = ema_a["p_calm"] + ema_a["p_choppy"] + ema_a["p_stress"]
        if _s > 0:
            ema_a["p_calm"] /= _s
            ema_a["p_choppy"] /= _s
            ema_a["p_stress"] /= _s

        sm_frag[i] = ema_a["p_fragile"]
        sm_calm[i] = ema_a["p_calm"]
        sm_chop[i] = ema_a["p_choppy"]
        sm_str[i] = ema_a["p_stress"]

        prev_macro = macro_sm["stable"] or "Durable"
        if ema_a["p_fragile"] >= MACRO_ENTER:
            macro_cand = "Fragile"
        elif ema_a["p_fragile"] <= MACRO_EXIT:
            macro_cand = "Durable"
        else:
            macro_cand = prev_macro

        prev_fast = fast_sm["stable"] or "Calm"
        if ema_a["p_stress"] > STRESS_ENTER:
            fast_cand = "Stress"
        elif prev_fast == "Calm":
            fast_cand = "Choppy" if ema_a["p_choppy"] > CHOPPY_ENTER else "Calm"
        elif prev_fast == "Choppy":
            fast_cand = "Calm" if ema_a["p_choppy"] < CHOPPY_EXIT else "Choppy"
        elif prev_fast == "Stress":
            if ema_a["p_stress"] < STRESS_EXIT:
                fast_cand = "Choppy" if ema_a["p_choppy"] > ema_a["p_calm"] else "Calm"
            else:
                fast_cand = "Stress"
        else:
            fast_cand = prev_fast

        out_macro[i] = _update_stable(
            macro_sm, macro_cand, dates[i], MACRO_MIN_DUR.get(macro_cand, 10), COOLDOWN_MACRO, override=False
        )
        stress_override = STRESS_OVERRIDE_IMMEDIATE and fast_cand == "Stress"
        out_fast[i] = _update_stable(
            fast_sm, fast_cand, dates[i], FAST_MIN_DUR.get(fast_cand, 2), COOLDOWN_FAST, override=stress_override
        )
        out_combined[i] = f"{out_macro[i]}–{out_fast[i]}"

    # Preserve raw posteriors for the raw-vs-smoothed comparison in q1_corrected.
    df["p_fragile_raw"] = p_frag
    df["p_calm_raw"] = p_calm
    df["p_choppy_raw"] = p_chop
    df["p_stress_raw"] = p_str
    df["p_fragile_smooth"] = sm_frag
    df["p_calm_smooth"] = sm_calm
    df["p_choppy_smooth"] = sm_chop
    df["p_stress_smooth"] = sm_str
    # Overwrite the saved state labels with the smoothed (production) cascade.
    df["macro_state"] = out_macro
    df["fast_state"] = out_fast
    df["combined_state"] = out_combined
    return df


def main():
    args = parse_args()
    start = datetime.now()

    try:
        features_df = pd.read_csv(args.features_file, parse_dates=["Date"], index_col="Date").sort_index()
        market_df = pd.read_csv(args.market_file, parse_dates=["Date"], index_col="Date").sort_index()
        template_models = joblib.load(args.model_file)
        validate_model_bundle(template_models, features_df)

        splits = create_walkforward_splits(
            index=features_df.index,
            train_days=int(args.train_days),
            test_days=int(args.test_days),
            step_days=int(args.step_days),
        )
        if len(splits) == 0:
            raise RuntimeError("No valid walk-forward splits for current data/window config")

        all_predictions = []
        split_metrics = []
        failed_splits = []
        split_diagnostics = []

        for s in splits:
            split_id = int(s["split_id"])
            train_idx = s["train_idx"]
            test_idx = s["test_idx"]
            try:
                train_frame = features_df.loc[train_idx].dropna()
                if len(train_frame) < 300:
                    raise RuntimeError(f"Too few train rows after dropna: {len(train_frame)}")
                split_models = build_models_for_train(train_frame, template_models, args, split_id=split_id)

                test_features = features_df.loc[test_idx]
                preds = infer_regimes_with_models(test_features, split_models)
                if len(preds) == 0:
                    raise RuntimeError("No test predictions (dropna removed all rows)")
                preds = preds.copy()
                preds["split_id"] = split_id
                preds["train_start"] = pd.Timestamp(train_idx.min()).strftime("%Y-%m-%d")
                preds["train_end"] = pd.Timestamp(train_idx.max()).strftime("%Y-%m-%d")
                preds["test_start"] = pd.Timestamp(test_idx.min()).strftime("%Y-%m-%d")
                preds["test_end"] = pd.Timestamp(test_idx.max()).strftime("%Y-%m-%d")

                val_start = pd.Timestamp(test_idx.min())
                prior_days = market_df.index[market_df.index < val_start]
                context_start = prior_days[-1] if len(prior_days) > 0 else val_start
                market_slice = market_df.loc[
                    (market_df.index >= context_start) & (market_df.index <= pd.Timestamp(test_idx.max()))
                ]
                gate = evaluate_week8_gate(preds, market_slice)

                split_metrics.append(
                    {
                        "split_id": split_id,
                        "train_start": preds["train_start"].iloc[0],
                        "train_end": preds["train_end"].iloc[0],
                        "test_start": preds["test_start"].iloc[0],
                        "test_end": preds["test_end"].iloc[0],
                        "rows_predicted": int(len(preds)),
                        "gate_pass": bool(gate.get("overall_pass", False)),
                        "gate_reason": gate.get("reason"),
                        "gate_checks": gate.get("metrics", {}).get("checks", {}),
                    }
                )
                split_diagnostics.append(
                    compute_split_diagnostics(
                        preds=preds,
                        split_id=split_id,
                        gate_pass=bool(gate.get("overall_pass", False)),
                        gate_reason=gate.get("reason"),
                    )
                )
                all_predictions.append(preds)
            except Exception as exc:
                failed_splits.append({"split_id": split_id, "error": str(exc)})
                split_diagnostics.append(
                    {
                        "split_id": int(split_id),
                        "status": "FAILED",
                        "error": str(exc),
                        "instability_reasons": ["split_failure"],
                    }
                )

        if len(all_predictions) == 0:
            raise RuntimeError(f"All splits failed: {failed_splits}")

        wf_preds = pd.concat(all_predictions, axis=0).sort_index()
        wf_preds.index.name = "Date"
        # Apply production smoothing (adaptive-α EMA + hysteresis + min-duration)
        # so the saved combined_state is the tradeable cascade, not raw argmax.
        wf_preds = apply_smoothing_stack(wf_preds)

        pred_path = Path(args.pred_out) if args.pred_out else (OUTPUT_DIR / "walkforward_predictions.csv")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        wf_preds.to_csv(pred_path, index=True)

        split_df = pd.DataFrame(split_metrics)
        split_csv = OUTPUT_DIR / "walkforward_stability_metrics.csv"
        split_df.to_csv(split_csv, index=False)

        diagnostics_success = [d for d in split_diagnostics if d.get("status") == "SUCCESS"]
        diagnostics_payload = {
            "timestamp": datetime.now().isoformat(),
            "splits_total": int(len(splits)),
            "splits_succeeded": int(len(diagnostics_success)),
            "splits_failed": int(len(split_diagnostics) - len(diagnostics_success)),
            "zero_switch_splits": int(sum(1 for d in diagnostics_success if d.get("zero_switch", False))),
            "single_regime_splits": int(sum(1 for d in diagnostics_success if d.get("single_regime_window", False))),
            "gate_failed_splits": int(sum(1 for d in diagnostics_success if not d.get("gate_pass", False))),
            "instability_splits": int(sum(1 for d in diagnostics_success if len(d.get("instability_reasons", [])) > 0)),
            "per_split": split_diagnostics,
        }
        diagnostics_path = OUTPUT_DIR / "walkforward_diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")

        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": "SUCCESS",
            "features_file": str(args.features_file),
            "model_file": str(args.model_file),
            "train_days": int(args.train_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "base_seed": (int(args.base_seed) if args.base_seed is not None else None),
            "splits_total": int(len(splits)),
            "splits_succeeded": int(len(all_predictions)),
            "splits_failed": int(len(failed_splits)),
            "failed_splits": failed_splits,
            "predictions_rows": int(len(wf_preds)),
            "predictions_start": str(pd.Timestamp(wf_preds.index.min()).date()),
            "predictions_end": str(pd.Timestamp(wf_preds.index.max()).date()),
            "stability_summary": compute_stability_summary(wf_preds),
            "outputs": {
                "predictions_csv": str(pred_path),
                "split_metrics_csv": str(split_csv),
                "diagnostics_json": str(diagnostics_path),
            },
            "elapsed_seconds": float((datetime.now() - start).total_seconds()),
        }
        metrics_path = OUTPUT_DIR / "walkforward_metrics.json"
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["outputs"]["metrics_json"] = str(metrics_path)

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
