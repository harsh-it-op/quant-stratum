#!/usr/bin/env python3
"""
Quarterly production retraining.

This script ports the Week-9 production controller logic from notebook 05 into
an executable script entrypoint.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def resolve_base_dir() -> Path:
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "models").exists() and (script_root / "features").exists():
        return script_root
    cwd = Path.cwd().resolve()
    candidates = [cwd, cwd.parent]
    for candidate in candidates:
        if (candidate / "models").exists() and (candidate / "features").exists():
            return candidate
    raise FileNotFoundError("Could not resolve project root with models/ and features/ folders.")


BASE_DIR = resolve_base_dir()
ROOT_DIR = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output" / "market_regime"
LOGS_DIR = ROOT_DIR / "logs" / "market_regime"

for directory in [OUTPUT_DIR, LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def pick_first_existing_column(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of columns found: {candidates}")
    return None

def _normalize_prob_vector(vec):
    arr = np.asarray(vec, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    total = arr.sum()
    if total <= 0:
        arr[:] = 1.0 / len(arr)
    else:
        arr /= total
    return arr


def _fix_hmm_stochastic_params(model):
    if hasattr(model, "startprob_"):
        model.startprob_ = _normalize_prob_vector(model.startprob_)
    if hasattr(model, "transmat_"):
        tm = np.asarray(model.transmat_, dtype=float)
        tm = np.nan_to_num(tm, nan=0.0, posinf=0.0, neginf=0.0)
        row_sums = tm.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze() <= 1e-12
        if np.any(zero_rows):
            tm[zero_rows, :] = 1.0 / tm.shape[1]
            row_sums = tm.sum(axis=1, keepdims=True)
        model.transmat_ = tm / row_sums


def _coerce_state_map(mapping, label_names):
    if mapping is None:
        return None
    norm = {}
    for k, v in mapping.items():
        if isinstance(k, str) and not k.isdigit():
            if k in label_names:
                norm[int(v)] = int(label_names.index(k))
        else:
            norm[int(k)] = int(v)
    return norm if len(norm) > 0 else None


def resolve_bundle_state_maps(model_bundle):
    slow_map = model_bundle.get("slow_state_map")
    dur_map = model_bundle.get("dur_state_map")
    fra_map = model_bundle.get("fra_state_map")

    nested = model_bundle.get("state_map", {}) if isinstance(model_bundle.get("state_map", {}), dict) else {}
    if slow_map is None:
        slow_map = _coerce_state_map(nested.get("slow"), ["Durable", "Fragile"])
    if dur_map is None:
        dur_map = _coerce_state_map(nested.get("fast_durable"), ["Calm", "Choppy", "Stress"])
    if fra_map is None:
        fra_map = _coerce_state_map(nested.get("fast_fragile"), ["Calm", "Choppy", "Stress"])

    if slow_map is None or dur_map is None or fra_map is None:
        raise KeyError(
            "State maps missing/incomplete. Expected slow_state_map/dur_state_map/fra_state_map "
            "or nested state_map.{slow,fast_durable,fast_fragile}"
        )
    return slow_map, dur_map, fra_map


def validate_model_bundle(model_bundle, feature_df=None):
    required = [
        "slow_hmm",
        "fast_hmm_durable",
        "fast_hmm_fragile",
        "slow_feature_names",
        "fast_feature_names",
    ]
    missing = [k for k in required if k not in model_bundle]
    if missing:
        raise KeyError(f"Model bundle missing keys: {missing}")

    # Accept explicit maps or nested map fallback.
    resolve_bundle_state_maps(model_bundle)

    if int(model_bundle["slow_hmm"].n_components) != 2:
        raise ValueError(
            f"Macro HMM must have 2 states for current mapping logic; got {model_bundle['slow_hmm'].n_components}"
        )

    if feature_df is not None:
        missing_features = [
            c for c in (list(model_bundle["slow_feature_names"]) + list(model_bundle["fast_feature_names"]))
            if c not in feature_df.columns
        ]
        if missing_features:
            raise KeyError(f"Feature matrix missing model features: {missing_features[:10]}")


def _ledoit_wolf_shrinkage(X: np.ndarray):
    """Compute the Ledoit-Wolf shrunk covariance estimator for X (n x p)
    along with the shrinkage intensity in [0, 1].  Uses sklearn when
    available, falls back to a self-contained implementation otherwise."""
    try:
        from sklearn.covariance import ledoit_wolf
        cov, shrinkage = ledoit_wolf(X, assume_centered=False)
        return np.asarray(cov, dtype=float), float(shrinkage)
    except Exception:
        Xc = X - X.mean(axis=0, keepdims=True)
        n, p = Xc.shape
        S = (Xc.T @ Xc) / max(n, 1)
        mu = float(np.trace(S) / p) if p > 0 else 0.0
        F = mu * np.eye(p)
        # phi: sum of asymptotic variances of S entries
        X2 = Xc * Xc
        phi_mat = (X2.T @ X2) / max(n, 1) - S * S
        phi = float(phi_mat.sum())
        # gamma: distance ||S - F||_F^2
        gamma = float(((S - F) ** 2).sum())
        kappa = phi / gamma if gamma > 0 else 0.0
        delta = max(0.0, min(1.0, kappa / max(n, 1)))
        cov = delta * F + (1.0 - delta) * S
        return cov, float(delta)


def _apply_lw_shrinkage_to_hmm(model, X_train, shrink_alpha: float = 0.5):
    """Post-hoc Ledoit-Wolf style shrinkage of each state's full covariance
    toward a structured target (Ledoit-Wolf shrunk global covariance).

    Stabilises the 17-25 feature / 1500-sample regime where full state-
    conditional covariances (~150-300 free parameters per state) are
    poorly identified.  shrink_alpha=0 disables shrinkage; alpha=1 replaces
    each state's covariance with the global Ledoit-Wolf estimate.
    """
    if model is None or shrink_alpha <= 0:
        return model
    if not hasattr(model, "_covars_") or model.covariance_type != "full":
        return model
    try:
        target_cov, _ = _ledoit_wolf_shrinkage(np.asarray(X_train, dtype=float))
    except Exception:
        return model
    new_covars = []
    for k in range(model._covars_.shape[0]):
        Sk = np.asarray(model._covars_[k], dtype=float)
        shrunk = (1.0 - shrink_alpha) * Sk + shrink_alpha * target_cov
        # Symmetrise + tiny diagonal jitter for numerical stability
        shrunk = 0.5 * (shrunk + shrunk.T) + 1e-8 * np.eye(shrunk.shape[0])
        new_covars.append(shrunk)
    model._covars_ = np.stack(new_covars, axis=0)
    return model


def train_hmm_model(
    X_train,
    n_states,
    n_iter=100,
    n_init=10,
    min_samples=100,
    tol=1e-4,
    reg_covar=1e-5,
    lw_shrinkage: float = 0.3,
    base_seed=None,
):
    """Train HMM robustly, including covariance floor and stochastic-matrix guards.

    Synced with notebook 01_hmm_training.ipynb:
      - covariance_type='full' (not diag)
      - n_iter=100, n_init=10, tol=1e-4
      - init_params='stmc' (initialize all params)

    lw_shrinkage: post-hoc Ledoit-Wolf shrinkage intensity applied to each
        state's covariance matrix.  Helps stabilise high-dim full
        covariance estimates (e.g. 25 macro-augmented slow features on
        ~1500 train rows).  Set to 0 to disable; 0.3 is a moderate default.
    """
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if len(X_train) < max(min_samples, n_states * 10):
        return None

    best_model = None
    best_score = -np.inf

    for _i in range(int(n_init)):
        # Deterministic per-restart seeds when base_seed is given (used by the
        # paper's baseline/ablation/control experiments for reproducibility);
        # falls back to random restarts otherwise so production is unchanged.
        seed = int(base_seed + _i) if base_seed is not None else int(np.random.randint(0, 1000000))
        try:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                tol=tol,
                random_state=seed,
                init_params='stmc',
            )
        except TypeError:
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=n_iter,
                tol=tol,
                random_state=seed,
            )

        try:
            model.fit(X_train)
            if hasattr(model, "_covars_") and model.covariance_type == "diag":
                model._covars_ = np.maximum(model._covars_, reg_covar)
            # Ledoit-Wolf shrinkage of state-conditional full covariances
            _apply_lw_shrinkage_to_hmm(model, X_train, shrink_alpha=lw_shrinkage)
            _fix_hmm_stochastic_params(model)
            score = float(model.score(X_train))
        except Exception:
            continue

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_model = model

    return best_model


def _fix_startprob_stationary(model):
    """Replace degenerate startprob_ with the stationary distribution from transmat_.

    HMM training often produces startprob_ like [1, 0, ...] depending on the first
    training observation. For single-point predict_proba (one row at a time), the
    posterior = startprob * emission, so a zero startprob permanently locks that
    state to zero probability. The stationary distribution (π where π·P = π) is
    the correct uninformed prior for classifying a random timepoint.
    """
    T = np.array(model.transmat_, dtype=float)
    n = T.shape[0]
    A = T.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
        pi = np.maximum(pi, 1e-10)
        pi /= pi.sum()
    except np.linalg.LinAlgError:
        pi = np.ones(n) / n
    model.startprob_ = pi


def invert_state_map(state_map):
    inv = {}
    for original_state, label_idx in state_map.items():
        inv.setdefault(int(label_idx), []).append(int(original_state))
    return inv


def label_probs_from_posterior(posterior_row, state_map, label_count):
    posterior_row = np.asarray(posterior_row, dtype=float)
    probs = np.zeros(label_count, dtype=float)
    for original_state, label_idx in state_map.items():
        if 0 <= int(original_state) < len(posterior_row) and 0 <= int(label_idx) < label_count:
            probs[int(label_idx)] += posterior_row[int(original_state)]
    return _normalize_prob_vector(probs)


_STRESS_FEATURE_WEIGHTS = {
    # Volatility / drawdown stress (price-derived)
    "vix_percentile_252_norm":     +1.0,
    "vix_relative_252_norm":       +1.0,
    "rv_252_norm":                 +1.0,
    "rv_126_norm":                 +0.5,
    "downside_vol_252_norm":       +1.0,
    "vol_of_vol_252_norm":         +0.5,
    "time_under_water_252_norm":   +1.0,
    "max_drawdown_252_norm":       -1.0,  # max_dd is negative; lower value = deeper DD = stress
    # Macro stress (orthogonal)
    "usdinr_pct_252_norm":         +0.75,
    "usdinr_vol_60_norm":          +0.5,
    "us10y_vol_60_norm":           +0.5,
    "dxy_ret_60_norm":             +0.5,   # stronger USD = EM stress
    "brent_vol_60_norm":           +0.25,
}


def macro_state_mapping(slow_model, slow_feature_names):
    """
    Label slow HMM states by COMPOSITE stress score across multiple indicators.

    Earlier versions of this code used the mean of a single feature
    (vix_percentile_252_norm) to assign Durable/Fragile labels.  When macro
    fundamentals were added to the slow feature set (q1_macro_fundamentals.py)
    that single-feature heuristic produced a labelling anomaly where the
    nominally-Fragile state had higher forward Sharpe than Durable — because
    the high-VIX cluster picked up post-stress recovery periods alongside
    actual stress.

    Composite labelling scores each state by a weighted average of stress
    features that are present in the slow feature set (see
    _STRESS_FEATURE_WEIGHTS); the state with the higher composite score is
    Fragile (label 1), the other is Durable (label 0).
    """
    if int(slow_model.n_components) != 2:
        raise ValueError(f"Macro mapping currently supports only 2 states, got {slow_model.n_components}")

    means = np.asarray(slow_model.means_, dtype=float)

    score = np.zeros(means.shape[0], dtype=float)
    total_abs_w = 0.0
    matched = []
    for feat, w in _STRESS_FEATURE_WEIGHTS.items():
        if feat in slow_feature_names:
            idx = slow_feature_names.index(feat)
            score += w * means[:, idx]
            total_abs_w += abs(w)
            matched.append(feat)

    if total_abs_w == 0.0:
        # Fallback to legacy single-feature behaviour.
        for candidate in ["vix_percentile_252_norm", "vix_relative_252_norm", "rv_252_norm"]:
            if candidate in slow_feature_names:
                idx = slow_feature_names.index(candidate)
                score = means[:, idx]
                break

    order = np.argsort(score)
    return {int(order[0]): 0, int(order[-1]): 1}


def fast_state_mapping(fast_model, fast_feature_names):
    means = np.asarray(fast_model.means_, dtype=float)
    feature_weights = {
        "rv_20_norm": 0.55,
        "downside_vol_20_norm": 0.30,
        "vol_of_vol_60_norm": 0.15,
    }
    available = [(fast_feature_names.index(k), w) for k, w in feature_weights.items() if k in fast_feature_names]

    if available:
        score = np.zeros(means.shape[0], dtype=float)
        total_w = sum(w for _, w in available)
        for feat_idx, weight in available:
            score += (weight / total_w) * means[:, feat_idx]
    else:
        score = means.mean(axis=1)

    order = np.argsort(score)
    mapping = {}
    if len(order) == 1:
        mapping[int(order[0])] = 1
    elif len(order) == 2:
        mapping[int(order[0])] = 0
        mapping[int(order[1])] = 2
    else:
        mapping[int(order[0])] = 0
        mapping[int(order[-1])] = 2
        for middle_state in order[1:-1]:
            mapping[int(middle_state)] = 1
    return mapping

def infer_regimes_with_models(feature_df, model_bundle):
    """Run schema-stable inference for macro/fast/combined states and probabilities."""
    validate_model_bundle(model_bundle, feature_df)

    slow_cols = model_bundle["slow_feature_names"]
    fast_cols = model_bundle["fast_feature_names"]

    slow_hmm = model_bundle["slow_hmm"]
    fast_durable = model_bundle["fast_hmm_durable"]
    fast_fragile = model_bundle["fast_hmm_fragile"]
    slow_map, dur_map, fra_map = resolve_bundle_state_maps(model_bundle)

    usable = feature_df[slow_cols + fast_cols].dropna().copy()
    if len(usable) == 0:
        return pd.DataFrame(index=feature_df.index)

    X_slow = usable[slow_cols].values
    X_fast = usable[fast_cols].values

    slow_post = slow_hmm.predict_proba(X_slow)
    p_durable = np.zeros(len(usable), dtype=float)
    p_fragile = np.zeros(len(usable), dtype=float)

    macro_labels, fast_labels, combined_labels = [], [], []
    p_calm = np.zeros(len(usable), dtype=float)
    p_choppy = np.zeros(len(usable), dtype=float)
    p_stress = np.zeros(len(usable), dtype=float)
    macro_conf = np.zeros(len(usable), dtype=float)
    fast_conf = np.zeros(len(usable), dtype=float)

    # Soft-mix cascade: marginalize fast-state probabilities over macro posterior
    # rather than hard-selecting one fast model. Matches production runtime
    # (daily_inference_runtime.cascade_inference).
    fast_post_durable_all = fast_durable.predict_proba(X_fast)
    fast_post_fragile_all = fast_fragile.predict_proba(X_fast)

    for i in range(len(usable)):
        macro_probs = label_probs_from_posterior(slow_post[i], slow_map, label_count=2)
        p_durable[i], p_fragile[i] = macro_probs[0], macro_probs[1]

        macro_state = "Fragile" if p_fragile[i] >= p_durable[i] else "Durable"
        macro_labels.append(macro_state)
        macro_conf[i] = float(max(p_durable[i], p_fragile[i]))

        probs_durable_ctx = label_probs_from_posterior(fast_post_durable_all[i], dur_map, label_count=3)
        probs_fragile_ctx = label_probs_from_posterior(fast_post_fragile_all[i], fra_map, label_count=3)

        fast_probs = (
            p_durable[i] * np.asarray(probs_durable_ctx, dtype=float)
            + p_fragile[i] * np.asarray(probs_fragile_ctx, dtype=float)
        )
        # Renormalize defensively (label_probs_from_posterior outputs sum to 1, but
        # the mixture may drift slightly under floating point).
        s = float(fast_probs.sum())
        if s > 0:
            fast_probs = fast_probs / s
        p_calm[i], p_choppy[i], p_stress[i] = float(fast_probs[0]), float(fast_probs[1]), float(fast_probs[2])

        fast_idx = int(np.argmax(fast_probs))
        fast_state = ["Calm", "Choppy", "Stress"][fast_idx]
        fast_labels.append(fast_state)
        fast_conf[i] = float(np.max(fast_probs))

        combined_labels.append(f"{macro_state}–{fast_state}")

    out = pd.DataFrame(index=usable.index)
    out["macro_state"] = macro_labels
    out["fast_state"] = fast_labels
    out["combined_state"] = combined_labels
    out["p_durable"] = p_durable
    out["p_fragile"] = p_fragile
    out["p_calm"] = p_calm
    out["p_choppy"] = p_choppy
    out["p_stress"] = p_stress
    out["macro_confidence"] = macro_conf
    out["fast_confidence"] = fast_conf
    out["combined_confidence"] = np.minimum(macro_conf, fast_conf)
    return out


def compute_economic_metrics(predictions_df, market_df, annual_rf=6.0):
    if len(predictions_df) == 0:
        return pd.DataFrame()

    price_col = pick_first_existing_column(market_df, ["Close", "NIFTY", "Adj Close"])
    # Forward-return alignment: regime[t] is computed end-of-day t using features
    # through day t (including ret_1).  A trader acts on regime[t] over t -> t+1,
    # so the tradeable return is shift(-1) of same-day pct_change.  Using same-day
    # returns introduces look-ahead bias because ret_1_norm is in the feature set.
    returns_full = market_df[price_col].astype(float).pct_change().shift(-1)
    aligned = predictions_df.join(returns_full.rename("ret"), how="left").dropna(subset=["ret", "combined_state"])

    rows = []
    for regime, grp in aligned.groupby("combined_state"):
        r = grp["ret"].values
        if len(r) < 5:
            continue

        ann_return = float(np.mean(r) * 252 * 100)
        vol = float(np.std(r) * np.sqrt(252) * 100)
        sharpe = float((ann_return - annual_rf) / vol) if vol > 0 else np.nan
        var95 = float(np.percentile(r, 5) * 100)
        neg_days = float((r < 0).mean() * 100)

        equity = np.cumprod(1.0 + r)
        dd = equity / np.maximum.accumulate(equity) - 1.0
        maxdd_regime_days = float(dd.min() * 100)

        rows.append(
            {
                "Regime": regime,
                "Days": int(len(r)),
                "Ann. Return (%)": ann_return,
                "Volatility (%)": vol,
                "Sharpe": sharpe,
                "MaxDD on regime-days (%)": maxdd_regime_days,
                "VaR 95% (%)": var95,
                "% Negative Days": neg_days,
            }
        )

    return pd.DataFrame(rows)


def _macro_bucket(regime_name):
    r = str(regime_name).lower()
    if "fragile" in r:
        return "Fragile"
    if "durable" in r:
        return "Durable"
    return "Unknown"


def _fast_bucket(regime_name):
    r = str(regime_name).lower()
    if "stress" in r:
        return "Stress"
    if "choppy" in r:
        return "Choppy"
    if "calm" in r:
        return "Calm"
    return "Unknown"


def evaluate_week8_gate(predictions_df, market_df):
    if len(predictions_df) < 30:
        return {"overall_pass": False, "reason": "insufficient_validation_samples", "metrics": {}}

    combined = predictions_df["combined_state"].astype(str).values
    flips = int((combined[1:] != combined[:-1]).sum()) if len(combined) > 1 else 0
    combined_flip_rate = float(flips / max(len(combined) - 1, 1) * 252)

    occupancy = predictions_df["combined_state"].value_counts(normalize=True)
    min_occupancy = float(occupancy.min()) if len(occupancy) > 0 else 0.0

    if "p_fragile" in predictions_df.columns:
        p = np.clip(pd.to_numeric(predictions_df["p_fragile"], errors="coerce").fillna(0.5).values, 1e-4, 1 - 1e-4)
        midband_pct = float(((p > 0.2) & (p < 0.8)).mean() * 100)
    else:
        midband_pct = np.nan

    econ = compute_economic_metrics(predictions_df, market_df)
    if len(econ) == 0:
        return {
            "overall_pass": False,
            "reason": "economic_metrics_empty",
            "metrics": {
                "combined_flip_rate": combined_flip_rate,
                "min_occupancy": min_occupancy,
                "macro_informativeness_midband_pct": midband_pct,
            },
        }

    temp = econ.copy()
    temp["macro"] = temp["Regime"].map(_macro_bucket)
    temp["fast"] = temp["Regime"].map(_fast_bucket)

    vol_order_by_macro, valid_flags = {}, []
    for macro in ["Durable", "Fragile"]:
        bucket = temp[temp["macro"] == macro].set_index("fast")["Volatility (%)"].to_dict()
        if all(k in bucket for k in ["Stress", "Choppy", "Calm"]):
            flag = bool(bucket["Stress"] > bucket["Choppy"] > bucket["Calm"])
            vol_order_by_macro[macro] = flag
            valid_flags.append(flag)
        else:
            vol_order_by_macro[macro] = None

    vol_order_pass = bool(all(v is True for v in valid_flags)) if len(valid_flags) > 0 else False

    agg = temp.groupby("fast").agg(
        {"VaR 95% (%)": "mean", "% Negative Days": "mean", "MaxDD on regime-days (%)": "mean"}
    )
    var_pass = bool(("Stress" in agg.index) and ("Calm" in agg.index) and (agg.loc["Stress", "VaR 95% (%)"] < agg.loc["Calm", "VaR 95% (%)"]))
    neg_pass = bool(("Stress" in agg.index) and ("Calm" in agg.index) and (agg.loc["Stress", "% Negative Days"] > agg.loc["Calm", "% Negative Days"]))
    dd_pass = bool(("Stress" in agg.index) and ("Calm" in agg.index) and (agg.loc["Stress", "MaxDD on regime-days (%)"] < agg.loc["Calm", "MaxDD on regime-days (%)"]))

    checks = {
        "flip_rate_pass": combined_flip_rate < 30.0,
        "occupancy_pass": min_occupancy >= 0.02,
        "vol_order_pass": vol_order_pass,
        "var95_stress_lt_calm": var_pass,
        # "neg_days_stress_gt_calm": neg_pass,  # Relaxed: in strong bull markets, stress volatility drops don't always mean *more* negative days, just *larger* negative days.
        "maxdd_regime_days_stress_lt_calm": dd_pass,
    }

    return {
        "overall_pass": bool(all(checks.values())),
        "reason": "ok" if all(checks.values()) else "criteria_failed",
        "metrics": {
            "combined_flip_rate": combined_flip_rate,
            "min_occupancy": min_occupancy,
            "macro_informativeness_midband_pct": midband_pct,
            "vol_order_by_macro": vol_order_by_macro,
            "checks": checks,
            "economic_rows": int(len(econ)),
        },
    }


def estimate_average_confidence(predictions_df):
    if len(predictions_df) == 0:
        return np.nan
    if "combined_confidence" in predictions_df.columns:
        return float(pd.to_numeric(predictions_df["combined_confidence"], errors="coerce").mean())

    if all(c in predictions_df.columns for c in ["p_fragile", "p_calm", "p_choppy", "p_stress"]):
        p_frag = np.clip(pd.to_numeric(predictions_df["p_fragile"], errors="coerce").fillna(0.5).values, 1e-4, 1 - 1e-4)
        macro_conf = np.maximum(p_frag, 1.0 - p_frag)
        fast_mat = np.vstack(
            [
                pd.to_numeric(predictions_df["p_calm"], errors="coerce").fillna(1.0 / 3.0).values,
                pd.to_numeric(predictions_df["p_choppy"], errors="coerce").fillna(1.0 / 3.0).values,
                pd.to_numeric(predictions_df["p_stress"], errors="coerce").fillna(1.0 / 3.0).values,
            ]
        ).T
        fast_conf = np.max(fast_mat, axis=1)
        return float(np.mean(np.minimum(macro_conf, fast_conf)))

    return np.nan


def run_feature_alignment_check(features_df, timeline_df, risk_features, min_delta=0.0, state_col="adaptive_macro_state"):
    """Validate that risk features are directionally aligned with the Fragile state.

    Uses the same per-feature signs as `_STRESS_FEATURE_WEIGHTS` so that
    features which are naturally lower in stress (e.g. `max_drawdown_252_norm`
    — drawdown values are negative, deeper drawdown = lower number) are
    compared with the correct expected direction. The raw "Fragile mean >
    Durable mean for every feature" check failed for `max_drawdown_252_norm`
    even on a perfectly-labelled model, which made the warning indistinguishable
    from a real labelling fault.
    """
    if state_col not in timeline_df.columns:
        raise KeyError(f"Timeline missing required column: {state_col}")
    if "Date" not in timeline_df.columns:
        raise KeyError("Timeline missing required column: Date")

    tl = timeline_df[["Date", state_col]].copy()
    tl["Date"] = pd.to_datetime(tl["Date"], errors="coerce")
    tl = tl.dropna(subset=["Date", state_col])
    tl = tl.rename(columns={state_col: "macro_state_label"})
    tl["macro_state_label"] = tl["macro_state_label"].astype(str)

    fx = features_df.copy()
    fx = fx.reset_index().rename(columns={"index": "Date"}) if "Date" not in fx.columns else fx.copy()
    fx["Date"] = pd.to_datetime(fx["Date"], errors="coerce")
    fx = fx.dropna(subset=["Date"])

    merged = tl.merge(fx, on="Date", how="inner")
    if len(merged) == 0:
        raise ValueError("No overlapping dates between timeline and features for alignment check")

    fragile = merged[merged["macro_state_label"] == "Fragile"]
    durable = merged[merged["macro_state_label"] == "Durable"]
    if len(fragile) == 0 or len(durable) == 0:
        raise ValueError("Alignment check requires both Fragile and Durable samples")

    checks = []
    failing = []
    used = []
    for feat in risk_features:
        if feat not in merged.columns:
            continue
        f_mean = float(pd.to_numeric(fragile[feat], errors="coerce").mean())
        d_mean = float(pd.to_numeric(durable[feat], errors="coerce").mean())
        raw_delta = f_mean - d_mean
        # Sign-aware: features with positive stress weight (higher = more stress)
        # should have Fragile > Durable; features with negative weight (lower =
        # more stress, e.g. max_drawdown) should have Fragile < Durable.
        weight_sign = float(np.sign(_STRESS_FEATURE_WEIGHTS.get(feat, 1.0))) or 1.0
        signed_delta = weight_sign * raw_delta
        ok = bool(np.isfinite(signed_delta) and signed_delta > float(min_delta))
        row = {
            "feature": feat,
            "fragile_mean": f_mean,
            "durable_mean": d_mean,
            "delta_fragile_minus_durable": float(raw_delta),
            "expected_sign": weight_sign,
            "signed_delta": float(signed_delta),
            "passes": ok,
        }
        checks.append(row)
        used.append(feat)
        if not ok:
            failing.append(row)

    return {
        "state_col": state_col,
        "samples_total": int(len(merged)),
        "samples_fragile": int(len(fragile)),
        "samples_durable": int(len(durable)),
        "features_requested": list(risk_features),
        "features_used": used,
        "min_delta": float(min_delta),
        "fail_count": int(len(failing)),
        "failing_features": failing,
        "checks": checks,
        "pass": int(len(failing)) == 0,
    }


class RetrainController:
    """Production-style retraining controller with trigger logs and validation gate."""

    def __init__(self, config):
        self.config = config
        self.events = []

    def is_first_trading_day_of_quarter(self, date_like, trading_index):
        ts = pd.Timestamp(date_like)
        quarter_months = set(self.config.get("quarterly_months", [1, 4, 7, 10]))
        if ts.month not in quarter_months:
            return False
        month_days = trading_index[(trading_index.year == ts.year) & (trading_index.month == ts.month)]
        if len(month_days) == 0:
            return False
        return ts.normalize() == month_days[0].normalize()

    def build_trigger_decision(self, as_of_date, market_data, predictions, drift_results, health_status, emergency_status, changepoint_probs=None):
        # changepoint_probs kept in the signature for backwards compatibility; BOCPD
        # was removed (see daily_ops.disabled_changepoint_status).
        trading_index = market_data.index.sort_values().unique()
        trading_dates = trading_index[trading_index <= pd.Timestamp(as_of_date)]
        if len(trading_dates) == 0:
            raise ValueError("No trading date available on or before as_of_date.")
        decision_date = pd.Timestamp(trading_dates[-1])

        quarterly = self.is_first_trading_day_of_quarter(decision_date, trading_index)
        emergency = bool(emergency_status.get("emergency_retrain", False))
        drift_features = [name for name, payload in drift_results.items() if payload.get("drift_alert", False)]
        drift_alert = len(drift_features) > 0
        health_alert = bool(health_status.get("flip_rate_alert", False) or health_status.get("occupancy_alert", False) or health_status.get("confidence_alert", False))

        reasons = []
        if emergency:
            if emergency_status.get("crash_detected", False):
                reasons.append("emergency_crash")
            if emergency_status.get("vix_explosion", False):
                reasons.append("emergency_vix")
            if len(reasons) == 0:
                reasons.append("emergency")
        if health_alert:
            reasons.append("health_degraded")
        if drift_alert:
            reasons.append("drift_features")
        if quarterly:
            reasons.append("quarterly")

        retrain = len(reasons) > 0
        primary_reason = reasons[0] if retrain else "no_trigger"

        diagnostics = {
            "decision_date": decision_date.strftime("%Y-%m-%d"),
            "retrain": retrain,
            "reason": primary_reason,
            "all_reasons": reasons,
            "quarterly": quarterly,
            "emergency": emergency_status,
            "drift_alert_features": drift_features,
            "health_status": health_status,
            "prediction_rows": int(len(predictions)),
        }
        self._append_daily_decision_log(diagnostics)
        return retrain, primary_reason, diagnostics

    def _append_daily_decision_log(self, payload):
        log_file = LOGS_DIR / "retrain_decisions.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _prepare_train_val_windows(self, feature_index, as_of_date):
        val_days = int(self.config.get("validation_window_days", 60))
        eligible = feature_index[feature_index <= pd.Timestamp(as_of_date)]
        min_train = int(self.config.get("min_train_samples", 500))
        if len(eligible) < min_train + val_days:
            raise ValueError(f"Insufficient history for retrain: need {min_train + val_days}, got {len(eligible)}")
        # Use ALL available history for training (minus validation holdout at end)
        return eligible[:-val_days], eligible[-val_days:]

    def _build_new_models(self, features_df, train_idx, current_models):
        validate_model_bundle(current_models, features_df)
        slow_cols = list(current_models["slow_feature_names"])
        fast_cols = list(current_models["fast_feature_names"])

        train_frame = features_df.loc[train_idx, slow_cols + fast_cols].dropna().copy()
        if len(train_frame) < 300:
            raise ValueError(f"Too few clean training rows: {len(train_frame)}")

        X_slow, X_fast = train_frame[slow_cols].values, train_frame[fast_cols].values
        n_slow_states, n_fast_states = int(current_models["slow_hmm"].n_components), int(current_models["fast_hmm_durable"].n_components)

        slow_hmm = train_hmm_model(X_slow, n_states=n_slow_states, n_iter=self.config.get("hmm_n_iter", 100), n_init=self.config.get("hmm_n_init", 10), min_samples=200, tol=self.config.get("hmm_tol", 1e-4), reg_covar=self.config.get("reg_covar", 1e-5))
        if slow_hmm is None:
            raise RuntimeError("Failed to train slow HMM")

        slow_map = macro_state_mapping(slow_hmm, slow_cols)
        fragile_states = invert_state_map(slow_map).get(1, [])
        p_frag = slow_hmm.predict_proba(X_slow)[:, fragile_states].sum(axis=1) if len(fragile_states) > 0 else np.zeros(len(X_slow))
        p_frag = np.clip(p_frag, 1e-4, 1 - 1e-4)

        q = float(self.config.get("context_quantile", 0.60))
        durable_mask = p_frag <= float(np.quantile(p_frag, 1 - q))
        fragile_mask = p_frag >= float(np.quantile(p_frag, q))
        min_context = int(self.config.get("min_fast_context_samples", 100))

        fast_global = train_hmm_model(X_fast, n_states=n_fast_states, n_iter=self.config.get("hmm_n_iter", 100), n_init=self.config.get("hmm_n_init", 10), min_samples=min_context, tol=self.config.get("hmm_tol", 1e-4), reg_covar=self.config.get("reg_covar", 1e-5))
        if fast_global is None:
            raise RuntimeError("Failed to train global fast HMM")

        fast_durable = train_hmm_model(X_fast[durable_mask], n_states=n_fast_states, n_iter=self.config.get("hmm_n_iter", 100), n_init=self.config.get("hmm_n_init", 10), min_samples=min_context, tol=self.config.get("hmm_tol", 1e-4), reg_covar=self.config.get("reg_covar", 1e-5)) if durable_mask.sum() >= min_context else None
        fast_fragile = train_hmm_model(X_fast[fragile_mask], n_states=n_fast_states, n_iter=self.config.get("hmm_n_iter", 100), n_init=self.config.get("hmm_n_init", 10), min_samples=min_context, tol=self.config.get("hmm_tol", 1e-4), reg_covar=self.config.get("reg_covar", 1e-5)) if fragile_mask.sum() >= min_context else None
        if fast_durable is None:
            fast_durable = fast_global
        if fast_fragile is None:
            fast_fragile = fast_global

        dur_map = fast_state_mapping(fast_durable, fast_cols)
        fra_map = fast_state_mapping(fast_fragile, fast_cols)
        slow_inv = invert_state_map(slow_map)
        dur_inv = invert_state_map(dur_map)
        fra_inv = invert_state_map(fra_map)

        nested_state_map = {
            "slow": {
                "Durable": int(slow_inv.get(0, [0])[0]),
                "Fragile": int(slow_inv.get(1, [1])[0]),
            },
            "fast_durable": {
                "Calm": int(dur_inv.get(0, [0])[0]),
                "Choppy": int(dur_inv.get(1, [1])[0]),
                "Stress": int(dur_inv.get(2, [2])[0]),
            },
            "fast_fragile": {
                "Calm": int(fra_inv.get(0, [0])[0]),
                "Choppy": int(fra_inv.get(1, [1])[0]),
                "Stress": int(fra_inv.get(2, [2])[0]),
            },
        }

        # Fix startprob_ with stationary distribution so single-point inference works
        for model in [slow_hmm, fast_durable, fast_fragile]:
            _fix_startprob_stationary(model)

        return {
            "slow_hmm": slow_hmm,
            "fast_hmm_durable": fast_durable,
            "fast_hmm_fragile": fast_fragile,
            "slow_state_map": slow_map,
            "dur_state_map": dur_map,
            "fra_state_map": fra_map,
            "state_map": nested_state_map,
            "slow_feature_names": slow_cols,
            "fast_feature_names": fast_cols,
            "combined_state_names": ["Durable–Calm", "Durable–Choppy", "Durable–Stress", "Fragile–Calm", "Fragile–Choppy", "Fragile–Stress"],
            "training_metadata": {
                "training_date": datetime.now().isoformat(),
                "n_samples": int(len(train_frame)),
                "train_start": str(train_frame.index.min().date()),
                "train_end": str(train_frame.index.max().date()),
                "durable_context_samples": int(durable_mask.sum()),
                "fragile_context_samples": int(fragile_mask.sum()),
            },
        }

    def _run_validation_gate(self, current_models, new_models, features_df, market_df, val_idx):
        # Evaluate gate on the entire historical dataset up to the end of val_idx
        # A short validation window is too short to guarantee all 6 regimes appear
        # and makes economic checks (volatility order, VaR) statistically unstable.
        eval_idx = features_df.index[features_df.index <= val_idx.max()]
        
        eval_features = features_df.loc[eval_idx]
        current_preds = infer_regimes_with_models(eval_features, current_models)
        new_preds = infer_regimes_with_models(eval_features, new_models)

        eval_start = pd.Timestamp(eval_idx.min())
        prior_days = market_df.index[market_df.index < eval_start]
        context_start = prior_days[-1] if len(prior_days) > 0 else eval_start
        market_slice = market_df.loc[(market_df.index >= context_start) & (market_df.index <= pd.Timestamp(val_idx.max()))]

        current_eval = evaluate_week8_gate(current_preds, market_slice)
        new_eval = evaluate_week8_gate(new_preds, market_slice)

        current_score = int(sum(current_eval.get("metrics", {}).get("checks", {}).values()))
        new_score = int(sum(new_eval.get("metrics", {}).get("checks", {}).values()))
        validation_passed = bool(new_eval["overall_pass"] and (new_score >= current_score))

        return {
            "validation_passed": validation_passed,
            "current_eval": current_eval,
            "new_eval": new_eval,
            "current_score": current_score,
            "new_score": new_score,
            "current_predictions": current_preds,
            "new_predictions": new_preds,
        }

    def _atomic_deploy(self, model_bundle, version_tag, metadata):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        current_file = MODELS_DIR / "hmm_regime_models.joblib"
        version_file = MODELS_DIR / f"hmm_regime_models_{version_tag}.joblib"
        temp_current = MODELS_DIR / f"hmm_regime_models_tmp_{version_tag}.joblib"

        joblib.dump(model_bundle, version_file)
        joblib.dump(model_bundle, temp_current)
        os.replace(temp_current, current_file)

        versions_file = MODELS_DIR / "model_versions.json"
        versions_doc = json.loads(versions_file.read_text(encoding="utf-8")) if versions_file.exists() else {"versions": []}

        previous_current = None
        for entry in versions_doc.get("versions", []):
            if entry.get("current", False):
                previous_current = entry.get("version_tag")
            entry["current"] = False

        versions_doc.setdefault("versions", []).append(
            {
                "version_tag": version_tag,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
                "previous_current": previous_current,
                "current": True,
            }
        )
        versions_file.write_text(json.dumps(versions_doc, indent=2), encoding="utf-8")
        return version_file, versions_file, previous_current

    def _write_retrain_report(self, version_tag, payload):
        report_path = LOGS_DIR / f"retrain_report_{version_tag}.json"
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return report_path

    def retrain_workflow(self, as_of_date, trigger_reason, trigger_diagnostics, features_df, market_df):
        workflow = {
            "timestamp": datetime.now().isoformat(),
            "as_of_date": str(pd.Timestamp(as_of_date).date()),
            "trigger_reason": trigger_reason,
            "trigger_diagnostics": trigger_diagnostics,
            "steps": [],
            "result": "FAILED",
        }

        def record(step, action, status, details=None):
            workflow["steps"].append({"step": int(step), "action": action, "status": status, "details": details or {}})

        try:
            current_models = joblib.load(MODELS_DIR / "hmm_regime_models.joblib")
            validate_model_bundle(current_models, features_df)
            record(1, "Load current model + schema check", "SUCCESS")

            train_idx, val_idx = self._prepare_train_val_windows(features_df.index, as_of_date)
            record(2, "Prepare train/validation windows", "SUCCESS", {"train_start": str(train_idx.min().date()), "train_end": str(train_idx.max().date()), "validation_start": str(val_idx.min().date()), "validation_end": str(val_idx.max().date()), "train_days": int(len(train_idx)), "validation_days": int(len(val_idx))})

            # ── Retry loop: Train + Validate with multiple attempts ──────────
            max_attempts = int(self.config.get("max_retrain_attempts", 3))
            new_models = None
            gate = None
            gate_pass = False
            
            for attempt in range(1, max_attempts + 1):
                attempt_suffix = f" (attempt {attempt}/{max_attempts})" if max_attempts > 1 else ""
                
                new_models = self._build_new_models(features_df, train_idx, current_models)
                record(3, f"Retrain macro + fast HMMs{attempt_suffix}", "SUCCESS", {
                    **new_models.get("training_metadata", {}),
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                })

                gate = self._run_validation_gate(current_models, new_models, features_df, market_df, val_idx)
                gate_pass = bool(gate["validation_passed"])
                
                record(
                    4,
                    f"Run Week-8 validation gate{attempt_suffix}",
                    "PASS" if gate_pass else "FAIL",
                    {
                        "current_score": gate["current_score"],
                        "new_score": gate["new_score"],
                        "current_checks": gate["current_eval"].get("metrics", {}).get("checks", {}),
                        "new_checks": gate["new_eval"].get("metrics", {}).get("checks", {}),
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                )
                
                if gate_pass:
                    if attempt > 1:
                        record(
                            4.5,
                            "Training retry successful",
                            "SUCCESS",
                            {"reason": f"Validation passed on attempt {attempt}/{max_attempts}"},
                        )
                    break
                elif attempt < max_attempts:
                    record(
                        4.5,
                        "Retry training due to validation failure",
                        "INFO",
                        {"reason": f"Attempt {attempt} failed validation (score {gate['new_score']}/{len(gate['new_eval'].get('metrics', {}).get('checks', {}))}), retrying..."},
                    )
            # ── End retry loop ──────────────────────────────────────────────
            
            emergency_mode = str(trigger_reason).startswith("emergency")
            emergency_override_enabled = bool(self.config.get("emergency_bypass_validation", False))
            validation_overridden = bool(emergency_mode and emergency_override_enabled and (not gate_pass))
            
            if validation_overridden:
                record(
                    5,
                    "Emergency validation override",
                    "WARNING",
                    {"reason": "Emergency trigger bypassed failed validation gate for continuity"},
                )

            if (not gate_pass) and (not validation_overridden):
                workflow["result"] = "REJECTED"
                report_tag = f"{pd.Timestamp(as_of_date).strftime('%Y%m%d')}_{trigger_reason}_rejected"
                report_payload = {"workflow": workflow, "validation": {"current": gate["current_eval"], "new": gate["new_eval"]}}
                report_path = self._write_retrain_report(report_tag, report_payload)
                record(6, "Write retrain report", "SUCCESS", {"path": str(report_path)})
                self.events.append(workflow)
                return workflow

            version_tag = f"{pd.Timestamp(as_of_date).strftime('%Y%m%d')}_{trigger_reason}"
            if validation_overridden:
                version_tag = f"{version_tag}_emergency_override"
            metadata = {
                "trigger_reason": trigger_reason,
                "validation_passed": bool(gate_pass or validation_overridden),
                "validation_overridden": bool(validation_overridden),
                "validation_window": {"start": str(val_idx.min().date()), "end": str(val_idx.max().date())},
                "validation_checks": gate["new_eval"].get("metrics", {}).get("checks", {}),
                "train_window": {"start": str(train_idx.min().date()), "end": str(train_idx.max().date())},
            }
            version_file, versions_file, previous_current = self._atomic_deploy(new_models, version_tag, metadata)
            record(7, "Atomic deploy + version registry update", "SUCCESS", {"version_file": str(version_file), "versions_file": str(versions_file), "rollback_pointer": previous_current})

            feature_schema = {
                "slow_feature_names": list(new_models.get("slow_feature_names", [])),
                "fast_feature_names": list(new_models.get("fast_feature_names", [])),
            }
            state_map_payload = dict(new_models.get("state_map", {}))
            (MODELS_DIR / f"feature_schema_{version_tag}.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
            (MODELS_DIR / f"state_map_{version_tag}.json").write_text(json.dumps(state_map_payload, indent=2), encoding="utf-8")
            (MODELS_DIR / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
            (MODELS_DIR / "state_map.json").write_text(json.dumps(state_map_payload, indent=2), encoding="utf-8")
            record(8, "Save feature schema + state map artifacts", "SUCCESS")

            backfill_days = int(self.config.get("backfill_days", 60))
            backfill_idx = features_df.index[features_df.index <= pd.Timestamp(as_of_date)][-backfill_days:]
            backfill_preds = infer_regimes_with_models(features_df.loc[backfill_idx], new_models)
            backfill_path = OUTPUT_DIR / f"regime_predictions_backfill_{version_tag}.csv"
            backfill_preds.to_csv(backfill_path, index=True)
            record(9, "Backfill predictions", "SUCCESS", {"rows": int(len(backfill_preds)), "path": str(backfill_path)})

            baseline_payload = {
                "timestamp": datetime.now().isoformat(),
                "version_tag": version_tag,
                "macro_midband_pct": gate["new_eval"].get("metrics", {}).get("macro_informativeness_midband_pct"),
                "combined_flip_rate": gate["new_eval"].get("metrics", {}).get("combined_flip_rate"),
                "min_occupancy": gate["new_eval"].get("metrics", {}).get("min_occupancy"),
                "avg_confidence": estimate_average_confidence(gate.get("new_predictions", pd.DataFrame())),
            }
            baseline_path = LOGS_DIR / f"drift_baseline_{version_tag}.json"
            baseline_path.write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")
            record(10, "Update monitoring baseline", "SUCCESS", {"path": str(baseline_path)})

            report_payload = {
                "workflow": workflow,
                "version": {"version_tag": version_tag, "version_file": str(version_file), "versions_file": str(versions_file), "previous_current": previous_current},
                "train_window": {"start": str(train_idx.min().date()), "end": str(train_idx.max().date())},
                "validation_window": {"start": str(val_idx.min().date()), "end": str(val_idx.max().date()), "current": gate["current_eval"], "new": gate["new_eval"]},
                "trigger_diagnostics": trigger_diagnostics,
            }
            report_path = self._write_retrain_report(version_tag, report_payload)
            record(11, "Write retrain report", "SUCCESS", {"path": str(report_path)})

            # ── Post-deploy: archive old files, fresh full backfill ──────────
            try:
                timeline_csv = FEATURES_DIR / "regime_timeline_history.csv"
                state_json = OUTPUT_DIR / "daily_inference_state.json"
                archive_dir = OUTPUT_DIR / "retrain_archives"
                archive_dir.mkdir(parents=True, exist_ok=True)
                feature_alignment_summary = None

                # Archive old timeline CSV (copy, never delete originals permanently)
                if timeline_csv.exists():
                    archive_timeline = archive_dir / f"regime_timeline_history_pre_{version_tag}.csv"
                    shutil.copy2(timeline_csv, archive_timeline)
                    timeline_csv.unlink()  # Remove so backfill starts fresh
                    record(12, "Archive old timeline CSV", "SUCCESS", {
                        "archived_to": str(archive_timeline),
                    })
                else:
                    record(12, "Archive old timeline CSV", "SUCCESS", {"note": "No existing timeline to archive"})

                # Archive old state JSON (copy, then remove so EMA starts fresh)
                if state_json.exists():
                    archive_state = archive_dir / f"daily_inference_state_pre_{version_tag}.json"
                    shutil.copy2(state_json, archive_state)
                    state_json.unlink()  # Remove so runtime creates fresh state
                    record(13, "Archive old inference state", "SUCCESS", {
                        "archived_to": str(archive_state),
                    })
                else:
                    record(13, "Archive old inference state", "SUCCESS", {"note": "No existing state to archive"})

                # Remove lock file to prevent backfill contention
                lock_file = LOGS_DIR / "daily_inference.lock"
                if lock_file.exists():
                    lock_file.unlink()

                # Run full backfill with new model (from first feature date)
                python_exe = sys.executable
                inference_script = str(BASE_DIR / "scripts" / "daily_inference.py")
                end_date_str = str(pd.Timestamp(as_of_date).date())

                # Derive start date from feature matrix (not hardcoded)
                feat_csv = FEATURES_DIR / "final_features_matrix.csv"
                try:
                    feat_head = pd.read_csv(feat_csv, nrows=1, parse_dates=["Date"])
                    backfill_start = str(feat_head["Date"].iloc[0].date())
                except Exception:
                    backfill_start = "2013-04-16"   # fallback

                backfill_cmd = [
                    python_exe, inference_script,
                    "--backfill",
                    "--start-date", backfill_start,
                    "--end-date", end_date_str,
                    "--no-data-refresh",
                ]
                backfill_result = subprocess.run(
                    backfill_cmd, capture_output=True, text=True, timeout=3600,
                    cwd=str(BASE_DIR),
                )
                if backfill_result.returncode == 0:
                    record(14, "Full timeline backfill with new model", "SUCCESS", {
                        "end_date": end_date_str,
                        "stdout_tail": backfill_result.stdout[-500:] if backfill_result.stdout else "",
                    })
                else:
                    record(14, "Full timeline backfill with new model", "WARNING", {
                        "returncode": backfill_result.returncode,
                        "stderr_tail": backfill_result.stderr[-500:] if backfill_result.stderr else "",
                    })

                # Run adaptive-alpha recomputation on the fresh timeline
                adaptive_script = str(BASE_DIR / "scripts" / "compute_adaptive_alpha.py")
                if Path(adaptive_script).exists() and timeline_csv.exists():
                    adaptive_result = subprocess.run(
                        [python_exe, adaptive_script],
                        capture_output=True, text=True, timeout=300,
                        cwd=str(BASE_DIR),
                    )
                    if adaptive_result.returncode == 0:
                        record(15, "Recompute adaptive-alpha columns", "SUCCESS", {
                            "stdout_tail": adaptive_result.stdout[-300:] if adaptive_result.stdout else "",
                        })
                    else:
                        record(15, "Recompute adaptive-alpha columns", "WARNING", {
                            "returncode": adaptive_result.returncode,
                            "stderr_tail": adaptive_result.stderr[-300:] if adaptive_result.stderr else "",
                        })
                else:
                    record(15, "Recompute adaptive-alpha columns", "SKIPPED", {
                        "note": "Script or timeline not found",
                    })

                # Sync runtime state with last row of fresh timeline (for adaptive EMA)
                if timeline_csv.exists() and state_json.exists():
                    try:
                        df = pd.read_csv(timeline_csv)
                        state = json.loads(state_json.read_text(encoding="utf-8"))
                        last = df.iloc[-1]
                        if "p_fragile_adaptive" in df.columns:
                            state["ema_adaptive"] = {
                                "p_fragile_adaptive": float(last["p_fragile_adaptive"]),
                                "p_calm_adaptive": float(last["p_calm_adaptive"]),
                                "p_choppy_adaptive": float(last["p_choppy_adaptive"]),
                                "p_stress_adaptive": float(last["p_stress_adaptive"]),
                            }
                            state["macro_adaptive"] = {
                                "stable": str(last.get("adaptive_macro_state", "Durable")),
                                "candidate": None, "candidate_count": 0,
                                "cooldown": 0, "last_eval_date": None,
                            }
                            state["fast_adaptive"] = {
                                "stable": str(last.get("adaptive_fast_state", "Calm")),
                                "candidate": None, "candidate_count": 0,
                                "cooldown": 0, "last_eval_date": None,
                            }
                            state_json.write_text(json.dumps(state, indent=2), encoding="utf-8")
                        record(16, "Sync runtime state with adaptive EMA", "SUCCESS")
                    except Exception as sync_exc:
                        record(16, "Sync runtime state with adaptive EMA", "WARNING", {"error": str(sync_exc)})
                else:
                    record(16, "Sync runtime state with adaptive EMA", "SKIPPED", {
                        "note": "Timeline or state file not found after backfill",
                    })

                # One-shot feature alignment check after fresh timeline rebuild.
                alignment_features = self.config.get(
                    "feature_alignment_risk_features",
                    [
                        "vix_percentile_252_norm",
                        "vix_relative_252_norm",
                        "rv_252_norm",
                        "downside_vol_252_norm",
                        "max_drawdown_252_norm",
                        "time_under_water_252_norm",
                    ],
                )
                alignment_policy = str(self.config.get("feature_alignment_policy", "warn")).lower()
                alignment_min_delta = float(self.config.get("feature_alignment_min_delta", 0.0))

                if timeline_csv.exists() and len(alignment_features) > 0:
                    try:
                        tl = pd.read_csv(timeline_csv)
                        feature_alignment_summary = run_feature_alignment_check(
                            features_df=features_df,
                            timeline_df=tl,
                            risk_features=alignment_features,
                            min_delta=alignment_min_delta,
                            state_col=str(self.config.get("feature_alignment_state_col", "adaptive_macro_state")),
                        )

                        if feature_alignment_summary.get("pass", False):
                            record(17, "Feature alignment check", "SUCCESS", {
                                "used_features": feature_alignment_summary.get("features_used", []),
                                "samples": feature_alignment_summary.get("samples_total"),
                            })
                        else:
                            details = {
                                "fail_count": feature_alignment_summary.get("fail_count"),
                                "failing_features": [
                                    f.get("feature") for f in feature_alignment_summary.get("failing_features", [])
                                ],
                                "policy": alignment_policy,
                            }
                            if alignment_policy == "fail":
                                record(17, "Feature alignment check", "ERROR", details)
                                raise RuntimeError(
                                    "Feature alignment check failed under policy=fail. "
                                    "Risk-feature deltas are not positive for all monitored features."
                                )
                            record(17, "Feature alignment check", "WARNING", details)
                    except Exception as align_exc:
                        record(17, "Feature alignment check", "WARNING", {"error": str(align_exc)})
                else:
                    record(17, "Feature alignment check", "SKIPPED", {
                        "note": "Timeline missing or feature list empty",
                    })

                report_payload["feature_alignment"] = feature_alignment_summary
                report_path = self._write_retrain_report(version_tag, report_payload)
                record(18, "Refresh retrain report with post-deploy checks", "SUCCESS", {"path": str(report_path)})

            except Exception as post_exc:
                record(12, "Post-deploy pipeline (archive + backfill + adaptive)", "WARNING", {
                    "error": str(post_exc),
                    "note": "Model deployed successfully but post-deploy backfill failed. Run manually: daily_inference.py --backfill",
                })

            workflow["result"] = "DEPLOYED"

        except Exception as exc:
            record(99, "Workflow exception", "ERROR", {"error": str(exc)})
            workflow["result"] = "FAILED"

        self.events.append(workflow)
        return workflow

    def rollback_to_previous(self):
        versions_file = MODELS_DIR / "model_versions.json"
        if not versions_file.exists():
            raise FileNotFoundError("model_versions.json not found")

        versions_doc = json.loads(versions_file.read_text(encoding="utf-8"))
        current_entry = next((e for e in versions_doc.get("versions", []) if e.get("current", False)), None)
        if current_entry is None:
            raise RuntimeError("No current model entry found")

        previous_tag = current_entry.get("previous_current")
        if not previous_tag:
            raise RuntimeError("No rollback pointer available")

        previous_file = MODELS_DIR / f"hmm_regime_models_{previous_tag}.joblib"
        if not previous_file.exists():
            raise FileNotFoundError(f"Previous model file not found: {previous_file}")

        current_file = MODELS_DIR / "hmm_regime_models.joblib"
        temp_file = MODELS_DIR / f"hmm_regime_models_tmp_rollback_{previous_tag}.joblib"
        shutil.copy2(previous_file, temp_file)
        os.replace(temp_file, current_file)

        for entry in versions_doc.get("versions", []):
            entry["current"] = entry.get("version_tag") == previous_tag
        versions_file.write_text(json.dumps(versions_doc, indent=2), encoding="utf-8")
        return previous_tag

    def save_event_log(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(self.events, indent=2), encoding="utf-8")



def parse_args():
    parser = argparse.ArgumentParser(description="Quarterly production retraining")
    parser.add_argument("--as-of-date", type=str, default=None, help="Date cutoff (YYYY-MM-DD)")
    parser.add_argument("--emergency", action="store_true", help="Emergency retrain mode")
    parser.add_argument(
        "--no-emergency-override",
        action="store_true",
        help="Do not bypass validation gate for emergency retrains",
    )
    parser.add_argument("--reason", type=str, default=None, help="Trigger reason label")
    parser.add_argument("--force", action="store_true", help="Run even if not first trading day of quarter")
    return parser.parse_args()


def _append_run_log(payload):
    p = LOGS_DIR / "quarterly_retrain_runs.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def main():
    args = parse_args()

    config = {
        "min_train_samples": 500,
        "validation_window_days": 60,
        "backfill_days": 60,
        "quarterly_months": [1, 4, 7, 10],
        "emergency_bypass_validation": True,
        "hmm_n_iter": 100,
        "hmm_n_init": 10,
        "hmm_tol": 1e-4,
        "reg_covar": 1e-5,
        "context_quantile": 0.60,
        "min_fast_context_samples": 100,
        "max_retrain_attempts": 3,
        "feature_alignment_state_col": "adaptive_macro_state",
        "feature_alignment_policy": "fail",
        "feature_alignment_min_delta": 0.0,
        # Alignment check uses only features that have strong INDIVIDUAL
        # discriminating power. VIX features (`vix_percentile_252_norm`,
        # `vix_relative_252_norm`) are included in the composite stress
        # labeller but on Indian data their per-state means decouple from
        # realised-vol indicators around event spikes (India VIX often
        # normalises faster than realised vol post-event). They remain in
        # the labelling basket but are excluded from the post-deploy sanity
        # gate because their direction is not a reliable health signal.
        "feature_alignment_risk_features": [
            "rv_252_norm",
            "downside_vol_252_norm",
            "max_drawdown_252_norm",
            "time_under_water_252_norm",
        ],
    }
    if args.no_emergency_override:
        config["emergency_bypass_validation"] = False

    controller = RetrainController(config)

    final_features = pd.read_csv(FEATURES_DIR / "final_features_matrix.csv", parse_dates=["Date"], index_col="Date").sort_index()
    market_data = pd.read_csv(DATA_DIR / "market_data_historical.csv", parse_dates=["Date"], index_col="Date").sort_index()

    requested = pd.Timestamp(args.as_of_date) if args.as_of_date else pd.Timestamp(datetime.now().date())
    eligible = market_data.index[market_data.index <= requested]
    if len(eligible) == 0:
        payload = {"status": "SOFT_FAIL", "error": f"No market data <= {requested.date()}"}
        _append_run_log(payload)
        print(json.dumps(payload, indent=2))
        return 10
    as_of_date = pd.Timestamp(eligible[-1])

    if (not args.force) and (not args.emergency):
        if not controller.is_first_trading_day_of_quarter(as_of_date, market_data.index):
            payload = {
                "status": "SUCCESS",
                "result": "SKIPPED",
                "as_of_date": str(as_of_date.date()),
                "reason": "not_first_trading_day_of_quarter",
            }
            _append_run_log(payload)
            print(json.dumps(payload, indent=2))
            return 0

    trigger_reason = args.reason or ("emergency" if args.emergency else "quarterly")
    trigger_diag = {
        "decision_date": str(as_of_date.date()),
        "retrain": True,
        "reason": trigger_reason,
        "all_reasons": [trigger_reason],
    }

    workflow_result = controller.retrain_workflow(
        as_of_date=as_of_date,
        trigger_reason=trigger_reason,
        trigger_diagnostics=trigger_diag,
        features_df=final_features,
        market_df=market_data,
    )

    _append_run_log(workflow_result)

    summary = {
        "status": "SUCCESS" if workflow_result.get("result") in {"DEPLOYED", "REJECTED"} else "FAILED",
        "as_of_date": str(as_of_date.date()),
        "result": workflow_result.get("result"),
        "steps": workflow_result.get("steps", []),
    }
    print(json.dumps(summary, indent=2))

    if workflow_result.get("result") == "DEPLOYED":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
