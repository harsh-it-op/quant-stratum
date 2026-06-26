"""
FastAPI backend for regime dashboard.

Endpoints:
    GET /api/current-regime
    GET /api/timeline
    GET /api/metrics
    GET /api/probabilities
    GET /api/health
    GET /api/ops
    GET /api/backfill
    GET /api/regime-changes
    GET /api/model-diagnostics
"""

from datetime import datetime
from pathlib import Path
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Market Regime API",
    description="Real-time market regime classification API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output" / "market_regime"
BEHAVIOR_OUTPUT_DIR = BASE_DIR / "output" / "behavior_regime"
LOGS_DIR = BASE_DIR / "logs" / "market_regime"
BEHAVIOR_LOGS_DIR = BASE_DIR / "logs" / "behavior_regime"

# Market Regime paths
MARKET_MODELS_DIR = BASE_DIR / "market_regime" / "models"
MARKET_FEATURES_DIR = BASE_DIR / "market_regime" / "features"
REGIME_TIMELINE_FILE = Path(
    os.getenv("REGIME_TIMELINE_FILE", str(MARKET_FEATURES_DIR / "regime_timeline_history.csv"))
)

# Behavior Regime paths
BEHAVIOR_MARKET_MODELS_DIR = BASE_DIR / "behavior_regime" / "models"
BEHAVIOR_MARKET_FEATURES_DIR = BASE_DIR / "behavior_regime" / "features"
BEHAVIOR_PREDICTIONS_FILE = BEHAVIOR_OUTPUT_DIR / "behavior_regime_predictions.csv"

TIMELINE_RUNTIME_FILE = REGIME_TIMELINE_FILE  # Keep for backward compatibility


def _safe_float(value, default=0.0):
    try:
        v = float(value)
        if not np.isfinite(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_json(path: Path, default=None):
    if default is None:
        default = {}
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _tail_jsonl(path: Path, limit=200):
    if not path.exists():
        return []
    lines = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)
        payloads = []
        for raw in lines[-int(limit) :]:
            try:
                payloads.append(json.loads(raw))
            except Exception:
                continue
        return payloads
    except Exception:
        return []


def normalize_combined_state(value: str) -> str:
    if value is None:
        return "Unknown"
    text = str(value).strip()
    text = text.replace("Ã¢â‚¬â€œ", "-").replace("â€“", "-").replace("â€”", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    if "-" in text:
        parts = [p.strip() for p in text.split("-", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            return f"{parts[0]}-{parts[1]}"
    return text


def _timeline_base_df() -> pd.DataFrame:
    """Read timeline directly from regime_timeline_history.csv (single source of truth)"""
    timeline_file = TIMELINE_RUNTIME_FILE
    if not timeline_file.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(timeline_file, parse_dates=["Date"])
        if len(df) == 0:
            return pd.DataFrame()
        
        required = {"Date", "macro_state", "fast_state", "combined_state"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame()
        
        df = df.dropna(subset=["Date"]).copy()
        df = df.sort_values("Date")
        return df
    except Exception:
        return pd.DataFrame()



def _normalize_state_map(mapping, label_names, n_states):
    if mapping is None:
        if n_states != len(label_names):
            return {i: min(i, len(label_names) - 1) for i in range(n_states)}
        return {i: i for i in range(n_states)}
    norm = {}
    for k, v in mapping.items():
        if isinstance(k, str) and not k.isdigit():
            if k in label_names:
                norm[int(v)] = int(label_names.index(k))
        else:
            norm[int(k)] = int(v)
    for i in range(n_states):
        if i not in norm:
            norm[i] = min(i, len(label_names) - 1)
    return norm


def _invert_map(mapping):
    inv = {}
    for k, v in mapping.items():
        inv.setdefault(int(v), []).append(int(k))
    return inv


def load_latest_state() -> Dict:
    try:
        # Read directly from regime_timeline_history.csv (full history)
        timeline_file = TIMELINE_RUNTIME_FILE
        if not timeline_file.exists():
            raise FileNotFoundError(f"Timeline file not found: {timeline_file}")
        
        df = pd.read_csv(timeline_file, parse_dates=["Date"])
        if len(df) == 0:
            raise ValueError("Timeline file is empty")
        
        # Get the last row (most recent date)
        row = df.iloc[-1]
        
        # Extract probabilities
        p_fragile = float(_safe_float(row.get("p_fragile_smooth"), 0.5))
        p_calm = float(_safe_float(row.get("p_calm_smooth"), 1.0 / 3.0))
        p_choppy = float(_safe_float(row.get("p_choppy_smooth"), 1.0 / 3.0))
        p_stress = float(_safe_float(row.get("p_stress_smooth"), 1.0 / 3.0))
        p_durable = float(1.0 - p_fragile)
        
        # Extract confidence values
        macro_conf = float(_safe_float(row.get("macro_confidence"), max(p_durable, p_fragile)))
        fast_conf = float(_safe_float(row.get("fast_confidence"), max(p_calm, p_choppy, p_stress)))
        combined_conf = float(_safe_float(row.get("combined_confidence", row.get("confidence")), min(macro_conf, fast_conf)))
        
        # Use file modification time as timestamp
        file_mtime = datetime.fromtimestamp(timeline_file.stat().st_mtime).isoformat()
        
        out = {
            "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
            "timestamp": file_mtime,
            "macro_state": str(row.get("macro_state", "Unknown")),
            "fast_state": str(row.get("fast_state", "Unknown")),
            "combined_state": normalize_combined_state(str(row.get("combined_state", "Unknown"))),
            "p_durable_smooth": p_durable,
            "p_fragile_smooth": p_fragile,
            "p_calm_smooth": p_calm,
            "p_choppy_smooth": p_choppy,
            "p_stress_smooth": p_stress,
            "macro_confidence": macro_conf,
            "fast_confidence": fast_conf,
            "confidence": combined_conf,
            "attribution": {"slow": [], "fast": []},  # Not stored in CSV
            "guardrails": {},  # Not stored in CSV
        }

        # Raw regime labels (argmax of raw probabilities)
        p_fragile_raw = float(_safe_float(row.get("p_fragile_raw"), 0.5))
        p_durable_raw = 1.0 - p_fragile_raw
        p_calm_raw = float(_safe_float(row.get("p_calm_raw"), 1.0 / 3.0))
        p_choppy_raw = float(_safe_float(row.get("p_choppy_raw"), 1.0 / 3.0))
        p_stress_raw = float(_safe_float(row.get("p_stress_raw"), 1.0 / 3.0))
        raw_macro = "Fragile" if p_fragile_raw > p_durable_raw else "Durable"
        raw_fast_probs = {"Calm": p_calm_raw, "Choppy": p_choppy_raw, "Stress": p_stress_raw}
        raw_fast = max(raw_fast_probs, key=raw_fast_probs.get)
        out["raw_macro_state"] = raw_macro
        out["raw_fast_state"] = raw_fast
        out["raw_combined_state"] = f"{raw_macro}\u2013{raw_fast}"
        out["p_durable_raw"] = p_durable_raw
        out["p_fragile_raw"] = p_fragile_raw
        out["p_calm_raw"] = p_calm_raw
        out["p_choppy_raw"] = p_choppy_raw
        out["p_stress_raw"] = p_stress_raw

        # Adaptive-\u03b1 (combined) regime data
        if "p_fragile_adaptive" in df.columns:
            p_frag_a = float(_safe_float(row.get("p_fragile_adaptive"), 0.5))
            p_dur_a = 1.0 - p_frag_a
            out["adaptive_macro_state"] = str(row.get("adaptive_macro_state", "Unknown"))
            out["adaptive_fast_state"] = str(row.get("adaptive_fast_state", "Unknown"))
            out["adaptive_combined_state"] = normalize_combined_state(str(row.get("adaptive_combined_state", "Unknown")))
            out["p_durable_adaptive"] = p_dur_a
            out["p_fragile_adaptive"] = p_frag_a
            out["p_calm_adaptive"] = float(_safe_float(row.get("p_calm_adaptive"), 1.0 / 3.0))
            out["p_choppy_adaptive"] = float(_safe_float(row.get("p_choppy_adaptive"), 1.0 / 3.0))
            out["p_stress_adaptive"] = float(_safe_float(row.get("p_stress_adaptive"), 1.0 / 3.0))

        return out
    except Exception as exc:
        logger.error("Error loading latest state: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def load_timeline(days: Optional[int] = None) -> List[Dict]:
    try:
        df = _timeline_base_df()
        if len(df) == 0:
            raise FileNotFoundError(f"Timeline file not found or empty: {REGIME_TIMELINE_FILE}")
        if days is not None and int(days) > 0:
            df = df.tail(int(days))

        # Check if raw probability columns exist in the data
        has_raw = "p_fragile_raw" in df.columns
        has_adaptive = "p_fragile_adaptive" in df.columns

        timeline = []
        for _, row in df.iterrows():
            p_fragile = _safe_float(row.get("p_fragile_smooth", row.get("p_fragile", 0.5)), 0.5)
            p_calm = _safe_float(row.get("p_calm_smooth", row.get("p_calm", 1.0 / 3.0)), 1.0 / 3.0)
            p_choppy = _safe_float(row.get("p_choppy_smooth", row.get("p_choppy", 1.0 / 3.0)), 1.0 / 3.0)
            p_stress = _safe_float(row.get("p_stress_smooth", row.get("p_stress", 1.0 / 3.0)), 1.0 / 3.0)
            p_durable = 1.0 - p_fragile

            macro_conf = _safe_float(row.get("macro_confidence"), max(p_durable, p_fragile))
            fast_conf = _safe_float(row.get("fast_confidence"), max(p_calm, p_choppy, p_stress))
            combined_conf = _safe_float(row.get("combined_confidence", row.get("confidence")), min(macro_conf, fast_conf))

            entry = {
                "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
                "macro_state": str(row.get("macro_state", "Unknown")),
                "fast_state": str(row.get("fast_state", "Unknown")),
                "combined_state": normalize_combined_state(str(row.get("combined_state", "Unknown"))),
                "confidence": float(combined_conf),
                "macro_confidence": float(macro_conf),
                "fast_confidence": float(fast_conf),
                "p_durable_smooth": float(p_durable),
                "p_fragile_smooth": float(p_fragile),
                "p_calm_smooth": float(p_calm),
                "p_choppy_smooth": float(p_choppy),
                "p_stress_smooth": float(p_stress),
            }

            # Raw (unsmoothed) HMM probabilities — available after re-backfill
            if has_raw:
                p_fragile_raw = _safe_float(row.get("p_fragile_raw", 0.5), 0.5)
                entry["p_durable_raw"] = float(1.0 - p_fragile_raw)
                entry["p_fragile_raw"] = float(p_fragile_raw)
                entry["p_calm_raw"] = float(_safe_float(row.get("p_calm_raw", 1.0 / 3.0), 1.0 / 3.0))
                entry["p_choppy_raw"] = float(_safe_float(row.get("p_choppy_raw", 1.0 / 3.0), 1.0 / 3.0))
                entry["p_stress_raw"] = float(_safe_float(row.get("p_stress_raw", 1.0 / 3.0), 1.0 / 3.0))

            # Adaptive-α probabilities and regime labels
            if has_adaptive:
                p_frag_a = _safe_float(row.get("p_fragile_adaptive", 0.5), 0.5)
                entry["p_durable_adaptive"] = float(1.0 - p_frag_a)
                entry["p_fragile_adaptive"] = float(p_frag_a)
                entry["p_calm_adaptive"] = float(_safe_float(row.get("p_calm_adaptive", 1.0 / 3.0), 1.0 / 3.0))
                entry["p_choppy_adaptive"] = float(_safe_float(row.get("p_choppy_adaptive", 1.0 / 3.0), 1.0 / 3.0))
                entry["p_stress_adaptive"] = float(_safe_float(row.get("p_stress_adaptive", 1.0 / 3.0), 1.0 / 3.0))
                entry["adaptive_macro_state"] = str(row.get("adaptive_macro_state", "Unknown"))
                entry["adaptive_fast_state"] = str(row.get("adaptive_fast_state", "Unknown"))
                entry["adaptive_combined_state"] = normalize_combined_state(str(row.get("adaptive_combined_state", "Unknown")))

            timeline.append(entry)

        # Compute 5-day derivative (rate of change) of smoothed probabilities
        for i, entry in enumerate(timeline):
            if i < 5:
                entry["d_fragile"] = 0.0
                entry["d_durable"] = 0.0
                entry["d_calm"] = 0.0
                entry["d_choppy"] = 0.0
                entry["d_stress"] = 0.0
            else:
                prev = timeline[i - 5]
                entry["d_fragile"] = round(entry["p_fragile_smooth"] - prev["p_fragile_smooth"], 6)
                entry["d_durable"] = round(entry["p_durable_smooth"] - prev["p_durable_smooth"], 6)
                entry["d_calm"] = round(entry["p_calm_smooth"] - prev["p_calm_smooth"], 6)
                entry["d_choppy"] = round(entry["p_choppy_smooth"] - prev["p_choppy_smooth"], 6)
                entry["d_stress"] = round(entry["p_stress_smooth"] - prev["p_stress_smooth"], 6)

        return timeline
    except Exception as exc:
        logger.error("Error loading timeline: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _metrics_candidates() -> List[Path]:
    return [
        OUTPUT_DIR / "economic_metrics.csv",
        OUTPUT_DIR / "walkforward_economic_metrics.csv",
        MARKET_MODELS_DIR / "regime_statistics.csv",
    ]


def load_economic_metrics() -> Dict:
    try:
        metrics_file = next((p for p in _metrics_candidates() if p.exists()), None)
        if metrics_file is None:
            return {}

        df = pd.read_csv(metrics_file)
        metrics = {}

        if "Regime" in df.columns:
            for _, row in df.iterrows():
                regime = normalize_combined_state(str(row.get("Regime", "Unknown")))
                ann_return = _safe_float(row.get("Ann. Return (%)", row.get("Annual Return (%)", 0.0)))
                vol = _safe_float(row.get("Volatility (%)", 0.0))
                sharpe = _safe_float(row.get("Sharpe", 0.0))
                max_dd = _safe_float(
                    row.get("MaxDD on regime-days (%)", row.get("Max Drawdown (%)", row.get("MaxDD (%)", 0.0)))
                )
                var95 = _safe_float(row.get("VaR 95% (%)", row.get("VaR95 (%)", np.nan)), np.nan)
                neg_days = _safe_float(row.get("% Negative Days", row.get("Negative Days (%)", np.nan)), np.nan)
                metrics[regime] = {
                    "return": ann_return / 100.0,
                    "volatility": vol / 100.0,
                    "sharpe": sharpe,
                    "max_dd": max_dd / 100.0,
                    "count": _safe_int(row.get("Days", row.get("Count", 0)), 0),
                    "var95": (var95 / 100.0) if np.isfinite(var95) else None,
                    "neg_days": (neg_days / 100.0) if np.isfinite(neg_days) else None,
                }
        elif "regime" in df.columns:
            for _, row in df.iterrows():
                regime = normalize_combined_state(str(row.get("regime", "Unknown")))
                metrics[regime] = {
                    "return": _safe_float(row.get("mean_return", row.get("return", 0.0))),
                    "volatility": _safe_float(row.get("volatility", 0.0)),
                    "sharpe": _safe_float(row.get("sharpe", 0.0)),
                    "max_dd": _safe_float(row.get("max_drawdown", row.get("max_dd", 0.0))),
                    "count": _safe_int(row.get("count", 0), 0),
                    "mean_duration": _safe_float(row.get("mean_duration", 0.0)),
                }

        return metrics
    except Exception as exc:
        logger.error("Error loading metrics: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def _expected_duration_from_transmat(transmat):
    tm = np.asarray(transmat, dtype=float)
    out = []
    for i in range(tm.shape[0]):
        stay = float(np.clip(tm[i, i], 1e-8, 0.999999))
        out.append(float(1.0 / max(1.0 - stay, 1e-8)))
    return out


def _empirical_durations(col: str) -> Dict:
    """Compute empirical streak durations from regime_timeline_history.csv."""
    try:
        df = pd.read_csv(TIMELINE_RUNTIME_FILE)
        if len(df) == 0 or col not in df.columns:
            return {}
        vals = df[col].astype(str).tolist()
        streaks: Dict[str, list] = {}
        current = vals[0]
        count = 1
        for i in range(1, len(vals)):
            if vals[i] == current:
                count += 1
            else:
                streaks.setdefault(current, []).append(count)
                current = vals[i]
                count = 1
        # Exclude the last ongoing streak (it's incomplete)
        result = {}
        for state, durs in streaks.items():
            import statistics
            result[state] = {
                "n": len(durs),
                "mean": float(round(sum(durs) / len(durs), 1)),
                "median": float(round(statistics.median(durs), 1)),
                "min": int(min(durs)),
                "max": int(max(durs)),
            }
        return result
    except Exception:
        return {}


def _aggregate_label_transition(row_probs, state_map, label_count):
    arr = np.asarray(row_probs, dtype=float)
    out = np.zeros(label_count, dtype=float)
    for orig_idx, label_idx in state_map.items():
        if 0 <= int(orig_idx) < len(arr) and 0 <= int(label_idx) < label_count:
            out[int(label_idx)] += arr[int(orig_idx)]
    s = float(out.sum())
    if s > 0:
        out /= s
    return out


def _resolve_bundle_maps(bundle):
    slow_map = bundle.get("slow_state_map")
    dur_map = bundle.get("dur_state_map")
    fra_map = bundle.get("fra_state_map")
    nested = bundle.get("state_map", {}) if isinstance(bundle.get("state_map", {}), dict) else {}
    if slow_map is None:
        slow_map = nested.get("slow")
    if dur_map is None:
        dur_map = nested.get("fast_durable")
    if fra_map is None:
        fra_map = nested.get("fast_fragile")
    slow_map = _normalize_state_map(slow_map, ["Durable", "Fragile"], bundle["slow_hmm"].n_components)
    dur_map = _normalize_state_map(dur_map, ["Calm", "Choppy", "Stress"], bundle["fast_hmm_durable"].n_components)
    fra_map = _normalize_state_map(fra_map, ["Calm", "Choppy", "Stress"], bundle["fast_hmm_fragile"].n_components)
    return slow_map, dur_map, fra_map


def _timeline_df(days: Optional[int] = None) -> pd.DataFrame:
    df = _timeline_base_df()
    if len(df) == 0:
        return pd.DataFrame()
    if days is not None and int(days) > 0:
        df = df.tail(int(days))
    if len(df) == 0:
        return df

    df["combined_state"] = df["combined_state"].apply(normalize_combined_state)
    df["macro_state"] = df["macro_state"].astype(str)
    df["fast_state"] = df["fast_state"].astype(str)
    df["p_fragile_smooth"] = pd.to_numeric(df.get("p_fragile_smooth", df.get("p_fragile", 0.5)), errors="coerce").fillna(0.5)
    df["p_durable_smooth"] = 1.0 - df["p_fragile_smooth"]
    df["p_calm_smooth"] = pd.to_numeric(df.get("p_calm_smooth", df.get("p_calm", 1.0 / 3.0)), errors="coerce").fillna(1.0 / 3.0)
    df["p_choppy_smooth"] = pd.to_numeric(
        df.get("p_choppy_smooth", df.get("p_choppy", 1.0 / 3.0)), errors="coerce"
    ).fillna(1.0 / 3.0)
    df["p_stress_smooth"] = pd.to_numeric(
        df.get("p_stress_smooth", df.get("p_stress", 1.0 / 3.0)), errors="coerce"
    ).fillna(1.0 / 3.0)
    return df


def _normalized_entropy(prob_row: np.ndarray) -> float:
    probs = np.asarray(prob_row, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    h = -float(np.sum(probs * np.log(probs)))
    hmax = float(np.log(len(probs))) if len(probs) > 1 else 1.0
    return float(h / hmax) if hmax > 0 else 0.0


def _window_stats(df: pd.DataFrame, window_days: int):
    out = {
        "days_used": 0,
        "occupancy": {},
        "dominant_state": None,
        "dominant_share": None,
        "min_occupancy": None,
        "collapse_detected": False,
        "degenerate_detected": False,
        "flip_rates": {"combined": None, "macro": None, "fast": None},
    }
    if len(df) == 0:
        return out
    wdf = df.tail(int(window_days)).copy()
    out["days_used"] = int(len(wdf))
    if len(wdf) == 0:
        return out

    occ = wdf["combined_state"].value_counts(normalize=True)
    out["occupancy"] = {str(k): float(v) for k, v in occ.to_dict().items()}
    if len(occ) > 0:
        out["dominant_state"] = str(occ.index[0])
        out["dominant_share"] = float(occ.iloc[0])
        out["min_occupancy"] = float(occ.min())
        out["collapse_detected"] = bool(float(occ.min()) < 0.02)
        out["degenerate_detected"] = bool(float(occ.iloc[0]) > 0.90)

    def _flip_rate(series: pd.Series):
        vals = series.astype(str).values
        if len(vals) < 2:
            return 0.0
        flips = int((vals[1:] != vals[:-1]).sum())
        return float(flips / max(len(vals) - 1, 1) * 252.0)

    out["flip_rates"] = {
        "combined": _flip_rate(wdf["combined_state"]),
        "macro": _flip_rate(wdf["macro_state"]),
        "fast": _flip_rate(wdf["fast_state"]),
    }
    return out


def _mapped_label_means(model, feature_names, state_map, label_names):
    means = np.asarray(model.means_, dtype=float)
    label_means = {}
    for li, lname in enumerate(label_names):
        mapped_states = [orig for orig, lbl in state_map.items() if int(lbl) == li and int(orig) < means.shape[0]]
        if len(mapped_states) == 0:
            label_means[lname] = np.full(means.shape[1], np.nan)
        else:
            label_means[lname] = np.nanmean(means[mapped_states, :], axis=0)
    return label_means


def _pick_feature_subset(feature_names: List[str], priority: List[str], max_items: int = 8):
    chosen = []
    for f in priority:
        if f in feature_names and f not in chosen:
            chosen.append(f)
    if len(chosen) < max_items:
        for f in feature_names:
            if f not in chosen:
                chosen.append(f)
            if len(chosen) >= max_items:
                break
    return chosen[:max_items]


def load_model_diagnostics(days: int = 252, view: str = "smoothed"):
    model_path = MARKET_MODELS_DIR / "hmm_regime_models.joblib"
    if not model_path.exists():
        return {"available": False, "message": "model file missing"}

    bundle = joblib.load(model_path)
    if any(k not in bundle for k in ["slow_hmm", "fast_hmm_durable", "fast_hmm_fragile"]):
        return {"available": False, "message": "model bundle schema invalid"}

    slow_model = bundle["slow_hmm"]
    fast_d_model = bundle["fast_hmm_durable"]
    fast_f_model = bundle["fast_hmm_fragile"]
    slow_map, dur_map, fra_map = _resolve_bundle_maps(bundle)

    slow_labels = ["Durable", "Fragile"]
    fast_labels = ["Calm", "Choppy", "Stress"]

    slow_inv = _invert_map(slow_map)
    dur_inv = _invert_map(dur_map)
    fra_inv = _invert_map(fra_map)

    slow_tm = np.asarray(slow_model.transmat_, dtype=float)
    dur_tm = np.asarray(fast_d_model.transmat_, dtype=float)
    fra_tm = np.asarray(fast_f_model.transmat_, dtype=float)

    slow_expected = _expected_duration_from_transmat(slow_tm)
    dur_expected = _expected_duration_from_transmat(dur_tm)
    fra_expected = _expected_duration_from_transmat(fra_tm)

    state = load_latest_state()
    # Pick current state based on selected view
    if view == "raw":
        current_macro = str(state.get("raw_macro_state", state.get("macro_state", "Durable")))
        current_fast = str(state.get("raw_fast_state", state.get("fast_state", "Calm")))
    elif view == "combined":
        current_macro = str(state.get("adaptive_macro_state", state.get("macro_state", "Durable")))
        current_fast = str(state.get("adaptive_fast_state", state.get("fast_state", "Calm")))
    else:
        current_macro = str(state.get("macro_state", "Durable"))
        current_fast = str(state.get("fast_state", "Calm"))
    current_macro_label_idx = slow_labels.index(current_macro) if current_macro in slow_labels else 0
    current_macro_model_idx = slow_inv.get(current_macro_label_idx, [0])[0]

    fast_map = dur_map if current_macro == "Durable" else fra_map
    fast_inv = dur_inv if current_macro == "Durable" else fra_inv
    fast_tm = dur_tm if current_macro == "Durable" else fra_tm
    fast_expected = dur_expected if current_macro == "Durable" else fra_expected

    current_fast_label_idx = fast_labels.index(current_fast) if current_fast in fast_labels else 0
    current_fast_model_idx = fast_inv.get(current_fast_label_idx, [0])[0]

    macro_next = _aggregate_label_transition(slow_tm[current_macro_model_idx], slow_map, 2)
    fast_next = _aggregate_label_transition(fast_tm[current_fast_model_idx], fast_map, 3)

    combined_next = []
    for mi, mlabel in enumerate(slow_labels):
        for fi, flabel in enumerate(fast_labels):
            combined_next.append({"state": f"{mlabel}-{flabel}", "probability": float(macro_next[mi] * fast_next[fi])})
    combined_next = sorted(combined_next, key=lambda x: x["probability"], reverse=True)

    # "If regime changes" — exclude current combined state, renormalize
    current_combined = f"{current_macro}-{current_fast}"
    change_total = 0.0
    combined_if_change = []
    for item in combined_next:
        p = item["probability"] if item["state"] != current_combined else 0.0
        combined_if_change.append({"state": item["state"], "probability": p})
        change_total += p
    if change_total > 0:
        for item in combined_if_change:
            item["probability"] = item["probability"] / change_total
    combined_if_change = sorted(combined_if_change, key=lambda x: x["probability"], reverse=True)

    hist_df_all = _timeline_df(days=None)
    hist_df_range = _timeline_df(days=None if int(days) <= 0 else int(days))

    occ_60 = _window_stats(hist_df_all, 60)
    occ_252 = _window_stats(hist_df_all, 252)

    entropy_series = []
    if len(hist_df_range) > 0:
        for _, r in hist_df_range.iterrows():
            macro_entropy = _normalized_entropy([float(r["p_durable_smooth"]), float(r["p_fragile_smooth"])])
            fast_entropy = _normalized_entropy(
                [float(r["p_calm_smooth"]), float(r["p_choppy_smooth"]), float(r["p_stress_smooth"])]
            )
            entropy_series.append(
                {
                    "date": pd.Timestamp(r["Date"]).strftime("%Y-%m-%d"),
                    "macro_entropy": float(macro_entropy),
                    "fast_entropy": float(fast_entropy),
                    "combined_entropy": float((macro_entropy + fast_entropy) / 2.0),
                }
            )

    slow_features = list(bundle.get("slow_feature_names", []))
    fast_features = list(bundle.get("fast_feature_names", []))
    slow_priority = [
        "vix_percentile_252_norm",
        "vix_relative_252_norm",
        "rv_252_norm",
        "max_drawdown_252_norm",
        "downside_vol_252_norm",
        "time_under_water_252_norm",
    ]
    fast_priority = [
        "rv_20_norm",
        "downside_vol_20_norm",
        "vol_of_vol_60_norm",
        "hl_range_norm",
        "return_5_norm",
    ]
    slow_subset = _pick_feature_subset(slow_features, slow_priority, max_items=8)
    fast_subset = _pick_feature_subset(fast_features, fast_priority, max_items=8)

    slow_label_means = _mapped_label_means(slow_model, slow_features, slow_map, ["Durable", "Fragile"])
    dur_label_means = _mapped_label_means(fast_d_model, fast_features, dur_map, ["Calm", "Choppy", "Stress"])
    fra_label_means = _mapped_label_means(fast_f_model, fast_features, fra_map, ["Calm", "Choppy", "Stress"])

    slow_rows = []
    for f in slow_subset:
        fi = slow_features.index(f)
        d = float(slow_label_means["Durable"][fi])
        fr = float(slow_label_means["Fragile"][fi])
        slow_rows.append({"feature": f, "Durable": d, "Fragile": fr, "delta_fragile_minus_durable": fr - d})

    dur_rows = []
    for f in fast_subset:
        fi = fast_features.index(f)
        calm = float(dur_label_means["Calm"][fi])
        choppy = float(dur_label_means["Choppy"][fi])
        stress = float(dur_label_means["Stress"][fi])
        dur_rows.append({"feature": f, "Calm": calm, "Choppy": choppy, "Stress": stress, "delta_stress_minus_calm": stress - calm})

    fra_rows = []
    for f in fast_subset:
        fi = fast_features.index(f)
        calm = float(fra_label_means["Calm"][fi])
        choppy = float(fra_label_means["Choppy"][fi])
        stress = float(fra_label_means["Stress"][fi])
        fra_rows.append({"feature": f, "Calm": calm, "Choppy": choppy, "Stress": stress, "delta_stress_minus_calm": stress - calm})

    diagnostics = {
        "available": True,
        "model_file": str(model_path),
        "model_info": {
            "slow_states": int(slow_model.n_components),
            "fast_durable_states": int(fast_d_model.n_components),
            "fast_fragile_states": int(fast_f_model.n_components),
            "slow_covariance_type": str(getattr(slow_model, "covariance_type", "unknown")),
            "fast_covariance_type": str(getattr(fast_d_model, "covariance_type", "unknown")),
            "slow_feature_count": len(bundle.get("slow_feature_names", [])),
            "fast_feature_count": len(bundle.get("fast_feature_names", [])),
            "slow_feature_names": list(bundle.get("slow_feature_names", [])),
            "fast_feature_names": list(bundle.get("fast_feature_names", [])),
        },
        "state_mapping": {
            "slow": bundle.get("state_map", {}).get("slow", {}),
            "fast_durable": bundle.get("state_map", {}).get("fast_durable", {}),
            "fast_fragile": bundle.get("state_map", {}).get("fast_fragile", {}),
        },
        "transitions": {
            "slow": slow_tm.tolist(),
            "fast_durable": dur_tm.tolist(),
            "fast_fragile": fra_tm.tolist(),
        },
        "expected_duration_days": {
            "slow": {slow_labels[i]: float(slow_expected[slow_inv.get(i, [0])[0]]) for i in range(len(slow_labels))},
            "fast_durable": {fast_labels[i]: float(dur_expected[dur_inv.get(i, [0])[0]]) for i in range(len(fast_labels))},
            "fast_fragile": {fast_labels[i]: float(fra_expected[fra_inv.get(i, [0])[0]]) for i in range(len(fast_labels))},
        },
        "empirical_duration_days": {
            "macro": _empirical_durations("macro_state"),
            "fast": _empirical_durations("fast_state"),
        },
        "diagnostics_view": view,
        "switch_risk": {
            "macro_switch_next_day": float(1.0 - slow_tm[current_macro_model_idx, current_macro_model_idx]),
            "fast_switch_next_day": float(1.0 - fast_tm[current_fast_model_idx, current_fast_model_idx]),
            "macro_context": current_macro,
        },
        "next_regime_forecast": combined_next,
        "next_regime_if_change": combined_if_change,
        "occupancy_60": occ_60,
        "occupancy_252": occ_252,
        "flip_rates_60": occ_60.get("flip_rates", {}),
        "flip_rates_252": occ_252.get("flip_rates", {}),
        "entropy_series": entropy_series,
        "emission_summary": {
            "slow": slow_rows,
            "fast_durable": dur_rows,
            "fast_fragile": fra_rows,
        },
    }
    return diagnostics


def load_ops_summary():
    ops_path = OUTPUT_DIR / "daily_operations" / "daily_ops_latest.json"
    if not ops_path.exists():
        ops_path = OUTPUT_DIR / "daily_ops_latest.json"  # legacy fallback
    health_path = OUTPUT_DIR / "health_checks" / "system_health_latest.json"
    if not health_path.exists():
        health_path = OUTPUT_DIR / "system_health_latest.json"  # legacy fallback
    ops = _read_json(ops_path, {})
    health = _read_json(health_path, {})
    return {
        "ops_available": bool(len(ops) > 0),
        "health_available": bool(len(health) > 0),
        "ops": ops,
        "system_health": health,
    }


def load_behavior_backfill_activity(limit=60):
    runs = _tail_jsonl(BEHAVIOR_LOGS_DIR / "behavior_automation_runs.jsonl", limit=max(100, int(limit)))
    if len(runs) == 0:
        return {"available": False, "recent_runs": []}

    latest = runs[-1]
    last_backfill = latest if latest.get("datasets", {}).get("missing_dates_count", 0) > 0 else None

    recent = []
    for r in runs[-int(limit) :]:
        datasets = r.get("datasets", {})
        recent.append(
            {
                "timestamp": r.get("timestamp"),
                "status": r.get("status"),
                "mode": "backfill" if datasets.get("missing_dates_count", 0) > 0 else "daily",
                "auto_backfill_triggered": datasets.get("missing_dates_count", 0) > 0,
                "auto_backfill_missing_days": _safe_int(datasets.get("missing_dates_count"), 0),
                "latest_data_date": datasets.get("latest_data_date"),
                "retrained": r.get("retrained", False)
            }
        )

    return {
        "available": True,
        "latest_run": recent[-1] if recent else None,
        "last_backfill_run": last_backfill,
        "recent_runs": recent,
        "source_max_dates": {
            "behavior_data": latest.get("datasets", {}).get("latest_data_date"),
        },
    }

def load_backfill_activity(limit=60):
    runs = _tail_jsonl(LOGS_DIR / "daily_inference_orchestrator.jsonl", limit=max(100, int(limit)))
    if len(runs) == 0:
        return {"available": False, "recent_runs": []}

    latest = runs[-1]
    backfill_runs = [r for r in runs if str(r.get("mode", "")).lower() == "backfill" or bool(r.get("auto_backfill_triggered"))]
    last_backfill = backfill_runs[-1] if backfill_runs else None

    recent = []
    for r in runs[-int(limit) :]:
        recent.append(
            {
                "timestamp": r.get("timestamp"),
                "run_date": r.get("run_date"),
                "asof_date": r.get("asof_date"),
                "mode": r.get("mode"),
                "status": r.get("status"),
                "auto_backfill_triggered": bool(r.get("auto_backfill_triggered", False)),
                "auto_backfill_missing_days": _safe_int(r.get("auto_backfill_missing_days"), 0),
                "missing_dates_count": _safe_int(r.get("missing_dates_count"), 0),
                "processed_dates_count": _safe_int(r.get("processed_dates_count"), 0),
                "requested_start_date": r.get("requested_start_date"),
                "requested_end_date": r.get("requested_end_date"),
                "error": r.get("error"),
            }
        )

    latest_missing = _safe_int((latest or {}).get("auto_backfill_missing_days"), 0)
    latest_mode = str((latest or {}).get("mode", "")).lower()
    if latest_mode == "backfill":
        summary_reason = "Backfill executed on latest run."
    elif latest_missing <= 0:
        summary_reason = "No missing trailing trading days after latest timeline date."
    elif latest_missing < 2:
        summary_reason = "Missing trailing days are below auto-backfill trigger threshold (default 2)."
    else:
        summary_reason = "Missing trailing days exist but latest run stayed in single mode."

    def _max_date(path: Path):
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path, parse_dates=["Date"])
            if "Date" not in df.columns or len(df) == 0:
                return None
            return pd.to_datetime(df["Date"]).max().strftime("%Y-%m-%d")
        except Exception:
            return None

    source_max_dates = {
        "market": _max_date(BASE_DIR / "data" / "processed" / "market_data_historical.csv"),
        "final_features": _max_date(BASE_DIR / "features" / "final_features_matrix.csv"),
        "slow_features": _max_date(BASE_DIR / "features" / "slow_features_matrix.csv"),
        "fast_features": _max_date(BASE_DIR / "features" / "fast_features_matrix.csv"),
        "timeline": _max_date(REGIME_TIMELINE_FILE),
    }
    timeline_df = _timeline_base_df()
    source_max_dates["timeline_current"] = (
        pd.to_datetime(timeline_df["Date"]).max().strftime("%Y-%m-%d") if len(timeline_df) > 0 else None
    )

    return {
        "available": True,
        "latest_run": latest,
        "last_backfill_run": last_backfill,
        "summary_reason": summary_reason,
        "source_max_dates": source_max_dates,
        "recent_runs": recent,
    }


def load_regime_changes(limit=20):
    timeline = load_timeline(days=0)
    if len(timeline) == 0:
        return []
    changes = []
    prev = None
    for row in timeline:
        cur = row.get("combined_state")
        if prev is None:
            prev = row
            continue
        if cur != prev.get("combined_state"):
            changes.append(
                {
                    "date": row.get("date"),
                    "from": prev.get("combined_state"),
                    "to": cur,
                    "confidence": _safe_float(row.get("confidence"), np.nan),
                    "macro_confidence": _safe_float(row.get("macro_confidence"), np.nan),
                    "fast_confidence": _safe_float(row.get("fast_confidence"), np.nan),
                    "severity": "CRITICAL" if "Stress" in str(cur) else ("WARNING" if "Choppy" in str(cur) else "INFO"),
                }
            )
        prev = row
    return list(reversed(changes[-int(limit) :]))


@app.get("/")
def root():
    return {
        "name": "Market Regime API",
        "version": "1.1.0",
        "docs": "/docs",
        "endpoints": {
            "current": "/api/current-regime",
            "timeline": "/api/timeline",
            "metrics": "/api/metrics",
            "probabilities": "/api/probabilities",
            "health": "/api/health",
            "ops": "/api/ops",
            "backfill": "/api/backfill",
            "changes": "/api/regime-changes",
            "model_diagnostics": "/api/model-diagnostics",
        },
    }


@app.get("/api/current-regime")
def get_current_regime():
    return {"success": True, "data": load_latest_state()}


@app.get("/api/timeline")
def get_timeline(days: Optional[int] = None):
    # days <= 0 or None means full/max range.
    timeline = load_timeline(days=None if (days is None or int(days) <= 0) else int(days))
    return {"success": True, "count": len(timeline), "data": timeline}


@app.get("/api/metrics")
def get_metrics():
    return {"success": True, "data": load_economic_metrics()}


@app.get("/api/probabilities")
def get_probabilities():
    state = load_latest_state()
    data = {
        "date": state["date"],
        "macro": {"Durable": state["p_durable_smooth"], "Fragile": state["p_fragile_smooth"]},
        "fast": {
            "Calm": state["p_calm_smooth"],
            "Choppy": state["p_choppy_smooth"],
            "Stress": state["p_stress_smooth"],
        },
        "confidence": state["confidence"],
        "macro_confidence": state["macro_confidence"],
        "fast_confidence": state["fast_confidence"],
    }
    # Raw probabilities
    if "p_fragile_raw" in state:
        data["raw_macro"] = {"Durable": state["p_durable_raw"], "Fragile": state["p_fragile_raw"]}
        data["raw_fast"] = {
            "Calm": state["p_calm_raw"],
            "Choppy": state["p_choppy_raw"],
            "Stress": state["p_stress_raw"],
        }
    # Adaptive-α probabilities
    if "p_fragile_adaptive" in state:
        data["adaptive_macro"] = {"Durable": state["p_durable_adaptive"], "Fragile": state["p_fragile_adaptive"]}
        data["adaptive_fast"] = {
            "Calm": state["p_calm_adaptive"],
            "Choppy": state["p_choppy_adaptive"],
            "Stress": state["p_stress_adaptive"],
        }
    return {"success": True, "data": data}


@app.get("/api/ops")
def get_ops():
    return {"success": True, "data": load_ops_summary()}


@app.get("/api/backfill")
def get_backfill(limit: int = 30):
    return {"success": True, "data": load_backfill_activity(limit=limit)}

@app.get("/api/behavior-backfill")
def get_behavior_backfill(limit: int = 30):
    return {"success": True, "data": load_behavior_backfill_activity(limit=limit)}


@app.get("/api/behavior-changes")
def get_behavior_changes(limit: int = 20):
    """Detect transitions in behavior regime states (slow or fast)."""
    df = _load_behavior_predictions(days=None)
    if df.empty:
        return {"success": True, "count": 0, "data": []}
    changes = []
    prev_slow = None
    prev_fast = None
    for _, row in df.iterrows():
        cur_slow = str(row.get("behavior_slow_state", "Unknown"))
        cur_fast = str(row.get("behavior_fast_state", "Unknown"))
        date_str = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")
        if prev_slow is not None:
            if cur_slow != prev_slow:
                changes.append({
                    "date": date_str,
                    "layer": "slow",
                    "from": prev_slow,
                    "to": cur_slow,
                    "confidence": _safe_float(row.get("behavior_slow_confidence"), 0),
                    "severity": "CRITICAL" if cur_slow == "Noisy" else ("WARNING" if cur_slow == "Mean-Reverting" else "INFO"),
                })
            if cur_fast != prev_fast:
                changes.append({
                    "date": date_str,
                    "layer": "fast",
                    "from": prev_fast,
                    "to": cur_fast,
                    "confidence": _safe_float(row.get("behavior_fast_confidence"), 0),
                    "severity": "CRITICAL" if cur_fast == "Noisy" else ("WARNING" if cur_fast == "Mean-Reverting" else "INFO"),
                })
        prev_slow = cur_slow
        prev_fast = cur_fast
    result = list(reversed(changes[-int(limit):]))
    return {"success": True, "count": len(result), "data": result}


@app.get("/api/regime-changes")
def get_regime_changes(limit: int = 20):
    return {"success": True, "count": int(limit), "data": load_regime_changes(limit=limit)}


@app.get("/api/model-diagnostics")
def get_model_diagnostics(days: int = 252, view: str = "smoothed"):
    return {"success": True, "data": load_model_diagnostics(days=days, view=view)}


@app.get("/api/health")
def get_health():
    try:
        state = load_latest_state()
        ts = state.get("timestamp")
        if ts is None:
            return {
                "status": "warning",
                "message": "Latest state has no timestamp",
                "last_update": None,
                "data_freshness": "unknown",
                "models_loaded": (MARKET_MODELS_DIR / "hmm_regime_models.joblib").exists(),
            }

        last_update = datetime.fromisoformat(str(ts))
        hours_old = (datetime.now() - last_update).total_seconds() / 3600.0
        freshness = "stale" if hours_old > 36 else "current"
        status = "warning" if freshness == "stale" else "healthy"

        return {
            "status": status,
            "last_update": ts,
            "data_freshness": freshness,
            "hours_since_update": round(hours_old, 1),
            "models_loaded": (MARKET_MODELS_DIR / "hmm_regime_models.joblib").exists(),
            "current_regime": state.get("combined_state", "Unknown"),
        }
    except Exception as exc:
        logger.error("Error in health endpoint: %s", exc)
        return {"status": "error", "message": str(exc)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
@app.get("/api/advisory")
def get_advisory():
    try:
        from backend.decision_engine import AdvisoryDecisionEngine
    except ModuleNotFoundError:
        from decision_engine import AdvisoryDecisionEngine
    import datetime
    
    try:
        timeline_file = os.getenv("REGIME_TIMELINE_FILE", str(MARKET_FEATURES_DIR / "regime_timeline_history.csv"))
        mrkt_df = pd.read_csv(timeline_file)
        behav_df = pd.read_csv(BEHAVIOR_OUTPUT_DIR / "behavior_regime_predictions.csv")
        
        mrkt = mrkt_df.iloc[-1]
        behav = behav_df.iloc[-1]
        
        # Calculate time since last flip for fast behavior
        try:
            states = behav_df["behavior_fast_state"].values
            curr_state = states[-1]
            idx = len(states) - 2
            while idx >= 0 and states[idx] == curr_state:
                idx -= 1
            time_since_last_fast_flip = len(states) - 1 - idx
        except Exception:
            time_since_last_fast_flip = 10
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": "Missing model outputs. Run pipelines first.", "details": str(e)}

    market_state = mrkt.get("adaptive_combined_state", "Unknown")
    market_confidence = float(mrkt.get("combined_confidence", mrkt.get("confidence", 0.5)))

    slow_state = behav.get("behavior_slow_state", "")
    slow_conf = float(behav.get("behavior_slow_confidence", 0.0))
    fast_state = behav.get("behavior_fast_state", "")
    fast_conf = float(behav.get("behavior_fast_confidence", 0.0))
    fast_gap = float(behav.get("behavior_fast_prob_gap", 0.0))
    hybrid_action = behav.get("hybrid_action", "")

    # Calculate simulated dynamic stock features
    stock_data = {
        "rsi": 65.0,  # TODO: Calculate dynamically
        "momentum_20d": 1.5,
        "relative_strength": "outperform"
    }

    policy = AdvisoryDecisionEngine.evaluate(
        market_regime=market_state,
        market_confidence=market_confidence,
        behavior_slow=slow_state,
        slow_conf=slow_conf,
        behavior_fast=fast_state,
        fast_conf=fast_conf,
        fast_gap=fast_gap,
        time_since_last_fast_flip=time_since_last_fast_flip,
        stock_data=stock_data
    )

    return {
        "market_regime": market_state,
        "behavior_slow": slow_state,
        "behavior_fast": fast_state,
        "behavior_fast_confidence": fast_conf,
        "hybrid_behavior": hybrid_action,
        "policy": {
            "composite_score": policy["composite_score"],
            "final_action": policy["final_action"],
            "exposure_limit": policy["exposure_limit"],
            "risk_status": policy["risk_status"],
            "tactical_override": policy["tactical_override"],
            "stock_filter": policy["stock_filter"],
            "reasoning": policy["reasoning"]
        },
        "stock_data": stock_data

    }


# ---------------------------------------------------------------------------
# Behavior Regime endpoints
# ---------------------------------------------------------------------------

BEHAVIOR_PREDICTIONS_FILE = BEHAVIOR_OUTPUT_DIR / "behavior_regime_predictions.csv"


def _load_behavior_predictions(days: Optional[int] = None) -> pd.DataFrame:
    if not BEHAVIOR_PREDICTIONS_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(BEHAVIOR_PREDICTIONS_FILE, parse_dates=["Date"])
    df = df.sort_values("Date")
    if days and int(days) > 0:
        df = df.tail(int(days))
    return df


@app.get("/api/behavior-timeline")
def get_behavior_timeline(days: int = 252):
    """Return behavior regime timeline for charting."""
    df = _load_behavior_predictions(days=days)
    if df.empty:
        return {"success": True, "data": []}
    records = []
    for _, row in df.iterrows():
        records.append({
            "date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
            "behavior_slow_state": str(row.get("behavior_slow_state", "Unknown")),
            "behavior_fast_state": str(row.get("behavior_fast_state", "Unknown")),
            "behavior_slow_confidence": _safe_float(row.get("behavior_slow_confidence"), 0),
            "behavior_fast_confidence": _safe_float(row.get("behavior_fast_confidence"), 0),
            "behavior_slow_prob_gap": _safe_float(row.get("behavior_slow_prob_gap"), 0),
            "behavior_fast_prob_gap": _safe_float(row.get("behavior_fast_prob_gap"), 0),
            "hybrid_action": str(row.get("hybrid_action", "")),
        })
    return {"success": True, "data": records}


@app.get("/api/behavior-diagnostics")
def get_behavior_diagnostics(days: int = 252):
    """Return behavior regime diagnostics: durations, distributions, divergence, quality."""
    df = _load_behavior_predictions(days=days)
    if df.empty:
        return {"success": True, "data": {"available": False}}

    full_df = _load_behavior_predictions(days=None)

    # --- Current state ---
    latest = df.iloc[-1]
    current = {
        "date": pd.Timestamp(latest["Date"]).strftime("%Y-%m-%d"),
        "slow_state": str(latest.get("behavior_slow_state", "Unknown")),
        "fast_state": str(latest.get("behavior_fast_state", "Unknown")),
        "slow_confidence": _safe_float(latest.get("behavior_slow_confidence"), 0),
        "fast_confidence": _safe_float(latest.get("behavior_fast_confidence"), 0),
        "slow_prob_gap": _safe_float(latest.get("behavior_slow_prob_gap"), 0),
        "fast_prob_gap": _safe_float(latest.get("behavior_fast_prob_gap"), 0),
        "hybrid_action": str(latest.get("hybrid_action", "")),
    }

    # --- State distribution (window) ---
    slow_dist = df["behavior_slow_state"].value_counts(normalize=True).to_dict()
    fast_dist = df["behavior_fast_state"].value_counts(normalize=True).to_dict()

    # --- Hybrid distribution ---
    hybrid_dist = df["hybrid_action"].value_counts(normalize=True).to_dict()

    # --- Empirical durations ---
    def _streak_stats(series):
        vals = series.tolist()
        streaks = {}
        current_val = vals[0]
        count = 1
        for i in range(1, len(vals)):
            if vals[i] == current_val:
                count += 1
            else:
                streaks.setdefault(str(current_val), []).append(count)
                current_val = vals[i]
                count = 1
        # exclude ongoing streak
        result = {}
        for state, durs in streaks.items():
            if len(durs) == 0:
                continue
            import statistics
            result[state] = {
                "n": len(durs),
                "mean": round(sum(durs) / len(durs), 1),
                "median": round(statistics.median(durs), 1),
                "min": min(durs),
                "max": max(durs),
            }
        return result

    slow_durations = _streak_stats(full_df["behavior_slow_state"])
    fast_durations = _streak_stats(full_df["behavior_fast_state"])

    # --- Fast vs Slow divergence ---
    divergence_mask = df["behavior_fast_state"] != df["behavior_slow_state"]
    divergence_pct = float(divergence_mask.mean()) if len(df) > 0 else 0.0

    # Average divergence duration
    div_runs = []
    run_len = 0
    for v in divergence_mask:
        if v:
            run_len += 1
        else:
            if run_len > 0:
                div_runs.append(run_len)
            run_len = 0
    if run_len > 0:
        div_runs.append(run_len)
    avg_div_duration = round(sum(div_runs) / len(div_runs), 1) if div_runs else 0.0

    # --- Confidence stats ---
    avg_fast_conf = float(df["behavior_fast_confidence"].mean()) if "behavior_fast_confidence" in df.columns else 0.0
    avg_slow_conf = float(df["behavior_slow_confidence"].mean()) if "behavior_slow_confidence" in df.columns else 0.0
    avg_fast_gap = float(df["behavior_fast_prob_gap"].mean()) if "behavior_fast_prob_gap" in df.columns else 0.0

    # --- Fast override usage stats (aligned with decision engine gate) ---
    try:
        from backend.decision_engine import AdvisoryDecisionEngine
    except ModuleNotFoundError:
        from decision_engine import AdvisoryDecisionEngine

    fast_conf_threshold = AdvisoryDecisionEngine.FAST_OVERRIDE_MIN_CONFIDENCE
    fast_persist_threshold = AdvisoryDecisionEngine.FAST_OVERRIDE_MIN_PERSIST_BARS

    fast_states = df["behavior_fast_state"].astype(str).tolist()
    fast_persist_days = []
    streak = 0
    prev = None
    for state in fast_states:
        if state == prev:
            streak += 1
        else:
            streak = 1
            prev = state
        fast_persist_days.append(streak)

    fast_conf_mask = df["behavior_fast_confidence"] >= fast_conf_threshold
    fast_persist_mask = pd.Series(fast_persist_days, index=df.index) >= fast_persist_threshold
    mismatch_mask = divergence_mask

    fast_override_used_mask = mismatch_mask & fast_conf_mask & fast_persist_mask
    fast_override_ignored_mask = mismatch_mask & (~fast_override_used_mask)

    fast_override_used_pct = float(fast_override_used_mask.mean()) if len(df) > 0 else 0.0
    fast_override_ignored_pct = float(fast_override_ignored_mask.mean()) if len(df) > 0 else 0.0
    mismatch_pct = float(mismatch_mask.mean()) if len(df) > 0 else 0.0

    fast_override_used_when_mismatch_pct = (
        float(fast_override_used_mask.sum() / mismatch_mask.sum())
        if mismatch_mask.sum() > 0
        else 0.0
    )

    # --- Transition counts ---
    def _transition_matrix(series):
        vals = series.tolist()
        states = sorted(set(vals))
        trans = {s: {t: 0 for t in states} for s in states}
        for i in range(1, len(vals)):
            trans[str(vals[i - 1])][str(vals[i])] += 1
        # Normalize
        matrix = {}
        for s in states:
            total = sum(trans[s].values())
            matrix[s] = {t: round(trans[s][t] / total, 4) if total > 0 else 0.0 for t in states}
        return matrix

    slow_transitions = _transition_matrix(full_df["behavior_slow_state"])
    fast_transitions = _transition_matrix(full_df["behavior_fast_state"])

    # --- Behavior quality score ---
    # Based on: state separation, stability, confidence, entropy
    score_components = []
    # 1. Confidence (higher = better) - contributes 0-3 points
    score_components.append(min(avg_fast_conf * 3, 3.0))
    # 2. Low entropy = good separation - contributes 0-2 points
    fast_probs = [fast_dist.get(s, 0) for s in ["Trending", "Mean-Reverting", "Noisy"]]
    fast_entropy = normalizedEntropy(fast_probs) if fast_probs else 1.0
    score_components.append((1 - fast_entropy) * 2)
    # 3. Stability (low flip rate = better) - contributes 0-2 points
    total_fast_streaks = sum(len([]) or s.get("n", 0) for s in fast_durations.values())
    avg_fast_dur = sum(s.get("mean", 0) * s.get("n", 0) for s in fast_durations.values()) / max(total_fast_streaks, 1)
    score_components.append(min(avg_fast_dur / 30, 2.0))
    # 4. Prob gap (higher = cleaner separation) - contributes 0-2 points
    score_components.append(min(avg_fast_gap * 3, 2.0))
    # 5. Low divergence (manageable) - contributes 0-1 point
    score_components.append(1.0 - min(divergence_pct, 1.0))
    quality_score = round(sum(score_components), 1)

    # --- Cross-layer alignment (if market timeline available) ---
    alignment = None
    try:
        market_df = _timeline_base_df()
        if not market_df.empty and len(df) > 0:
            market_df["date_str"] = market_df["Date"].dt.strftime("%Y-%m-%d")
            behav_dates = set(pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"))
            merged = market_df[market_df["date_str"].isin(behav_dates)].copy()
            if len(merged) > 0:
                # Simple check: stress alignment
                merged_b = df.copy()
                merged_b["date_str"] = pd.to_datetime(merged_b["Date"]).dt.strftime("%Y-%m-%d")
                m = merged.merge(merged_b[["date_str", "behavior_slow_state", "behavior_fast_state"]], on="date_str", how="inner")
                if len(m) > 0:
                    # Market stress + behavior noisy/mean-reverting = aligned
                    market_stress = m["fast_state"].isin(["Stress", "Choppy"])
                    behav_stress = m["behavior_fast_state"].isin(["Mean-Reverting", "Noisy"])
                    market_calm = m["fast_state"] == "Calm"
                    behav_calm = m["behavior_fast_state"] == "Trending"
                    aligned = (market_stress & behav_stress) | (market_calm & behav_calm)
                    alignment = {
                        "alignment_pct": round(float(aligned.mean()) * 100, 1),
                        "conflict_pct": round(float((~aligned).mean()) * 100, 1),
                        "sample_size": len(m),
                    }
    except Exception:
        pass

    # --- Behavior economic metrics ---
    behav_econ = {}
    try:
        market_df = pd.read_csv(
            BASE_DIR / "data" / "processed" / "market_data_historical.csv",
            parse_dates=["Date"],
        )
        market_df = market_df.sort_values("Date")
        market_df["return"] = market_df["Close"].pct_change()
        behav_full = _load_behavior_predictions(days=None)
        behav_full["date_str"] = pd.to_datetime(behav_full["Date"]).dt.strftime("%Y-%m-%d")
        market_df["date_str"] = market_df["Date"].dt.strftime("%Y-%m-%d")
        m = market_df.merge(behav_full[["date_str", "behavior_slow_state", "behavior_fast_state"]], on="date_str", how="inner")
        if len(m) > 10:
            for col_name, label in [("behavior_slow_state", "slow"), ("behavior_fast_state", "fast")]:
                group_metrics = {}
                for state, grp in m.groupby(col_name):
                    rets = grp["return"].dropna()
                    if len(rets) < 5:
                        continue
                    ann_ret = float(rets.mean() * 252)
                    vol = float(rets.std() * np.sqrt(252))
                    sharpe = float(ann_ret / vol) if vol > 0 else 0.0
                    cum = (1 + rets).cumprod()
                    max_dd = float(((cum / cum.cummax()) - 1).min())
                    group_metrics[str(state)] = {
                        "ann_return": round(ann_ret, 4),
                        "volatility": round(vol, 4),
                        "sharpe": round(sharpe, 2),
                        "max_dd": round(max_dd, 4),
                        "days": len(rets),
                        "neg_days_pct": round(float((rets < 0).mean()), 3),
                    }
                behav_econ[label] = group_metrics
    except Exception:
        pass

    return {
        "success": True,
        "data": {
            "available": True,
            "current": current,
            "state_distribution": {"slow": slow_dist, "fast": fast_dist},
            "hybrid_distribution": hybrid_dist,
            "durations": {"slow": slow_durations, "fast": fast_durations},
            "divergence": {
                "mismatch_pct": round(divergence_pct * 100, 1),
                "avg_duration_days": avg_div_duration,
            },
            "confidence_stats": {
                "avg_fast_confidence": round(avg_fast_conf, 3),
                "avg_slow_confidence": round(avg_slow_conf, 3),
                "avg_fast_prob_gap": round(avg_fast_gap, 3),
                "fast_override_conf_threshold": round(float(fast_conf_threshold), 2),
                "fast_override_persistence_days": int(fast_persist_threshold),
                "fast_override_ignored_pct": round(fast_override_ignored_pct * 100, 1),
                "fast_override_used_pct": round(fast_override_used_pct * 100, 1),
                "fast_override_used_when_mismatch_pct": round(fast_override_used_when_mismatch_pct * 100, 1),
                "behavior_mismatch_pct": round(mismatch_pct * 100, 1),
            },
            "transitions": {"slow": slow_transitions, "fast": fast_transitions},
            "quality_score": quality_score,
            "alignment": alignment,
            "economic_metrics": behav_econ,
            "window_days": len(df),
        },
    }


def _normalized_entropy_py(values):
    """Python-side normalized entropy for quality score."""
    arr = [max(float(v), 1e-12) for v in values]
    s = sum(arr)
    if s <= 0:
        return 0
    p = [v / s for v in arr]
    import math
    h = -sum(pi * math.log(pi) for pi in p)
    hmax = math.log(len(p)) if len(p) > 1 else 1.0
    return h / hmax if hmax > 0 else 0


def normalizedEntropy(values):
    """Wrapper matching the frontend function name."""
    return _normalized_entropy_py(values)

# ==========================================
# ADMIN DASHBOARD ROUTES
# ==========================================
@app.get("/api/admin/system-status")
def get_admin_system_status():
    import os, time
    from pathlib import Path

    # Files to track for Pipeline Health
    target_files = {
        "market_data_historical.csv": MARKET_FEATURES_DIR / "market_data_historical.csv",
        "final_features_matrix.csv": MARKET_FEATURES_DIR / "final_features_matrix.csv",
        "regime_timeline_history.csv": MARKET_FEATURES_DIR / "regime_timeline_history.csv",
        "behavior_regime_predictions.csv": BEHAVIOR_OUTPUT_DIR / "behavior_regime_predictions.csv",
        "daily_inference_state.json": MARKET_FEATURES_DIR / "daily_inference_state.json",
        "feature_build_metadata.json": MARKET_FEATURES_DIR / "feature_build_metadata.json"
    }

    files_health = []
    for name, path in target_files.items():
        if path.exists():
            mod_time = os.path.getmtime(path)
            files_health.append({
                "file": name,
                "exists": True,
                "last_modified": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mod_time))
            })
        else:
            files_health.append({
                "file": name,
                "exists": False,
                "last_modified": None
            })

    # Read latest behavior CSV
    stability = {"max_slow_conf": 0, "max_fast_conf": 0, "mismatch_rate": 0}
    try:
        bs_path = BEHAVIOR_OUTPUT_DIR / "behavior_regime_predictions.csv"
        if bs_path.exists():
            df_b = pd.read_csv(bs_path)
            if not df_b.empty:
                stability["max_slow_conf"] = float(df_b.get("behavior_slow_confidence", 0).max())
                stability["max_fast_conf"] = float(df_b.get("behavior_fast_confidence", 0).max())
                mismatches = (df_b["behavior_fast_state"] != df_b["behavior_slow_state"]).sum()
                stability["mismatch_rate"] = float(mismatches / len(df_b))
    except Exception as e:
        logger.error(f"Error reading behavior predictions: {e}")

    # Generate synthetic metrics for missing NextJS components until models embed them
    return {
        "success": True,
        "pipeline_health": files_health,
        "model_performance": {
            "status": "Healthy",
            "log_likelihood_avg": -15.42 
        },
        "stability": stability,
        "drift_retraining": {
            "days_since_last_retrain": 2,
            "drift_score": 0.12
        },
        "system_quality_score": 98.5
    }
