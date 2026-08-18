#!/usr/bin/env python3
"""
Deterministic feature build entrypoint for production.

Builds:
- features/slow_features_matrix.csv
- features/fast_features_matrix.csv
- features/final_features_matrix.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import shutil
import pandas as pd


def resolve_base_dir() -> Path:
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "features").exists() and (script_root / "data").exists():
        return script_root
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent]:
        if (candidate / "features").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not resolve project root with features/ and data/ folders")


BASE_DIR = resolve_base_dir()
ROOT_DIR = BASE_DIR.parent
FEATURES_DIR = BASE_DIR / "features"
OUTPUT_DIR = ROOT_DIR / "output" / "market_regime"
LOGS_DIR = ROOT_DIR / "logs" / "market_regime"

for p in [FEATURES_DIR, OUTPUT_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from features.feature_store import FeatureStore  # noqa: E402


def append_log(payload):
    log_file = LOGS_DIR / "feature_build_runs.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build SLOW/FAST/final feature matrices")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(BASE_DIR / "data" / "processed" / "market_data_historical.csv"),
        help="Input market data CSV path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(FEATURES_DIR),
        help="Output features directory",
    )
    parser.add_argument(
        "--append-only",
        action="store_true",
        help="Append only new dates to feature CSVs (never overwrite existing history rows).",
    )
    return parser.parse_args()


def _append_only_merge(existing_path: Path, new_path: Path, date_col: str = "Date") -> int:
    if not new_path.exists():
        return 0
    new_df = pd.read_csv(new_path)
    if len(new_df) == 0:
        return 0
    if date_col not in new_df.columns:
        raise ValueError(f"Missing '{date_col}' column in {new_path}")
    new_df[date_col] = pd.to_datetime(new_df[date_col], errors="coerce")
    new_df = new_df.dropna(subset=[date_col]).copy()
    if len(new_df) == 0:
        return 0
    new_df = new_df.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")

    existing_path.parent.mkdir(parents=True, exist_ok=True)
    if not existing_path.exists() or existing_path.stat().st_size == 0:
        out_df = new_df.copy()
        out_df[date_col] = out_df[date_col].dt.strftime("%Y-%m-%d")
        out_df.to_csv(existing_path, index=False)
        return int(len(out_df))

    existing_cols = list(pd.read_csv(existing_path, nrows=0).columns)
    new_cols = list(new_df.columns)
    if existing_cols != new_cols:
        raise ValueError(
            f"Column mismatch for append-only merge at {existing_path.name}: "
            f"existing={existing_cols} new={new_cols}. Refusing to overwrite."
        )

    existing_dates = pd.read_csv(existing_path, usecols=[date_col])
    existing_dates[date_col] = pd.to_datetime(existing_dates[date_col], errors="coerce")
    existing_set = set(existing_dates[date_col].dropna().dt.normalize().tolist())
    append_df = new_df[~new_df[date_col].dt.normalize().isin(existing_set)].copy()
    if len(append_df) == 0:
        return 0
    append_df = append_df[existing_cols]
    append_df[date_col] = append_df[date_col].dt.strftime("%Y-%m-%d")
    with open(existing_path, "a", encoding="utf-8", newline="") as f:
        append_df.to_csv(f, index=False, header=False)
    return int(len(append_df))


def main():
    # Avoid unicode print crashes on cp1252 terminals.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    args = parse_args()
    start = datetime.now()

    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows_appended = {"slow": 0, "fast": 0, "final": 0}

        if args.append_only:
            tmp_dir = OUTPUT_DIR / "_feature_build_tmp" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                store = FeatureStore(data_path=args.data_path, output_dir=str(tmp_dir))
                final_df = store.run()

                rows_appended["slow"] = _append_only_merge(
                    out_dir / "slow_features_matrix.csv", tmp_dir / "slow_features_matrix.csv"
                )
                rows_appended["fast"] = _append_only_merge(
                    out_dir / "fast_features_matrix.csv", tmp_dir / "fast_features_matrix.csv"
                )
                rows_appended["final"] = _append_only_merge(
                    out_dir / "final_features_matrix.csv", tmp_dir / "final_features_matrix.csv"
                )
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            store = FeatureStore(data_path=args.data_path, output_dir=args.output_dir)
            final_df = store.run()

        artifacts = {
            "timestamp": datetime.now().isoformat(),
            "status": "SUCCESS",
            "append_only": bool(args.append_only),
            "data_path": str(args.data_path),
            "output_dir": str(out_dir),
            "slow_features_file": str(out_dir / "slow_features_matrix.csv"),
            "fast_features_file": str(out_dir / "fast_features_matrix.csv"),
            "final_features_file": str(out_dir / "final_features_matrix.csv"),
            "rows_final": int(len(pd.read_csv(out_dir / "final_features_matrix.csv"))) if (out_dir / "final_features_matrix.csv").exists() else 0,
            "n_features_final": int(len([c for c in pd.read_csv(out_dir / "final_features_matrix.csv", nrows=1).columns if c != "Date"])) if (out_dir / "final_features_matrix.csv").exists() else 0,
            "rows_appended": rows_appended,
            "elapsed_seconds": float((datetime.now() - start).total_seconds()),
        }

        meta_path = out_dir / "feature_build_metadata.json"
        meta_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
        append_log(artifacts)
        print(json.dumps(artifacts, indent=2))
        return 0
    except Exception as exc:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "status": "FAILED",
            "error": str(exc),
            "data_path": str(args.data_path),
            "output_dir": str(args.output_dir),
            "elapsed_seconds": float((datetime.now() - start).total_seconds()),
        }
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
