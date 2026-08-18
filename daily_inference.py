#!/usr/bin/env python3
"""
Daily Regime Inference Orchestration Script

Usage:
    python scripts/daily_inference.py [--date YYYY-MM-DD]
    python scripts/daily_inference.py --backfill [--end-date YYYY-MM-DD]

Exit codes:
    0  -> SUCCESS
    10 -> SOFT_FAIL (e.g., no fresh trading data yet)
    1  -> HARD_FAIL
"""

import sys
import json
import argparse
import subprocess
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

MARKET_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MARKET_DIR.parent
LOGS_DIR = PROJECT_ROOT / 'logs' / 'market_regime'
MODELS_DIR = MARKET_DIR / 'models'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'market_regime'
FEATURES_DIR = MARKET_DIR / 'features'
DATA_DIR = MARKET_DIR / 'data' / 'processed'

LOGS_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

from data.ingestion.nse_calendar import NSECalendar


def append_orchestrator_log(payload):
    log_file = LOGS_DIR / 'daily_inference_orchestrator.jsonl'
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload) + '\n')


def _safe_max_date(csv_path):
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=['Date'])
        if 'Date' not in df.columns or len(df) == 0:
            return None
        dt = pd.to_datetime(df['Date'], errors='coerce').dropna()
        if len(dt) == 0:
            return None
        return pd.Timestamp(dt.max()).normalize()
    except Exception:
        return None


def _append_only_csv(existing_path, new_path, date_col='Date'):
    existing_path = Path(existing_path)
    new_path = Path(new_path)
    if not new_path.exists():
        return {'status': 'NO_NEW_FILE', 'rows_appended': 0}

    new_df = pd.read_csv(new_path)
    if date_col not in new_df.columns:
        raise ValueError(f"Missing '{date_col}' in new data: {new_path}")
    if len(new_df) == 0:
        return {'status': 'NO_NEW_ROWS', 'rows_appended': 0}

    new_df[date_col] = pd.to_datetime(new_df[date_col], errors='coerce')
    new_df = new_df.dropna(subset=[date_col]).copy()
    if len(new_df) == 0:
        return {'status': 'NO_VALID_DATES', 'rows_appended': 0}
    new_df = new_df.sort_values(date_col).drop_duplicates(subset=[date_col], keep='last')

    existing_path.parent.mkdir(parents=True, exist_ok=True)
    if not existing_path.exists() or existing_path.stat().st_size == 0:
        out_df = new_df.copy()
        out_df[date_col] = out_df[date_col].dt.strftime('%Y-%m-%d')
        out_df.to_csv(existing_path, index=False)
        return {
            'status': 'CREATED',
            'rows_appended': int(len(out_df)),
            'new_max_date': out_df[date_col].max(),
        }

    existing_cols = list(pd.read_csv(existing_path, nrows=0).columns)
    new_cols = list(new_df.columns)
    if existing_cols != new_cols:
        raise ValueError(
            f'Column mismatch for append-only update: existing={existing_cols} new={new_cols}. '
            'Refusing to overwrite history.'
        )

    existing_dates = pd.read_csv(existing_path, usecols=[date_col])
    existing_dates[date_col] = pd.to_datetime(existing_dates[date_col], errors='coerce')
    existing_set = set(existing_dates[date_col].dropna().dt.normalize().tolist())
    append_df = new_df[~new_df[date_col].dt.normalize().isin(existing_set)].copy()
    if len(append_df) == 0:
        return {'status': 'UP_TO_DATE', 'rows_appended': 0}

    append_df = append_df[existing_cols]
    append_df[date_col] = append_df[date_col].dt.strftime('%Y-%m-%d')
    with open(existing_path, 'a', encoding='utf-8', newline='') as f:
        append_df.to_csv(f, index=False, header=False)

    return {
        'status': 'APPENDED',
        'rows_appended': int(len(append_df)),
        'new_max_date': append_df[date_col].max(),
    }


def run_data_refresh(run_date, years=15):
    """
    Refresh upstream market data and rebuild feature matrices.
    """
    py = sys.executable
    run_ts = pd.Timestamp(run_date).normalize()
    # yfinance end is typically exclusive, so request through next day.
    fetch_end = (run_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    market_path = DATA_DIR / 'market_data_historical.csv'
    market_max_before = _safe_max_date(market_path)
    fetch_start = None
    if market_max_before is not None:
        fetch_start_ts = pd.Timestamp(market_max_before) + pd.Timedelta(days=1)
        fetch_start = fetch_start_ts.strftime('%Y-%m-%d')
    else:
        fetch_start_ts = run_ts - pd.Timedelta(days=int(years) * 365 + 30)
        fetch_start = fetch_start_ts.strftime('%Y-%m-%d')

    fetch_details = {
        'status': 'SKIPPED',
        'fetch_start_date_requested': fetch_start,
        'fetch_end_date_requested': fetch_end,
        'rows_appended_market': 0,
        'market_max_before': market_max_before.strftime('%Y-%m-%d') if market_max_before is not None else None,
    }

    need_fetch = pd.Timestamp(fetch_start) < pd.Timestamp(fetch_end)
    temp_fetch_path = LOGS_DIR / f'_market_delta_{run_ts.strftime("%Y%m%d")}_{os.getpid()}.csv'

    fetch_cmd = [
        py,
        str(PROJECT_ROOT / 'data' / 'ingestion' / 'fetch_nse.py'),
        '--start-date',
        fetch_start,
        '--end-date',
        fetch_end,
        '--output',
        str(temp_fetch_path),
    ]
    build_cmd = [
        py,
        str(MARKET_DIR / 'scripts' / 'build_features.py'),
        '--data-path',
        str(market_path),
        '--output-dir',
        str(FEATURES_DIR),
        '--append-only',
    ]

    if need_fetch:
        fetch_proc = subprocess.run(fetch_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        no_new_data_markers = [
            'No data returned from Yahoo Finance',
            'No VIX data returned from Yahoo Finance',
        ]
        fetch_out = (fetch_proc.stdout or '') + '\n' + (fetch_proc.stderr or '')
        if fetch_proc.returncode != 0 and not any(marker in fetch_out for marker in no_new_data_markers):
            temp_fetch_path.unlink(missing_ok=True)
            return {
                'status': 'FAILED',
                'failed_step': 'fetch_nse',
                'fetch_returncode': int(fetch_proc.returncode),
                'fetch_stdout_tail': fetch_proc.stdout[-3000:],
                'fetch_stderr_tail': fetch_proc.stderr[-3000:],
                'fetch_start_date_requested': fetch_start,
                'fetch_end_date_requested': fetch_end,
            }
        if fetch_proc.returncode == 0 and temp_fetch_path.exists():
            merge_out = _append_only_csv(market_path, temp_fetch_path, date_col='Date')
            fetch_details.update(
                {
                    'status': 'SUCCESS',
                    'fetch_returncode': int(fetch_proc.returncode),
                    'fetch_stdout_tail': fetch_proc.stdout[-1000:],
                    'rows_appended_market': int(merge_out.get('rows_appended', 0)),
                    'market_append_status': merge_out.get('status'),
                }
            )
        else:
            fetch_details.update(
                {
                    'status': 'NO_NEW_DATA',
                    'fetch_returncode': int(fetch_proc.returncode),
                    'fetch_stdout_tail': fetch_proc.stdout[-1000:],
                    'fetch_stderr_tail': fetch_proc.stderr[-1000:],
                }
            )
    else:
        fetch_details.update({'status': 'UP_TO_DATE'})

    temp_fetch_path.unlink(missing_ok=True)

    market_max_after = _safe_max_date(market_path)
    fetch_details['market_max_after'] = market_max_after.strftime('%Y-%m-%d') if market_max_after is not None else None
    final_max_before = _safe_max_date(FEATURES_DIR / 'final_features_matrix.csv')
    slow_max_before = _safe_max_date(FEATURES_DIR / 'slow_features_matrix.csv')
    fast_max_before = _safe_max_date(FEATURES_DIR / 'fast_features_matrix.csv')

    feature_files_missing = any(
        not (FEATURES_DIR / f).exists()
        for f in ['final_features_matrix.csv', 'slow_features_matrix.csv', 'fast_features_matrix.csv']
    )
    feature_max_before = min(
        [d for d in [final_max_before, slow_max_before, fast_max_before] if d is not None],
        default=None,
    )
    need_build = feature_files_missing or (market_max_after is not None and (feature_max_before is None or feature_max_before < market_max_after))

    if not need_build:
        return {
            'status': 'SUCCESS',
            'fetch': fetch_details,
            'build': {
                'status': 'UP_TO_DATE',
                'rows_appended_features': 0,
                'feature_max_before': feature_max_before.strftime('%Y-%m-%d') if feature_max_before is not None else None,
                'feature_max_after': feature_max_before.strftime('%Y-%m-%d') if feature_max_before is not None else None,
            },
            'fetch_start_date_requested': fetch_start,
            'fetch_end_date_requested': fetch_end,
        }

    build_proc = subprocess.run(build_cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if build_proc.returncode != 0:
        return {
            'status': 'FAILED',
            'failed_step': 'build_features',
            'fetch': fetch_details,
            'build_returncode': int(build_proc.returncode),
            'build_stdout_tail': build_proc.stdout[-3000:],
            'build_stderr_tail': build_proc.stderr[-3000:],
        }

    final_max_after = _safe_max_date(FEATURES_DIR / 'final_features_matrix.csv')
    slow_max_after = _safe_max_date(FEATURES_DIR / 'slow_features_matrix.csv')
    fast_max_after = _safe_max_date(FEATURES_DIR / 'fast_features_matrix.csv')
    feature_max_after = min(
        [d for d in [final_max_after, slow_max_after, fast_max_after] if d is not None],
        default=None,
    )

    return {
        'status': 'SUCCESS',
        'fetch': fetch_details,
        'build': {
            'status': 'SUCCESS',
            'build_returncode': int(build_proc.returncode),
            'build_stdout_tail': build_proc.stdout[-1000:],
            'feature_max_before': feature_max_before.strftime('%Y-%m-%d') if feature_max_before is not None else None,
            'feature_max_after': feature_max_after.strftime('%Y-%m-%d') if feature_max_after is not None else None,
        },
        'fetch_start_date_requested': fetch_start,
        'fetch_end_date_requested': fetch_end,
    }


def parse_date(value):
    return pd.Timestamp(datetime.strptime(value, '%Y-%m-%d')).normalize()


def load_market_dates():
    market_file = DATA_DIR / 'market_data_historical.csv'
    if not market_file.exists():
        raise FileNotFoundError(f'Missing market data file: {market_file}')
    df = pd.read_csv(market_file, parse_dates=['Date'])
    return pd.DatetimeIndex(sorted(df['Date'].dropna().dt.normalize().unique()))


def load_timeline_dates(timeline_file):
    if not timeline_file.exists():
        return set(), None
    hist = pd.read_csv(timeline_file, parse_dates=['Date'])
    if 'Date' not in hist.columns or len(hist) == 0:
        return set(), None
    dates = set(pd.to_datetime(hist['Date'], errors='coerce').dropna().dt.normalize().tolist())
    max_date = max(dates) if dates else None
    return dates, max_date


def get_expected_trading_dates(start_date, end_date):
    """
    Get expected NSE trading dates between start_date and end_date (inclusive).
    Falls back to weekdays if calendar API/cache is unavailable.
    """
    start_dt = pd.Timestamp(start_date).to_pydatetime()
    end_dt = pd.Timestamp(end_date).to_pydatetime()
    if start_dt > end_dt:
        return pd.DatetimeIndex([])

    try:
        cal = NSECalendar(cache_dir=str(PROJECT_ROOT / 'data' / 'ingestion' / '.cache'))
        days = cal.get_trading_days(start_dt, end_dt)
        idx = pd.DatetimeIndex(pd.to_datetime(days)).normalize()
        return idx
    except Exception:
        # Conservative fallback: weekdays only.
        return pd.date_range(start=start_date, end=end_date, freq='B').normalize()


def compute_missing_trailing_dates(run_end, timeline_file):
    """
    Compute missing trailing trading dates after current timeline tip, capped by available market dates.
    """
    market_dates = load_market_dates()
    timeline_dates, last_existing = load_timeline_dates(timeline_file)
    if len(market_dates) == 0:
        return pd.DatetimeIndex([]), timeline_dates, last_existing

    run_end = pd.Timestamp(run_end).normalize()
    market_cap = pd.Timestamp(market_dates.max()).normalize()
    end_effective = min(run_end, market_cap)

    if last_existing is None:
        start = pd.Timestamp(market_dates.min()).normalize()
    else:
        start = pd.Timestamp(last_existing).normalize() + pd.Timedelta(days=1)

    if start > end_effective:
        return pd.DatetimeIndex([]), timeline_dates, last_existing

    expected = get_expected_trading_dates(start, end_effective)
    market_set = set(pd.DatetimeIndex(market_dates).normalize().tolist())
    timeline_set = set(pd.DatetimeIndex(list(timeline_dates)).normalize().tolist())

    missing = [d for d in expected if d in market_set and d not in timeline_set]
    return pd.DatetimeIndex(missing), timeline_dates, last_existing


def run_single(pipeline, run_date):
    result = pipeline.run(run_date.to_pydatetime())
    return {
        'mode': 'single',
        'run_date': run_date.strftime('%Y-%m-%d'),
        'status': result.get('status'),
        'status_code': int(result.get('status_code', 1)),
        'elapsed_seconds': float(result.get('elapsed_seconds', -1.0)),
        'stage_times': result.get('stage_times', {}),
        'asof_date': result.get('asof_date'),
        'combined_state': result.get('combined_state'),
        'alert_severity': result.get('alert_severity'),
        'error': result.get('error'),
    }


def run_backfill(pipeline, run_start, run_end, timeline_file):
    pipeline.skip_lock = True  # Backfill is single-threaded; skip lock to avoid contention
    market_dates = load_market_dates()
    eligible = market_dates[(market_dates >= run_start) & (market_dates <= run_end)]

    timeline_dates, last_existing = load_timeline_dates(timeline_file)
    if last_existing is not None:
        # Append-only backfill: never rewrite or insert before current history tip.
        eligible = eligible[eligible > last_existing]

    missing_dates = [d for d in eligible if d not in timeline_dates]

    results = []
    success_count = 0
    soft_fail_count = 0
    hard_fail_count = 0

    for d in missing_dates:
        out = pipeline.run(d.to_pydatetime())
        results.append(out)
        status = out.get('status')
        if status == 'SUCCESS':
            success_count += 1
        elif status == 'SOFT_FAIL':
            soft_fail_count += 1
        else:
            hard_fail_count += 1

    if hard_fail_count > 0:
        status = 'HARD_FAIL'
        status_code = 1
    elif len(missing_dates) == 0:
        status = 'SUCCESS'
        status_code = 0
    elif success_count == 0 and soft_fail_count > 0:
        status = 'SOFT_FAIL'
        status_code = 10
    else:
        status = 'SUCCESS'
        status_code = 0

    elapsed_seconds = float(sum(float(r.get('elapsed_seconds', 0.0)) for r in results))
    return {
        'mode': 'backfill',
        'requested_start_date': run_start.strftime('%Y-%m-%d'),
        'requested_end_date': run_end.strftime('%Y-%m-%d'),
        'last_existing_date': last_existing.strftime('%Y-%m-%d') if last_existing is not None else None,
        'missing_dates_count': len(missing_dates),
        'processed_dates_count': len(results),
        'first_processed_date': missing_dates[0].strftime('%Y-%m-%d') if missing_dates else None,
        'last_processed_date': missing_dates[-1].strftime('%Y-%m-%d') if missing_dates else None,
        'success_count': success_count,
        'soft_fail_count': soft_fail_count,
        'hard_fail_count': hard_fail_count,
        'status': status,
        'status_code': status_code,
        'elapsed_seconds': elapsed_seconds,
        'error': None if hard_fail_count == 0 else 'One or more backfill runs failed',
    }


def main():
    parser = argparse.ArgumentParser(description='Daily regime inference orchestrator')
    parser.add_argument('--date', type=str, help='Run date YYYY-MM-DD (defaults to today)')
    parser.add_argument('--backfill', action='store_true', help='Backfill missing trailing dates only (append-only).')
    parser.add_argument('--no-auto-backfill', action='store_true', help='Disable auto backfill trigger for this run.')
    parser.add_argument('--auto-backfill-threshold', type=int, default=None, help='Trigger auto-backfill when missing trading days >= X.')
    parser.add_argument('--start-date', type=str, help='Backfill start date YYYY-MM-DD (optional).')
    parser.add_argument('--end-date', type=str, help='Backfill end date YYYY-MM-DD (defaults to --date or today).')
    parser.add_argument('--no-data-refresh', action='store_true', help='Skip fetching latest market data + feature rebuild.')
    parser.add_argument('--allow-stale-data', action='store_true', help='If data refresh fails, continue using local data.')
    parser.add_argument('--data-refresh-years', type=int, default=15, help='Years to fetch during data refresh.')
    args = parser.parse_args()

    run_date = parse_date(args.date) if args.date else pd.Timestamp(datetime.now()).normalize()
    start = datetime.now().isoformat()
    data_refresh = {'status': 'SKIPPED'}

    if not args.no_data_refresh:
        data_refresh = run_data_refresh(run_date=run_date, years=int(args.data_refresh_years))
        if data_refresh.get('status') != 'SUCCESS' and not args.allow_stale_data:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'run_date': run_date.strftime('%Y-%m-%d'),
                'status': 'HARD_FAIL',
                'status_code': 1,
                'error': f"Data refresh failed at step={data_refresh.get('failed_step')}",
                'data_refresh': data_refresh,
            }
            append_orchestrator_log(payload)
            print(json.dumps(payload, indent=2))
            return 1

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from daily_inference_runtime import DailyInferencePipeline
    except Exception as exc:
        payload = {
            'timestamp': datetime.now().isoformat(),
            'run_date': run_date.strftime('%Y-%m-%d'),
            'status': 'HARD_FAIL',
            'status_code': 1,
            'error': f'Failed to import runtime pipeline: {exc}',
        }
        append_orchestrator_log(payload)
        print(payload['error'])
        return 1

    config = {
        'ema_alpha_slow': 0.05,     # NB02: implicit (no EMA in batch, but 0.05 for daily smoothing)
        'ema_alpha_fast': 0.10,     # NB02: implicit fast smoothing
        'prob_floor_eps': 0.02,
        'prob_ceiling': 0.98,
        'hysteresis_enter': 0.60,   # NB02: enter_threshold=0.60
        'hysteresis_exit': 0.40,    # NB02: exit_threshold=0.40
        'macro_threshold_recalibration_enabled': True,
        'macro_threshold_recalibration_lookback_days': 756,
        'macro_threshold_recalibration_high_quantile': 0.70,
        'macro_threshold_recalibration_low_quantile': 0.30,
        'macro_threshold_enter_min': 0.55,
        'macro_threshold_enter_max': 0.80,
        'macro_threshold_exit_min': 0.20,
        'macro_threshold_exit_max': 0.45,
        'macro_threshold_min_gap': 0.12,
        'hysteresis_stress_enter': 0.65,  # NB02: p_stress > 0.65 to enter Stress
        'hysteresis_stress_exit': 0.35,   # NB02: p_stress < 0.35 to exit Stress
        'hysteresis_choppy_enter': 0.55,  # NB02: p_choppy > 0.55 to enter Choppy
        'hysteresis_choppy_exit': 0.35,   # NB02: p_choppy < 0.35 to exit Choppy
        'adaptive_saturation_high': 0.98,
        'adaptive_saturation_low': 0.02,
        'adaptive_saturation_max_days': 40,
        'adaptive_saturation_recenter_strength': 0.25,
        'min_days_durable': 15,     # NB02: MIN_DURATIONS_MACRO['Durable'] = 15
        'min_days_fragile': 10,     # NB02: MIN_DURATIONS_MACRO['Fragile'] = 10
        'min_days_calm': 2,         # NB02: MIN_DURATIONS_FAST['Calm'] = 2
        'min_days_choppy': 2,       # NB02: MIN_DURATIONS_FAST['Choppy'] = 2
        'min_days_stress': 3,       # NB02: MIN_DURATIONS_FAST['Stress'] = 3
        'cooldown_macro_days': 2,
        'cooldown_fast_days': 1,
        'stress_override_immediate': True,
        'lookback_days': 300,
        'state_history_window': 30,
        'attribution_stats_window': 756,
        'lock_stale_seconds': 6 * 3600,
        'yearly_occupancy_alarm_enabled': True,
        'yearly_occupancy_limit': 0.90,
        'yearly_occupancy_min_samples': 40,
        'timeline_file': FEATURES_DIR / 'regime_timeline_history.csv',
        'timeline_rows_dir': OUTPUT_DIR / 'timeline_rows',
        'state_store_file': OUTPUT_DIR / 'daily_inference_state.json',
        'run_log_file': LOGS_DIR / 'daily_inference_runs.jsonl',
        'lock_file': LOGS_DIR / 'daily_inference.lock',
        'auto_backfill_enabled': True,
        'auto_backfill_trigger_days': 2,
    }
    if args.auto_backfill_threshold is not None:
        config['auto_backfill_trigger_days'] = max(1, int(args.auto_backfill_threshold))

    pipeline = DailyInferencePipeline(
        models_path=MODELS_DIR / 'hmm_regime_models.joblib',
        guardrail_config=config,
    )

    auto_backfill_enabled = bool(config.get('auto_backfill_enabled', True)) and (not args.no_auto_backfill)
    timeline_file = Path(config['timeline_file'])

    auto_missing_dates = pd.DatetimeIndex([])
    auto_triggered = False
    if (not args.backfill) and auto_backfill_enabled:
        auto_missing_dates, _, auto_last_existing = compute_missing_trailing_dates(run_end=run_date, timeline_file=timeline_file)
        trigger_days = int(config.get('auto_backfill_trigger_days', 2))
        auto_triggered = len(auto_missing_dates) >= trigger_days
    else:
        auto_last_existing = None

    if args.backfill or auto_triggered:
        end_date = parse_date(args.end_date) if args.end_date else run_date
        _, last_existing = load_timeline_dates(timeline_file)

        if auto_triggered:
            requested_start = pd.Timestamp(auto_missing_dates.min()).normalize()
        elif args.start_date:
            requested_start = parse_date(args.start_date)
            if last_existing is not None and requested_start <= last_existing:
                # Preserve append-only semantics explicitly requested by user.
                requested_start = pd.Timestamp(last_existing) + pd.Timedelta(days=1)
        else:
            requested_start = (pd.Timestamp(last_existing) + pd.Timedelta(days=1)) if last_existing is not None else end_date

        result = run_backfill(
            pipeline=pipeline,
            run_start=requested_start.normalize(),
            run_end=end_date.normalize(),
            timeline_file=timeline_file,
        )
        result['auto_backfill_triggered'] = bool(auto_triggered)
        result['auto_backfill_missing_days'] = int(len(auto_missing_dates))
        result['auto_backfill_last_existing_date'] = (
            auto_last_existing.strftime('%Y-%m-%d') if auto_last_existing is not None else None
        )
    else:
        result = run_single(pipeline=pipeline, run_date=run_date)
        result['auto_backfill_triggered'] = False
        result['auto_backfill_missing_days'] = int(len(auto_missing_dates))
        result['auto_backfill_last_existing_date'] = (
            auto_last_existing.strftime('%Y-%m-%d') if auto_last_existing is not None else None
        )

    payload = {
        'timestamp': datetime.now().isoformat(),
        'start_timestamp': start,
        'run_date': run_date.strftime('%Y-%m-%d'),
        'mode': result.get('mode', 'single'),
        'status': result.get('status'),
        'status_code': int(result.get('status_code', 1)),
        'elapsed_seconds': float(result.get('elapsed_seconds', -1.0)),
        'stage_times': result.get('stage_times', {}),
        'asof_date': result.get('asof_date'),
        'combined_state': result.get('combined_state'),
        'alert_severity': result.get('alert_severity'),
        'requested_start_date': result.get('requested_start_date'),
        'requested_end_date': result.get('requested_end_date'),
        'last_existing_date': result.get('last_existing_date'),
        'missing_dates_count': result.get('missing_dates_count'),
        'processed_dates_count': result.get('processed_dates_count'),
        'auto_backfill_triggered': result.get('auto_backfill_triggered'),
        'auto_backfill_missing_days': result.get('auto_backfill_missing_days'),
        'auto_backfill_last_existing_date': result.get('auto_backfill_last_existing_date'),
        'data_refresh': data_refresh,
        'error': result.get('error'),
    }
    append_orchestrator_log(payload)

    print(json.dumps(payload, indent=2))
    return int(payload['status_code'])


if __name__ == '__main__':
    sys.exit(main())
