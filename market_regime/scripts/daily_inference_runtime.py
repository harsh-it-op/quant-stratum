# Canonical runtime implementation for daily inference (non-notebook source of truth).
import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def resolve_base_dir() -> Path:
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / 'models').exists() and (script_root / 'data').exists():
        return script_root
    cwd = Path.cwd().resolve()
    for candidate in [cwd, cwd.parent]:
        if (candidate / 'models').exists() and (candidate / 'data').exists():
            return candidate
    raise FileNotFoundError('Could not resolve project root with models/ and data/ folders')


BASE_DIR = resolve_base_dir()
DATA_DIR = BASE_DIR / 'data' / 'processed'
FEATURES_DIR = BASE_DIR / 'features'
MODELS_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'output'
LOGS_DIR = BASE_DIR / 'logs'

for dir_path in [OUTPUT_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class SoftFailError(Exception):
    """Non-fatal execution issue (e.g., no new trading day yet)."""


class DailyInferencePipeline:
    """Production EOD inference pipeline with continuity, guardrails, and runtime controls."""

    def __init__(self, models_path, guardrail_config):
        self.models = joblib.load(models_path)
        self.config = guardrail_config

        self.timeline_file = Path(self.config.get('timeline_file', FEATURES_DIR / 'regime_timeline_history.csv'))
        self.timeline_rows_dir = Path(self.config.get('timeline_rows_dir', OUTPUT_DIR / 'timeline_rows'))
        self.state_store_file = Path(self.config.get('state_store_file', OUTPUT_DIR / 'daily_inference_state.json'))
        self.run_log_file = Path(self.config.get('run_log_file', LOGS_DIR / 'daily_inference_runs.jsonl'))
        self.lock_file = Path(self.config.get('lock_file', LOGS_DIR / 'daily_inference.lock'))
        self.skip_lock = False

        self.lookback_days = int(self.config.get('lookback_days', 300))
        self.state_history_window = int(self.config.get('state_history_window', 30))
        self.attribution_stats_window = int(self.config.get('attribution_stats_window', 756))

        self.market_data = None
        self.features_all = None

        self.slow_features = list(self.models.get('slow_feature_names', []))
        self.fast_features = list(self.models.get('fast_feature_names', []))
        self.all_features = self.slow_features + self.fast_features

        self.slow_model = self.models['slow_hmm']
        self.fast_durable_model = self.models['fast_hmm_durable']
        self.fast_fragile_model = self.models['fast_hmm_fragile']

        self.slow_map = self._normalize_state_map(
            self.models.get('slow_state_map') or self.models.get('state_map', {}).get('slow'),
            ['Durable', 'Fragile'],
            expected_states=self.slow_model.n_components,
            map_name='slow',
        )
        self.fast_durable_map = self._normalize_state_map(
            self.models.get('dur_state_map') or self.models.get('state_map', {}).get('fast_durable'),
            ['Calm', 'Choppy', 'Stress'],
            expected_states=self.fast_durable_model.n_components,
            map_name='fast_durable',
        )
        self.fast_fragile_map = self._normalize_state_map(
            self.models.get('fra_state_map') or self.models.get('state_map', {}).get('fast_fragile'),
            ['Calm', 'Choppy', 'Stress'],
            expected_states=self.fast_fragile_model.n_components,
            map_name='fast_fragile',
        )

        self.runtime_state = {
            'ema_state': {
                'p_fragile_smooth': 0.5,
                'p_calm_smooth': 1 / 3,
                'p_choppy_smooth': 1 / 3,
                'p_stress_smooth': 1 / 3,
            },
            'macro': {
                'stable': None,
                'candidate': None,
                'candidate_count': 0,
                'cooldown': 0,
                'last_eval_date': None,
            },
            'fast': {
                'stable': None,
                'candidate': None,
                'candidate_count': 0,
                'cooldown': 0,
                'last_eval_date': None,
            },
            'ema_adaptive': {
                'p_fragile_adaptive': 0.5,
                'p_calm_adaptive': 1 / 3,
                'p_choppy_adaptive': 1 / 3,
                'p_stress_adaptive': 1 / 3,
            },
            'macro_adaptive': {
                'stable': None,
                'candidate': None,
                'candidate_count': 0,
                'cooldown': 0,
                'last_eval_date': None,
            },
            'fast_adaptive': {
                'stable': None,
                'candidate': None,
                'candidate_count': 0,
                'cooldown': 0,
                'last_eval_date': None,
            },
            'saturation': {
                'macro_high_count': 0,
                'macro_low_count': 0,
                'fast_high_state': None,
                'fast_high_count': 0,
            },
            'recent_states': [],
            'last_processed_date': None,
        }

        self._load_data_sources()
        self._validate_model_schema()
        self._override_startprobs_with_stationary()
        self._prepare_feature_stats(asof_date=self.features_all.index.max())
        self._restore_runtime_state()

        print('Pipeline initialized')
        print(f'  Models: SLOW({self.slow_model.n_components}), FAST_Durable({self.fast_durable_model.n_components}), FAST_Fragile({self.fast_fragile_model.n_components})')

    def _load_data_sources(self):
        self.market_data = pd.read_csv(DATA_DIR / 'market_data_historical.csv', parse_dates=['Date'], index_col='Date').sort_index()
        self.features_all = pd.read_csv(FEATURES_DIR / 'final_features_matrix.csv', parse_dates=['Date'], index_col='Date').sort_index()

    def _validate_model_schema(self):
        required_keys = ['slow_hmm', 'fast_hmm_durable', 'fast_hmm_fragile', 'slow_feature_names', 'fast_feature_names']
        missing = [k for k in required_keys if k not in self.models]
        if missing:
            raise KeyError(f'Model bundle missing keys: {missing}')

        missing_features = [f for f in self.all_features if f not in self.features_all.columns]
        if missing_features:
            raise KeyError(f'Feature matrix missing model features: {missing_features[:10]}')

    def _override_startprobs_with_stationary(self):
        """Override degenerate startprob_ with stationary distribution from transmat_.

        HMM startprob_ often degenerates to [1,0,...] or [0,...,1] depending on the
        first observation in the training window. For single-point inference (one feature
        row at a time), predict_proba returns startprob * emission_prob, so a zero in
        startprob permanently locks that state to zero probability.

        The stationary distribution (π where π·P = π) is the long-run equilibrium of
        the Markov chain and provides the correct uninformed prior for a random timepoint.
        """
        for name, model in [('slow', self.slow_model),
                            ('fast_durable', self.fast_durable_model),
                            ('fast_fragile', self.fast_fragile_model)]:
            T = np.array(model.transmat_, dtype=float)
            n = T.shape[0]
            # Solve π such that π·P = π and sum(π) = 1
            # Equivalent to (P^T - I)·π = 0 with constraint sum(π) = 1
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

            old_sp = model.startprob_.copy()
            model.startprob_ = pi
            print(f'  {name}: startprob {old_sp.round(3)} -> stationary {pi.round(4)}')

    def _prepare_feature_stats(self, asof_date=None):
        feature_block = self.features_all[self.all_features]
        if asof_date is not None:
            feature_block = feature_block.loc[feature_block.index <= pd.Timestamp(asof_date)]
        feature_block = feature_block.tail(self.attribution_stats_window)
        feature_block = feature_block.replace([np.inf, -np.inf], np.nan)
        self.feature_mu = feature_block.mean()
        self.feature_std = feature_block.std().replace(0, np.nan).fillna(1.0)

    def _normalize_state_map(self, mapping, label_names, expected_states, map_name):
        """
        Normalize mapping to {original_state_idx: canonical_label_idx}.

        Supports:
        - {orig_idx: label_idx}
        - {"Label": orig_idx}
        """
        if mapping is None:
            if len(label_names) != expected_states:
                raise ValueError(f'Missing mapping for {map_name} and cannot infer identity for {expected_states} states')
            print(f'WARNING: state_map missing for {map_name}; using identity fallback')
            return {i: i for i in range(expected_states)}

        norm = {}
        for k, v in mapping.items():
            if isinstance(k, str) and not k.isdigit():
                if k not in label_names:
                    continue
                orig_idx = int(v)
                label_idx = label_names.index(k)
            else:
                orig_idx = int(k)
                label_idx = int(v)
            norm[orig_idx] = label_idx

        missing_states = []
        for idx in range(expected_states):
            if idx not in norm:
                norm[idx] = idx if idx < len(label_names) else len(label_names) - 1
                missing_states.append(idx)
        if missing_states:
            print(
                f'WARNING: state_map incomplete for {map_name}; '
                f'filled missing states {missing_states} with identity fallback'
            )

        return norm

    def _restore_runtime_state(self):
        restored = False

        if self.state_store_file.exists():
            try:
                state = json.loads(self.state_store_file.read_text(encoding='utf-8'))
                if 'ema_state' in state:
                    self.runtime_state['ema_state'].update(state['ema_state'])
                if 'ema_adaptive' in state:
                    self.runtime_state['ema_adaptive'].update(state['ema_adaptive'])
                for layer in ['macro', 'fast', 'macro_adaptive', 'fast_adaptive']:
                    if layer in state:
                        self.runtime_state[layer].update(state[layer])
                if 'saturation' in state and isinstance(state['saturation'], dict):
                    self.runtime_state['saturation'].update(state['saturation'])
                if 'recent_states' in state and isinstance(state['recent_states'], list):
                    self.runtime_state['recent_states'] = state['recent_states'][-self.state_history_window:]
                self.runtime_state['last_processed_date'] = state.get('last_processed_date')
                restored = True
            except Exception:
                restored = False

        if not restored and self.timeline_file.exists():
            try:
                hist = pd.read_csv(self.timeline_file, parse_dates=['Date'])
                if len(hist) > 0:
                    hist = hist.sort_values('Date')
                    last = hist.iloc[-1]
                    self.runtime_state['macro']['stable'] = str(last['macro_state'])
                    self.runtime_state['fast']['stable'] = str(last['fast_state'])

                    if 'p_fragile' in hist.columns:
                        self.runtime_state['ema_state']['p_fragile_smooth'] = float(last['p_fragile'])
                    if 'p_calm' in hist.columns:
                        self.runtime_state['ema_state']['p_calm_smooth'] = float(last['p_calm'])
                    if 'p_choppy' in hist.columns:
                        self.runtime_state['ema_state']['p_choppy_smooth'] = float(last['p_choppy'])
                    if 'p_stress' in hist.columns:
                        self.runtime_state['ema_state']['p_stress_smooth'] = float(last['p_stress'])

                    # Restore adaptive EMA from CSV if columns exist
                    if 'p_fragile_adaptive' in hist.columns:
                        self.runtime_state['ema_adaptive']['p_fragile_adaptive'] = float(last['p_fragile_adaptive'])
                    if 'p_calm_adaptive' in hist.columns:
                        self.runtime_state['ema_adaptive']['p_calm_adaptive'] = float(last['p_calm_adaptive'])
                    if 'p_choppy_adaptive' in hist.columns:
                        self.runtime_state['ema_adaptive']['p_choppy_adaptive'] = float(last['p_choppy_adaptive'])
                    if 'p_stress_adaptive' in hist.columns:
                        self.runtime_state['ema_adaptive']['p_stress_adaptive'] = float(last['p_stress_adaptive'])
                    if 'adaptive_macro_state' in hist.columns:
                        self.runtime_state['macro_adaptive']['stable'] = str(last['adaptive_macro_state'])
                    if 'adaptive_fast_state' in hist.columns:
                        self.runtime_state['fast_adaptive']['stable'] = str(last['adaptive_fast_state'])

                    recent = hist.tail(self.state_history_window)
                    self.runtime_state['recent_states'] = [
                        {
                            'date': r['Date'].strftime('%Y-%m-%d') if hasattr(r['Date'], 'strftime') else str(r['Date']),
                            'macro_state': str(r['macro_state']),
                            'fast_state': str(r['fast_state']),
                            'combined_state': str(r['combined_state']) if 'combined_state' in hist.columns else f"{r['macro_state']}–{r['fast_state']}",
                        }
                        for _, r in recent.iterrows()
                    ]
                    self.runtime_state['last_processed_date'] = self.runtime_state['recent_states'][-1]['date']
            except Exception:
                pass

        self._save_runtime_state()

    def _save_runtime_state(self):
        payload = {
            'ema_state': self.runtime_state['ema_state'],
            'macro': self.runtime_state['macro'],
            'fast': self.runtime_state['fast'],
            'ema_adaptive': self.runtime_state.get('ema_adaptive', {}),
            'macro_adaptive': self.runtime_state.get('macro_adaptive', {}),
            'fast_adaptive': self.runtime_state.get('fast_adaptive', {}),
            'saturation': self.runtime_state.get('saturation', {}),
            'recent_states': self.runtime_state['recent_states'][-self.state_history_window:],
            'last_processed_date': self.runtime_state.get('last_processed_date'),
            'updated_at': datetime.now().isoformat(),
        }
        self._atomic_write_json(self.state_store_file, payload)

    def _atomic_write_json(self, path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        self._replace_with_retry(tmp, path)

    def _atomic_write_csv(self, path, df):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        df.to_csv(tmp, index=False)
        self._replace_with_retry(tmp, path)

    @staticmethod
    def _replace_with_retry(src, dst, retries=5, delay=0.1):
        """os.replace with retry for Windows WinError 5 (Access Denied).

        Antivirus, file indexers, and VS Code file watchers can briefly
        hold locks on .tmp or target files during rapid backfill iteration.
        A short retry loop resolves these transient conflicts.
        """
        for attempt in range(retries):
            try:
                os.replace(src, dst)
                return
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(delay * (attempt + 1))

    def _read_last_timeline_date(self):
        """Read last timeline date without loading whole CSV."""
        if not self.timeline_file.exists():
            return None

        with open(self.timeline_file, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end <= 0:
                return None

            pos = end - 1
            buf = bytearray()
            while pos >= 0:
                f.seek(pos)
                ch = f.read(1)
                if ch == b'\n' and buf:
                    break
                if ch not in (b'\n', b'\r'):
                    buf.extend(ch)
                pos -= 1

        if not buf:
            return None

        last_line = bytes(reversed(buf)).decode('utf-8')
        parsed = next(csv.reader([last_line]), None)
        if not parsed or len(parsed) == 0:
            return None

        try:
            return pd.Timestamp(parsed[0]).date()
        except Exception:
            return None

    def _append_timeline_row(self, row):
        """Append one row to timeline history (without full-file rebuild)."""
        self.timeline_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.timeline_file.exists():
            self._atomic_write_csv(self.timeline_file, row)
            return

        # Keep timeline schema consistent as new columns are introduced.
        with open(self.timeline_file, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            existing_cols = next(reader, [])

        row_cols = list(row.columns)
        if existing_cols and (existing_cols != row_cols):
            hist = pd.read_csv(self.timeline_file)
            for c in row_cols:
                if c not in hist.columns:
                    hist[c] = np.nan
            # Preserve new canonical row order first, then legacy extras.
            ordered = row_cols + [c for c in hist.columns if c not in row_cols]
            hist = hist[ordered]
            self._atomic_write_csv(self.timeline_file, hist)
            existing_cols = ordered

        asof_date = pd.Timestamp(row.iloc[0]['Date']).date()
        last_date = self._read_last_timeline_date()

        # Avoid duplicate same-date appends on reruns; daily row file still gets refreshed.
        if last_date == asof_date:
            return

        if existing_cols:
            row = row.reindex(columns=existing_cols, fill_value=np.nan)
        with open(self.timeline_file, 'a', encoding='utf-8', newline='') as f:
            row.to_csv(f, index=False, header=False)

    def _append_run_log(self, payload):
        self.run_log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.run_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload) + '\n')

    def _acquire_lock(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        if self.lock_file.exists():
            lock_age = time.time() - self.lock_file.stat().st_mtime
            stale_after = int(self.config.get('lock_stale_seconds', 6 * 3600))
            if lock_age > stale_after:
                self.lock_file.unlink(missing_ok=True)

        try:
            fd = os.open(str(self.lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(json.dumps({'pid': os.getpid(), 'started_at': datetime.now().isoformat()}))
        except FileExistsError:
            raise SoftFailError(f'Another inference run appears active (lock: {self.lock_file})')

    def _release_lock(self):
        try:
            self.lock_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _label_probs(self, posterior, mapping, label_names):
        probs = np.zeros(len(label_names), dtype=float)
        for orig_idx, label_idx in mapping.items():
            if 0 <= int(orig_idx) < len(posterior) and 0 <= int(label_idx) < len(label_names):
                probs[int(label_idx)] += posterior[int(orig_idx)]
        total = probs.sum()
        if total <= 0:
            probs[:] = 1.0 / len(probs)
        else:
            probs /= total
        return probs

    def _apply_prob_floor(self, probs):
        """Apply mandatory floor/ceiling then renormalize."""
        floor = float(self.config.get('prob_floor_eps', 0.02))
        ceiling = float(self.config.get('prob_ceiling', 0.98))
        floor = min(max(floor, 0.0), 0.49)
        ceiling = min(max(ceiling, 0.51), 1.0)

        arr = np.array(probs, dtype=float)
        arr = np.clip(arr, floor, ceiling)
        total = float(arr.sum())
        if total <= 0.0:
            arr[:] = 1.0 / len(arr)
        else:
            arr /= total
        return arr

    def _calibrate_macro_hysteresis(self):
        """Calibrate macro thresholds from recent raw macro probabilities with structural bias checks."""
        enter = float(self.config.get('hysteresis_enter', 0.60))
        exit_ = float(self.config.get('hysteresis_exit', 0.40))
        details = {'used_dynamic': False, 'enter': enter, 'exit': exit_, 'structural_bias_correction': False}

        if not bool(self.config.get('macro_threshold_recalibration_enabled', True)):
            return enter, exit_, details
        if not self.timeline_file.exists():
            return enter, exit_, details

        try:
            cols = ['Date', 'p_fragile_raw', 'p_fragile', 'adaptive_macro_state']
            full_hist = pd.read_csv(self.timeline_file, usecols=lambda c: c in cols)
            prob_col = 'p_fragile_raw' if 'p_fragile_raw' in full_hist.columns else ('p_fragile' if 'p_fragile' in full_hist.columns else None)
            if prob_col is None or len(full_hist) < 40:
                return enter, exit_, details
                
            # Structural Bias Check Over Multiple Windows (126d, 252d, full-sample)
            structural_skew = False
            if 'adaptive_macro_state' in full_hist.columns and len(full_hist) >= 252:
                has_bias_126 = (full_hist['adaptive_macro_state'].tail(126).value_counts(normalize=True).get("Fragile", 0)) > 0.60
                has_bias_252 = (full_hist['adaptive_macro_state'].tail(252).value_counts(normalize=True).get("Fragile", 0)) > 0.60
                has_bias_full = (full_hist['adaptive_macro_state'].value_counts(normalize=True).get("Fragile", 0)) > 0.60
                
                # If all windows show structural over-classification to Fragile
                if has_bias_126 and has_bias_252 and has_bias_full:
                    structural_skew = True

            lookback = int(self.config.get('macro_threshold_recalibration_lookback_days', 756))
            hist = full_hist.copy()
            if 'Date' in hist.columns:
                hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
                hist = hist.dropna(subset=['Date'])
                if len(hist) > lookback:
                    hist = hist.tail(lookback)
            elif len(hist) > lookback:
                hist = hist.tail(lookback)

            x = pd.to_numeric(hist[prob_col], errors='coerce').dropna()
            if len(x) < 40:
                return enter, exit_, details

            q_hi = float(self.config.get('macro_threshold_recalibration_high_quantile', 0.70))
            q_lo = float(self.config.get('macro_threshold_recalibration_low_quantile', 0.30))
            
            # Recalibrate definitions directly if we are structurally biased
            if structural_skew:
                q_hi = min(0.85, q_hi + 0.10) # Make it harder to enter Fragile
                q_lo = min(0.50, q_lo + 0.05) # Make it easier to exit Fragile

            dyn_enter = float(x.quantile(q_hi))
            dyn_exit = float(x.quantile(q_lo))

            enter_min = float(self.config.get('macro_threshold_enter_min', 0.55))
            enter_max = float(self.config.get('macro_threshold_enter_max', 0.80))
            exit_min = float(self.config.get('macro_threshold_exit_min', 0.20))
            exit_max = float(self.config.get('macro_threshold_exit_max', 0.45))
            
            if structural_skew:
                enter_min += 0.05
                enter_max += 0.05

            min_gap = float(self.config.get('macro_threshold_min_gap', 0.12))

            enter = float(np.clip(dyn_enter, enter_min, enter_max))
            exit_ = float(np.clip(dyn_exit, exit_min, exit_max))
            if enter - exit_ < min_gap:
                exit_ = max(exit_min, min(exit_max, enter - min_gap))

            details = {
                'used_dynamic': True,
                'prob_col': prob_col,
                'samples': int(len(x)),
                'enter': enter,
                'exit': exit_,
                'structural_bias_correction': structural_skew
            }
            return enter, exit_, details
        except Exception:
            return float(self.config.get('hysteresis_enter', 0.60)), float(self.config.get('hysteresis_exit', 0.40)), details

    def _apply_adaptive_saturation_guardrails(self, probabilities, asof_date):
        """Cap prolonged saturation and recenter adaptive EMA when stuck near extremes."""
        sat = self.runtime_state['saturation']
        ema_a = self.runtime_state['ema_adaptive']

        hi = float(self.config.get('adaptive_saturation_high', 0.98))
        lo = float(self.config.get('adaptive_saturation_low', 0.02))
        max_days = int(self.config.get('adaptive_saturation_max_days', 40))
        recenter = float(self.config.get('adaptive_saturation_recenter_strength', 0.25))
        recenter = min(max(recenter, 0.0), 1.0)

        if float(ema_a['p_fragile_adaptive']) >= hi:
            sat['macro_high_count'] = int(sat.get('macro_high_count', 0)) + 1
            sat['macro_low_count'] = 0
        elif float(ema_a['p_fragile_adaptive']) <= lo:
            sat['macro_low_count'] = int(sat.get('macro_low_count', 0)) + 1
            sat['macro_high_count'] = 0
        else:
            sat['macro_high_count'] = 0
            sat['macro_low_count'] = 0

        recentered = {
            'macro_recentering_applied': False,
            'fast_recentering_applied': False,
            'asof_date': pd.Timestamp(asof_date).strftime('%Y-%m-%d'),
        }
        if int(sat.get('macro_high_count', 0)) >= max_days or int(sat.get('macro_low_count', 0)) >= max_days:
            p_old = float(ema_a['p_fragile_adaptive'])
            p_new = (1.0 - recenter) * p_old + recenter * 0.5
            p_new = float(np.clip(p_new, 0.0, 1.0))
            ema_a['p_fragile_adaptive'] = p_new
            sat['macro_high_count'] = 0
            sat['macro_low_count'] = 0
            recentered['macro_recentering_applied'] = True
            recentered['macro_p_fragile_before'] = p_old
            recentered['macro_p_fragile_after'] = p_new

        fast_vec = np.array([
            float(ema_a['p_calm_adaptive']),
            float(ema_a['p_choppy_adaptive']),
            float(ema_a['p_stress_adaptive']),
        ], dtype=float)
        dominant_idx = int(np.argmax(fast_vec))
        dominant_state = ['Calm', 'Choppy', 'Stress'][dominant_idx]
        if float(fast_vec[dominant_idx]) >= hi:
            if sat.get('fast_high_state') == dominant_state:
                sat['fast_high_count'] = int(sat.get('fast_high_count', 0)) + 1
            else:
                sat['fast_high_state'] = dominant_state
                sat['fast_high_count'] = 1
        else:
            sat['fast_high_state'] = None
            sat['fast_high_count'] = 0

        if int(sat.get('fast_high_count', 0)) >= max_days:
            fast_old = fast_vec.copy()
            fast_new = (1.0 - recenter) * fast_old + recenter * np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
            fast_new = self._apply_prob_floor(fast_new)
            ema_a['p_calm_adaptive'] = float(fast_new[0])
            ema_a['p_choppy_adaptive'] = float(fast_new[1])
            ema_a['p_stress_adaptive'] = float(fast_new[2])
            sat['fast_high_state'] = None
            sat['fast_high_count'] = 0
            recentered['fast_recentering_applied'] = True
            recentered['fast_before'] = [float(x) for x in fast_old.tolist()]
            recentered['fast_after'] = [float(x) for x in fast_new.tolist()]

        probabilities['p_fragile'] = float(np.clip(probabilities['p_fragile'], 0.0, 1.0))
        probabilities['p_durable'] = float(1.0 - probabilities['p_fragile'])
        return recentered

    def _compute_yearly_occupancy_alarm(self, asof_date, macro_col='adaptive_macro_state'):
        """Raise warning when yearly Durable/Fragile occupancy breaches configured cap."""
        if not bool(self.config.get('yearly_occupancy_alarm_enabled', True)):
            return None
        if not self.timeline_file.exists():
            return None

        limit = float(self.config.get('yearly_occupancy_limit', 0.90))
        try:
            hist = pd.read_csv(self.timeline_file, usecols=['Date', macro_col])
            if macro_col not in hist.columns or len(hist) == 0:
                return None
            hist['Date'] = pd.to_datetime(hist['Date'], errors='coerce')
            hist = hist.dropna(subset=['Date', macro_col])
            if len(hist) == 0:
                return None

            year = int(pd.Timestamp(asof_date).year)
            curr = hist[hist['Date'].dt.year == year]
            if len(curr) < int(self.config.get('yearly_occupancy_min_samples', 40)):
                return None

            occ = curr[macro_col].astype(str).value_counts(normalize=True)
            durable = float(occ.get('Durable', 0.0))
            fragile = float(occ.get('Fragile', 0.0))
            max_occ = max(durable, fragile)
            if max_occ <= limit:
                return None

            dominant = 'Fragile' if fragile >= durable else 'Durable'
            return {
                'year': year,
                'dominant_state': dominant,
                'durable_occupancy': durable,
                'fragile_occupancy': fragile,
                'max_occupancy': max_occ,
                'limit': limit,
            }
        except Exception:
            return None

    def _resolve_asof_date(self, run_date):
        run_ts = pd.Timestamp(run_date).normalize()
        available = self.market_data.index[self.market_data.index <= run_ts]
        if len(available) == 0:
            raise SoftFailError(f'No market data available on/before {run_ts.date()}')
        asof_date = pd.Timestamp(available[-1])
        if asof_date < run_ts:
            print(f'  Data fallback: run_date={run_ts.date()} asof_date={asof_date.date()}')
        return run_ts, asof_date

    def fetch_data(self, run_date):
        print('\n[1/7] Fetching data...')
        run_ts, asof_date = self._resolve_asof_date(run_date)
        window = self.market_data.loc[:asof_date].tail(self.lookback_days)
        print(f'  Fetched {len(window)} rows through {asof_date.date()}')
        return run_ts, asof_date, window

    def compute_features(self, asof_date):
        print('\n[2/7] Loading features...')
        available = self.features_all.index[self.features_all.index <= asof_date]
        if len(available) == 0:
            raise SoftFailError(f'No features available on/before {asof_date.date()}')
        feat_date = pd.Timestamp(available[-1])

        row = self.features_all.loc[feat_date, self.all_features]
        if row.isna().any():
            raise SoftFailError(f'Feature row has NaN on {feat_date.date()}')

        # Use as-of trailing stats for attribution (avoid full-history leakage feel).
        self._prepare_feature_stats(asof_date=feat_date)

        X_slow = row[self.slow_features].values.reshape(1, -1)
        X_fast = row[self.fast_features].values.reshape(1, -1)
        print(f'  Feature date: {feat_date.date()} | SLOW={X_slow.shape[1]} FAST={X_fast.shape[1]}')
        return feat_date, X_slow, X_fast, row

    def cascade_inference(self, X_slow, X_fast):
        print('\n[3/7] Running cascade inference...')

        slow_post = self.slow_model.predict_proba(X_slow)[0]
        macro_probs = self._label_probs(slow_post, self.slow_map, ['Durable', 'Fragile'])
        macro_probs = self._apply_prob_floor(macro_probs)
        p_durable, p_fragile = float(macro_probs[0]), float(macro_probs[1])

        fast_post_durable = self.fast_durable_model.predict_proba(X_fast)[0]
        fast_post_fragile = self.fast_fragile_model.predict_proba(X_fast)[0]

        probs_durable_ctx = self._label_probs(fast_post_durable, self.fast_durable_map, ['Calm', 'Choppy', 'Stress'])
        probs_fragile_ctx = self._label_probs(fast_post_fragile, self.fast_fragile_map, ['Calm', 'Choppy', 'Stress'])
        probs_durable_ctx = self._apply_prob_floor(probs_durable_ctx)
        probs_fragile_ctx = self._apply_prob_floor(probs_fragile_ctx)

        p_calm = p_durable * probs_durable_ctx[0] + p_fragile * probs_fragile_ctx[0]
        p_choppy = p_durable * probs_durable_ctx[1] + p_fragile * probs_fragile_ctx[1]
        p_stress = p_durable * probs_durable_ctx[2] + p_fragile * probs_fragile_ctx[2]
        fast_probs = self._apply_prob_floor([p_calm, p_choppy, p_stress])
        p_calm, p_choppy, p_stress = float(fast_probs[0]), float(fast_probs[1]), float(fast_probs[2])

        macro_raw = 'Durable' if p_durable >= p_fragile else 'Fragile'
        fast_raw = ['Calm', 'Choppy', 'Stress'][int(np.argmax([p_calm, p_choppy, p_stress]))]

        probabilities = {
            'p_durable': float(p_durable),
            'p_fragile': float(p_fragile),
            'p_calm': float(p_calm),
            'p_choppy': float(p_choppy),
            'p_stress': float(p_stress),
        }

        print(f'  Raw state: {macro_raw}–{fast_raw}')
        return macro_raw, fast_raw, probabilities

    def _update_stable_state(self, layer, candidate_state, asof_date, min_consecutive, cooldown_days, override=False):
        state = self.runtime_state[layer]
        asof_str = pd.Timestamp(asof_date).strftime('%Y-%m-%d')

        if state['last_eval_date'] != asof_str and state['cooldown'] > 0:
            state['cooldown'] = max(0, int(state['cooldown']) - 1)
        state['last_eval_date'] = asof_str

        prev_stable = state['stable']
        switched = False

        if prev_stable is None:
            state['stable'] = candidate_state
            state['candidate'] = None
            state['candidate_count'] = 0
            return state['stable'], True, prev_stable

        if candidate_state == prev_stable:
            state['candidate'] = None
            state['candidate_count'] = 0
            return prev_stable, False, prev_stable

        if override:
            state['stable'] = candidate_state
            state['candidate'] = None
            state['candidate_count'] = 0
            state['cooldown'] = int(cooldown_days)
            switched = (prev_stable != state['stable'])
            return state['stable'], switched, prev_stable

        if state['cooldown'] > 0:
            return prev_stable, False, prev_stable

        if state['candidate'] == candidate_state:
            state['candidate_count'] = int(state['candidate_count']) + 1
        else:
            state['candidate'] = candidate_state
            state['candidate_count'] = 1

        if state['candidate_count'] >= int(min_consecutive):
            state['stable'] = candidate_state
            state['candidate'] = None
            state['candidate_count'] = 0
            state['cooldown'] = int(cooldown_days)
            switched = (prev_stable != state['stable'])

        return state['stable'], switched, prev_stable

    def apply_guardrails(self, probabilities, asof_date):
        print('\n[4/7] Applying guardrails...')

        alpha_slow = float(self.config.get('ema_alpha_slow', 0.05))
        alpha_fast = float(self.config.get('ema_alpha_fast', 0.10))
        enter, exit_, threshold_meta = self._calibrate_macro_hysteresis()

        ema = self.runtime_state['ema_state']
        ema['p_fragile_smooth'] = alpha_slow * probabilities['p_fragile'] + (1 - alpha_slow) * ema['p_fragile_smooth']
        ema['p_calm_smooth'] = alpha_fast * probabilities['p_calm'] + (1 - alpha_fast) * ema['p_calm_smooth']
        ema['p_choppy_smooth'] = alpha_fast * probabilities['p_choppy'] + (1 - alpha_fast) * ema['p_choppy_smooth']
        ema['p_stress_smooth'] = alpha_fast * probabilities['p_stress'] + (1 - alpha_fast) * ema['p_stress_smooth']

        smooth_fast = self._apply_prob_floor([ema['p_calm_smooth'], ema['p_choppy_smooth'], ema['p_stress_smooth']])
        ema['p_calm_smooth'] = float(smooth_fast[0])
        ema['p_choppy_smooth'] = float(smooth_fast[1])
        ema['p_stress_smooth'] = float(smooth_fast[2])
        ema['p_fragile_smooth'] = float(np.clip(ema['p_fragile_smooth'], 0.0, 1.0))

        # ── Macro hysteresis (NB02: apply_hysteresis_macro) ──
        prev_macro = self.runtime_state['macro']['stable'] or 'Durable'
        if ema['p_fragile_smooth'] >= enter:
            macro_candidate = 'Fragile'
        elif ema['p_fragile_smooth'] <= exit_:
            macro_candidate = 'Durable'
        else:
            macro_candidate = prev_macro

        # ── Fast hysteresis (NB02: apply_hysteresis_fast — stress-priority logic) ──
        stress_enter = float(self.config.get('hysteresis_stress_enter', 0.65))
        stress_exit = float(self.config.get('hysteresis_stress_exit', 0.35))
        choppy_enter = float(self.config.get('hysteresis_choppy_enter', 0.55))
        choppy_exit = float(self.config.get('hysteresis_choppy_exit', 0.35))
        p_calm = ema['p_calm_smooth']
        p_choppy = ema['p_choppy_smooth']
        p_stress = ema['p_stress_smooth']

        prev_fast = self.runtime_state['fast']['stable'] or 'Calm'
        # STRESS has highest priority (override anything) — NB02 logic
        if p_stress > stress_enter:
            fast_candidate = 'Stress'
        elif prev_fast == 'Calm':
            if p_choppy > choppy_enter:
                fast_candidate = 'Choppy'
            else:
                fast_candidate = 'Calm'
        elif prev_fast == 'Choppy':
            if p_choppy < choppy_exit:
                fast_candidate = 'Calm'
            else:
                fast_candidate = 'Choppy'
        elif prev_fast == 'Stress':
            if p_stress < stress_exit:
                fast_candidate = 'Choppy' if p_choppy > p_calm else 'Calm'
            else:
                fast_candidate = 'Stress'
        else:
            fast_candidate = prev_fast

        # ── Min-duration: state-specific (NB02: MIN_DURATIONS_MACRO/FAST) ──
        _MACRO_MIN_DUR_DEFAULTS = {'durable': 15, 'fragile': 10}
        _FAST_MIN_DUR_DEFAULTS = {'calm': 2, 'choppy': 2, 'stress': 3}
        macro_min_dur = int(self.config.get(
            f'min_days_{macro_candidate.lower()}',
            _MACRO_MIN_DUR_DEFAULTS.get(macro_candidate.lower(), 10)
        ))
        macro_state, macro_switched, macro_prev = self._update_stable_state(
            layer='macro',
            candidate_state=macro_candidate,
            asof_date=asof_date,
            min_consecutive=macro_min_dur,
            cooldown_days=int(self.config.get('cooldown_macro_days', 2)),
            override=False,
        )

        fast_min_dur = int(self.config.get(
            f'min_days_{fast_candidate.lower()}',
            _FAST_MIN_DUR_DEFAULTS.get(fast_candidate.lower(), 2)
        ))
        stress_override = bool(self.config.get('stress_override_immediate', True) and fast_candidate == 'Stress')
        fast_state, fast_switched, fast_prev = self._update_stable_state(
            layer='fast',
            candidate_state=fast_candidate,
            asof_date=asof_date,
            min_consecutive=fast_min_dur,
            cooldown_days=int(self.config.get('cooldown_fast_days', 1)),
            override=stress_override,
        )

        # ── Adaptive-α EMA (divergence-driven: α adapts when raw diverges from smooth) ──
        gamma = float(self.config.get('adaptive_gamma', 1.5))
        alpha_max_slow = float(self.config.get('adaptive_alpha_max_slow', 0.25))
        alpha_max_fast = float(self.config.get('adaptive_alpha_max_fast', 0.40))

        ema_a = self.runtime_state['ema_adaptive']

        # Macro (slow) adaptive-α
        div_frag = abs(probabilities['p_fragile'] - ema_a['p_fragile_adaptive'])
        a_slow_adaptive = alpha_slow + (alpha_max_slow - alpha_slow) * div_frag ** gamma
        ema_a['p_fragile_adaptive'] = a_slow_adaptive * probabilities['p_fragile'] + (1 - a_slow_adaptive) * ema_a['p_fragile_adaptive']

        # Fast adaptive-α
        for prob_key, ema_key in [('p_calm', 'p_calm_adaptive'), ('p_choppy', 'p_choppy_adaptive'), ('p_stress', 'p_stress_adaptive')]:
            div = abs(probabilities[prob_key] - ema_a[ema_key])
            a_fast_adaptive = alpha_fast + (alpha_max_fast - alpha_fast) * div ** gamma
            ema_a[ema_key] = a_fast_adaptive * probabilities[prob_key] + (1 - a_fast_adaptive) * ema_a[ema_key]

        adaptive_fast = self._apply_prob_floor([ema_a['p_calm_adaptive'], ema_a['p_choppy_adaptive'], ema_a['p_stress_adaptive']])
        ema_a['p_calm_adaptive'] = float(adaptive_fast[0])
        ema_a['p_choppy_adaptive'] = float(adaptive_fast[1])
        ema_a['p_stress_adaptive'] = float(adaptive_fast[2])
        ema_a['p_fragile_adaptive'] = float(np.clip(ema_a['p_fragile_adaptive'], 0.0, 1.0))

        saturation_meta = self._apply_adaptive_saturation_guardrails(probabilities, asof_date)

        # Adaptive macro hysteresis (same thresholds, faster-tracking EMA)
        prev_macro_a = self.runtime_state['macro_adaptive']['stable'] or 'Durable'
        if ema_a['p_fragile_adaptive'] >= enter:
            macro_candidate_a = 'Fragile'
        elif ema_a['p_fragile_adaptive'] <= exit_:
            macro_candidate_a = 'Durable'
        else:
            macro_candidate_a = prev_macro_a

        # Adaptive fast hysteresis (same stress-priority logic)
        p_calm_a = ema_a['p_calm_adaptive']
        p_choppy_a = ema_a['p_choppy_adaptive']
        p_stress_a = ema_a['p_stress_adaptive']
        prev_fast_a = self.runtime_state['fast_adaptive']['stable'] or 'Calm'
        if p_stress_a > stress_enter:
            fast_candidate_a = 'Stress'
        elif prev_fast_a == 'Calm':
            fast_candidate_a = 'Choppy' if p_choppy_a > choppy_enter else 'Calm'
        elif prev_fast_a == 'Choppy':
            fast_candidate_a = 'Calm' if p_choppy_a < choppy_exit else 'Choppy'
        elif prev_fast_a == 'Stress':
            fast_candidate_a = ('Choppy' if p_choppy_a > p_calm_a else 'Calm') if p_stress_a < stress_exit else 'Stress'
        else:
            fast_candidate_a = prev_fast_a

        # Adaptive min-duration
        macro_min_dur_a = int(self.config.get(
            f'min_days_{macro_candidate_a.lower()}',
            _MACRO_MIN_DUR_DEFAULTS.get(macro_candidate_a.lower(), 10)
        ))
        adaptive_macro_state, _, _ = self._update_stable_state(
            layer='macro_adaptive',
            candidate_state=macro_candidate_a,
            asof_date=asof_date,
            min_consecutive=macro_min_dur_a,
            cooldown_days=int(self.config.get('cooldown_macro_days', 2)),
            override=False,
        )
        fast_min_dur_a = int(self.config.get(
            f'min_days_{fast_candidate_a.lower()}',
            _FAST_MIN_DUR_DEFAULTS.get(fast_candidate_a.lower(), 2)
        ))
        stress_override_a = bool(self.config.get('stress_override_immediate', True) and fast_candidate_a == 'Stress')
        adaptive_fast_state, _, _ = self._update_stable_state(
            layer='fast_adaptive',
            candidate_state=fast_candidate_a,
            asof_date=asof_date,
            min_consecutive=fast_min_dur_a,
            cooldown_days=int(self.config.get('cooldown_fast_days', 1)),
            override=stress_override_a,
        )

        guardrail_meta = {
            'macro_candidate': macro_candidate,
            'fast_candidate': fast_candidate,
            'macro_switched': macro_switched,
            'fast_switched': fast_switched,
            'macro_prev': macro_prev,
            'fast_prev': fast_prev,
            'stress_override': stress_override,
            'adaptive_macro_state': adaptive_macro_state,
            'adaptive_fast_state': adaptive_fast_state,
            'adaptive_ema': ema_a.copy(),
            'macro_thresholds': threshold_meta,
            'saturation': saturation_meta,
        }

        print(f'  Stable state: {macro_state}–{fast_state}  (adaptive: {adaptive_macro_state}–{adaptive_fast_state})')
        return macro_state, fast_state, ema.copy(), guardrail_meta

    def generate_attribution(self, feature_row):
        print('\n[5/7] Generating attribution...')

        slow_vals = feature_row[self.slow_features]
        fast_vals = feature_row[self.fast_features]

        slow_z = (slow_vals - self.feature_mu[self.slow_features]) / self.feature_std[self.slow_features]
        fast_z = (fast_vals - self.feature_mu[self.fast_features]) / self.feature_std[self.fast_features]

        top_slow = (
            pd.DataFrame({'feature': self.slow_features, 'normalized_value': slow_vals.values, 'zscore_vs_history': slow_z.values})
            .assign(abs_z=lambda d: d['zscore_vs_history'].abs())
            .sort_values('abs_z', ascending=False)
            .head(3)
            .drop(columns=['abs_z'])
            .to_dict(orient='records')
        )

        top_fast = (
            pd.DataFrame({'feature': self.fast_features, 'normalized_value': fast_vals.values, 'zscore_vs_history': fast_z.values})
            .assign(abs_z=lambda d: d['zscore_vs_history'].abs())
            .sort_values('abs_z', ascending=False)
            .head(3)
            .drop(columns=['abs_z'])
            .to_dict(orient='records')
        )

        return {'slow': top_slow, 'fast': top_fast}

    def _build_alert(self, macro_state, fast_state, guardrail_meta, occupancy_alarm=None):
        prev_combined = None
        if self.runtime_state['recent_states']:
            prev_combined = self.runtime_state['recent_states'][-1].get('combined_state')

        curr_combined = f'{macro_state}–{fast_state}'

        if guardrail_meta['fast_switched'] and fast_state == 'Stress':
            return True, 'CRITICAL', f'CRITICAL REGIME CHANGE: {guardrail_meta["fast_prev"]} -> Stress'

        if guardrail_meta['fast_switched'] and guardrail_meta['fast_prev'] == 'Calm' and fast_state == 'Choppy':
            return True, 'WARNING', f'WARNING REGIME CHANGE: Calm -> Choppy ({curr_combined})'

        if guardrail_meta['macro_switched'] or guardrail_meta['fast_switched']:
            if prev_combined and prev_combined != curr_combined:
                return True, 'INFO', f'INFO REGIME CHANGE: {prev_combined} -> {curr_combined}'
            return True, 'INFO', f'INFO REGIME CHANGE: now {curr_combined}'

        if occupancy_alarm:
            msg = (
                f'WARNING OCCUPANCY: {occupancy_alarm["year"]} '
                f'{occupancy_alarm["dominant_state"]} occupancy '
                f'{occupancy_alarm["max_occupancy"]:.1%} exceeds '
                f'limit {occupancy_alarm["limit"]:.1%}'
            )
            return True, 'WARNING', msg

        return False, None, None

    def save_outputs(self, run_date, asof_date, feat_date, macro_state, fast_state, probabilities, probabilities_raw, attribution, guardrail_meta):
        print('\n[6/7] Saving outputs...')

        combined_state = f'{macro_state}–{fast_state}'
        p_fragile = float(probabilities.get('p_fragile_smooth', 0.5))
        p_calm = float(probabilities.get('p_calm_smooth', 1.0 / 3.0))
        p_choppy = float(probabilities.get('p_choppy_smooth', 1.0 / 3.0))
        p_stress = float(probabilities.get('p_stress_smooth', 1.0 / 3.0))
        p_durable = float(1.0 - p_fragile)
        macro_confidence = float(max(p_durable, p_fragile))
        fast_confidence = float(max(p_calm, p_choppy, p_stress))
        combined_confidence = float(min(macro_confidence, fast_confidence))

        daily_output = {
            'run_date': pd.Timestamp(run_date).strftime('%Y-%m-%d'),
            'asof_date': pd.Timestamp(asof_date).strftime('%Y-%m-%d'),
            'feature_date': pd.Timestamp(feat_date).strftime('%Y-%m-%d'),
            'macro_state': macro_state,
            'fast_state': fast_state,
            'combined_state': combined_state,
            'probabilities': probabilities,
            'p_durable_smooth': p_durable,
            'p_fragile_smooth': p_fragile,
            'p_calm_smooth': p_calm,
            'p_choppy_smooth': p_choppy,
            'p_stress_smooth': p_stress,
            'macro_confidence': macro_confidence,
            'fast_confidence': fast_confidence,
            'confidence': combined_confidence,
            'guardrails': guardrail_meta,
            'attribution': attribution,
            'timestamp': datetime.now().isoformat(),
        }

        # Individual daily JSON files removed to reduce clutter
        # Backend now reads from daily_inference_state.json

        # Raw (unsmoothed) HMM probabilities for early-warning view
        raw_cols = {
            'p_fragile_raw': float(probabilities_raw.get('p_fragile', 0.5)),
            'p_calm_raw': float(probabilities_raw.get('p_calm', 1.0 / 3.0)),
            'p_choppy_raw': float(probabilities_raw.get('p_choppy', 1.0 / 3.0)),
            'p_stress_raw': float(probabilities_raw.get('p_stress', 1.0 / 3.0)),
        }

        # Adaptive-α probabilities and regime labels
        adaptive_ema = guardrail_meta.get('adaptive_ema', {})
        adaptive_cols = {}
        if adaptive_ema:
            a_macro = guardrail_meta.get('adaptive_macro_state', '')
            a_fast = guardrail_meta.get('adaptive_fast_state', '')
            adaptive_cols = {
                'p_fragile_adaptive': float(adaptive_ema.get('p_fragile_adaptive', 0.5)),
                'p_calm_adaptive': float(adaptive_ema.get('p_calm_adaptive', 1.0 / 3.0)),
                'p_choppy_adaptive': float(adaptive_ema.get('p_choppy_adaptive', 1.0 / 3.0)),
                'p_stress_adaptive': float(adaptive_ema.get('p_stress_adaptive', 1.0 / 3.0)),
                'adaptive_macro_state': a_macro,
                'adaptive_fast_state': a_fast,
                'adaptive_combined_state': f'{a_macro}\u2013{a_fast}',
            }

        row = pd.DataFrame(
            [
                {
                    'Date': pd.Timestamp(asof_date),
                    'run_date': pd.Timestamp(run_date),
                    'feature_date': pd.Timestamp(feat_date),
                    'macro_state': macro_state,
                    'fast_state': fast_state,
                    'combined_state': combined_state,
                    'macro_confidence': macro_confidence,
                    'fast_confidence': fast_confidence,
                    'combined_confidence': combined_confidence,
                    'confidence': combined_confidence,
                    **probabilities,
                    **raw_cols,
                    **adaptive_cols,
                }
            ]
        )
        self.timeline_rows_dir.mkdir(parents=True, exist_ok=True)
        row_file = self.timeline_rows_dir / f"{pd.Timestamp(asof_date).strftime('%Y%m%d')}.csv"
        self._atomic_write_csv(row_file, row)
        self._append_timeline_row(row)

        self.runtime_state['recent_states'].append(
            {
                'date': pd.Timestamp(asof_date).strftime('%Y-%m-%d'),
                'macro_state': macro_state,
                'fast_state': fast_state,
                'combined_state': combined_state,
            }
        )
        self.runtime_state['recent_states'] = self.runtime_state['recent_states'][-self.state_history_window:]
        self.runtime_state['last_processed_date'] = pd.Timestamp(asof_date).strftime('%Y-%m-%d')
        self._save_runtime_state()

        occupancy_alarm = self._compute_yearly_occupancy_alarm(asof_date=asof_date)

        return None, self.timeline_file, occupancy_alarm  # No longer saving individual daily JSON files

    def run(self, run_date):
        stage_times = {}
        total_start = time.time()

        print('\n' + '=' * 80)
        print(f'DAILY INFERENCE PIPELINE: run_date={pd.Timestamp(run_date).date()}')
        print('=' * 80)

        try:
            if not self.skip_lock:
                self._acquire_lock()
            lock_acquired = True
        except SoftFailError as e:
            payload = {
                'run_date': pd.Timestamp(run_date).strftime('%Y-%m-%d'),
                'status': 'SOFT_FAIL',
                'status_code': 10,
                'error': str(e),
                'elapsed_seconds': time.time() - total_start,
                'timestamp': datetime.now().isoformat(),
            }
            self._append_run_log(payload)
            print(f'SOFT FAIL: {e}')
            return payload

        try:
            t = time.time()
            run_ts, asof_date, market_window = self.fetch_data(run_date)
            stage_times['fetch_seconds'] = time.time() - t

            t = time.time()
            feat_date, X_slow, X_fast, feature_row = self.compute_features(asof_date)
            stage_times['features_seconds'] = time.time() - t

            t = time.time()
            macro_raw, fast_raw, probabilities_raw = self.cascade_inference(X_slow, X_fast)
            stage_times['infer_seconds'] = time.time() - t

            t = time.time()
            macro_state, fast_state, probabilities_smooth, guardrail_meta = self.apply_guardrails(probabilities_raw, asof_date)
            stage_times['guardrails_seconds'] = time.time() - t

            t = time.time()
            attribution = self.generate_attribution(feature_row)
            stage_times['attribution_seconds'] = time.time() - t

            t = time.time()
            daily_file, timeline_file, occupancy_alarm = self.save_outputs(
                run_date=run_ts,
                asof_date=asof_date,
                feat_date=feat_date,
                macro_state=macro_state,
                fast_state=fast_state,
                probabilities=probabilities_smooth,
                probabilities_raw=probabilities_raw,
                attribution=attribution,
                guardrail_meta=guardrail_meta,
            )
            stage_times['save_seconds'] = time.time() - t

            t = time.time()
            changed, severity, alert_message = self._build_alert(
                macro_state,
                fast_state,
                guardrail_meta,
                occupancy_alarm=occupancy_alarm,
            )
            stage_times['alert_seconds'] = time.time() - t

            elapsed = time.time() - total_start
            result = {
                'run_date': run_ts.strftime('%Y-%m-%d'),
                'asof_date': pd.Timestamp(asof_date).strftime('%Y-%m-%d'),
                'feature_date': pd.Timestamp(feat_date).strftime('%Y-%m-%d'),
                'macro_state_raw': macro_raw,
                'fast_state_raw': fast_raw,
                'macro_state': macro_state,
                'fast_state': fast_state,
                'combined_state': f'{macro_state}–{fast_state}',
                'regime_changed': changed,
                'alert_severity': severity,
                'alert_message': alert_message,
                'occupancy_alarm': occupancy_alarm,
                'stage_times': stage_times,
                'elapsed_seconds': elapsed,
                'status': 'SUCCESS',
                'status_code': 0,
                'daily_file': str(daily_file),
                'timeline_file': str(timeline_file),
                'timestamp': datetime.now().isoformat(),
            }
            self._append_run_log(result)

            print('\n' + '=' * 80)
            print(f'PIPELINE COMPLETE: {elapsed:.2f} seconds')
            print(f'  Stable regime: {result["combined_state"]}')
            if changed:
                print(f'  {severity}: {alert_message}')
            print('=' * 80)
            return result

        except SoftFailError as e:
            elapsed = time.time() - total_start
            result = {
                'run_date': pd.Timestamp(run_date).strftime('%Y-%m-%d'),
                'status': 'SOFT_FAIL',
                'status_code': 10,
                'error': str(e),
                'stage_times': stage_times,
                'elapsed_seconds': elapsed,
                'timestamp': datetime.now().isoformat(),
            }
            self._append_run_log(result)
            print(f'SOFT FAIL: {e}')
            return result

        except Exception as e:
            elapsed = time.time() - total_start
            result = {
                'run_date': pd.Timestamp(run_date).strftime('%Y-%m-%d'),
                'status': 'HARD_FAIL',
                'status_code': 1,
                'error': str(e),
                'stage_times': stage_times,
                'elapsed_seconds': elapsed,
                'timestamp': datetime.now().isoformat(),
            }
            self._append_run_log(result)
            print(f'HARD FAIL: {e}')
            return result

        finally:
            if 'lock_acquired' in locals() and lock_acquired and not self.skip_lock:
                self._release_lock()
