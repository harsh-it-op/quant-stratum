"""
Retroactively compute adaptive-alpha EMA columns for the regime timeline CSV.

This script reads the existing regime_timeline_history.csv (which already has
raw HMM probabilities and fixed-alpha smoothed probabilities), and computes the
adaptive-alpha EMA + hysteresis + min-duration columns forward through the full
history. This is much faster than re-running the entire HMM inference pipeline.

Usage:
    python scripts/compute_adaptive_alpha.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMELINE_FILE = ROOT / "features" / "regime_timeline_history.csv"

# ── Adaptive-alpha parameters ──
ALPHA_SLOW_BASE = 0.05
ALPHA_FAST_BASE = 0.10
ALPHA_MAX_SLOW = 0.25
ALPHA_MAX_FAST = 0.40
GAMMA = 1.5

# ── Hysteresis thresholds (mirror runtime) ──
MACRO_ENTER = 0.60
MACRO_EXIT = 0.40
STRESS_ENTER = 0.65
STRESS_EXIT = 0.35
CHOPPY_ENTER = 0.55
CHOPPY_EXIT = 0.35

# ── Min-duration thresholds ──
MACRO_MIN_DUR = {'Durable': 15, 'Fragile': 10}
FAST_MIN_DUR = {'Calm': 2, 'Choppy': 2, 'Stress': 3}
COOLDOWN_MACRO = 2
COOLDOWN_FAST = 1
STRESS_OVERRIDE_IMMEDIATE = True


def run_adaptive_alpha():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print(f"Reading {TIMELINE_FILE} ...")
    df = pd.read_csv(TIMELINE_FILE, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    n = len(df)
    print(f"  {n} rows loaded")

    has_raw = "p_fragile_raw" in df.columns
    if not has_raw:
        print("ERROR: CSV missing raw probability columns. Run a full backfill first.")
        return

    # ── Initialize adaptive EMA state ──
    ema_a = {
        'p_fragile': 0.5,
        'p_calm': 1 / 3,
        'p_choppy': 1 / 3,
        'p_stress': 1 / 3,
    }

    # ── Min-duration state trackers ──
    macro_state = {'stable': None, 'candidate': None, 'count': 0, 'cooldown': 0, 'last_date': None}
    fast_state = {'stable': None, 'candidate': None, 'count': 0, 'cooldown': 0, 'last_date': None}

    # Output columns
    out_fragile = np.zeros(n)
    out_calm = np.zeros(n)
    out_choppy = np.zeros(n)
    out_stress = np.zeros(n)
    out_macro = [""] * n
    out_fast = [""] * n
    out_combined = [""] * n

    for i in range(n):
        row = df.iloc[i]
        date_str = pd.Timestamp(row["Date"]).strftime("%Y-%m-%d")

        # Raw probabilities for this day
        p_frag_raw = float(row.get("p_fragile_raw", 0.5))
        p_calm_raw = float(row.get("p_calm_raw", 1 / 3))
        p_choppy_raw = float(row.get("p_choppy_raw", 1 / 3))
        p_stress_raw = float(row.get("p_stress_raw", 1 / 3))

        # ── Adaptive-alpha EMA update ──
        # Macro (slow)
        div_frag = abs(p_frag_raw - ema_a['p_fragile'])
        a_slow = ALPHA_SLOW_BASE + (ALPHA_MAX_SLOW - ALPHA_SLOW_BASE) * div_frag ** GAMMA
        ema_a['p_fragile'] = a_slow * p_frag_raw + (1 - a_slow) * ema_a['p_fragile']

        # Fast
        for raw_val, key in [(p_calm_raw, 'p_calm'), (p_choppy_raw, 'p_choppy'), (p_stress_raw, 'p_stress')]:
            div = abs(raw_val - ema_a[key])
            a_fast = ALPHA_FAST_BASE + (ALPHA_MAX_FAST - ALPHA_FAST_BASE) * div ** GAMMA
            ema_a[key] = a_fast * raw_val + (1 - a_fast) * ema_a[key]

        # Renormalize fast triple onto the probability simplex — different α per state
        # otherwise causes cumulative drift away from sum=1.0 (mirrors _apply_prob_floor in runtime).
        _s = ema_a['p_calm'] + ema_a['p_choppy'] + ema_a['p_stress']
        if _s > 0:
            ema_a['p_calm'] /= _s
            ema_a['p_choppy'] /= _s
            ema_a['p_stress'] /= _s

        out_fragile[i] = ema_a['p_fragile']
        out_calm[i] = ema_a['p_calm']
        out_choppy[i] = ema_a['p_choppy']
        out_stress[i] = ema_a['p_stress']

        # ── Macro hysteresis ──
        prev_macro = macro_state['stable'] or 'Durable'
        if ema_a['p_fragile'] >= MACRO_ENTER:
            macro_cand = 'Fragile'
        elif ema_a['p_fragile'] <= MACRO_EXIT:
            macro_cand = 'Durable'
        else:
            macro_cand = prev_macro

        # ── Fast hysteresis (stress-priority) ──
        prev_fast = fast_state['stable'] or 'Calm'
        if ema_a['p_stress'] > STRESS_ENTER:
            fast_cand = 'Stress'
        elif prev_fast == 'Calm':
            fast_cand = 'Choppy' if ema_a['p_choppy'] > CHOPPY_ENTER else 'Calm'
        elif prev_fast == 'Choppy':
            fast_cand = 'Calm' if ema_a['p_choppy'] < CHOPPY_EXIT else 'Choppy'
        elif prev_fast == 'Stress':
            if ema_a['p_stress'] < STRESS_EXIT:
                fast_cand = 'Choppy' if ema_a['p_choppy'] > ema_a['p_calm'] else 'Calm'
            else:
                fast_cand = 'Stress'
        else:
            fast_cand = prev_fast

        # ── Min-duration logic ──
        out_macro[i] = _update_stable(macro_state, macro_cand, date_str,
                                       MACRO_MIN_DUR.get(macro_cand, 10), COOLDOWN_MACRO, override=False)
        stress_override = STRESS_OVERRIDE_IMMEDIATE and fast_cand == 'Stress'
        out_fast[i] = _update_stable(fast_state, fast_cand, date_str,
                                      FAST_MIN_DUR.get(fast_cand, 2), COOLDOWN_FAST, override=stress_override)
        out_combined[i] = f"{out_macro[i]}\u2013{out_fast[i]}"

    # ── Write back to CSV ──
    df['p_fragile_adaptive'] = out_fragile
    df['p_calm_adaptive'] = out_calm
    df['p_choppy_adaptive'] = out_choppy
    df['p_stress_adaptive'] = out_stress
    df['adaptive_macro_state'] = out_macro
    df['adaptive_fast_state'] = out_fast
    df['adaptive_combined_state'] = out_combined

    df.to_csv(TIMELINE_FILE, index=False)
    print(f"  Wrote {n} rows with adaptive-alpha columns to {TIMELINE_FILE}")

    # ── Summary stats ──
    print("\n-- Adaptive-alpha Summary --")
    for col in ['adaptive_macro_state', 'adaptive_fast_state', 'adaptive_combined_state']:
        print(f"\n  {col} distribution:")
        for val, cnt in df[col].value_counts().items():
            print(f"    {val}: {cnt} ({cnt/n*100:.1f}%)")

    # Compare regime-change count: fixed vs adaptive
    fixed_changes = (df['combined_state'] != df['combined_state'].shift()).sum() - 1
    adaptive_changes = (df['adaptive_combined_state'] != df['adaptive_combined_state'].shift()).sum() - 1
    print(f"\n  Regime changes (fixed-alpha):    {fixed_changes}")
    print(f"  Regime changes (adaptive-alpha): {adaptive_changes}")
    print(f"  Ratio: {adaptive_changes / max(fixed_changes, 1):.2f}x")


def _update_stable(state, candidate, date_str, min_consecutive, cooldown_days, override=False):
    """Min-duration state machine (mirrors runtime _update_stable_state)."""
    if state['last_date'] != date_str and state['cooldown'] > 0:
        state['cooldown'] = max(0, state['cooldown'] - 1)
    state['last_date'] = date_str

    prev = state['stable']

    if prev is None:
        state['stable'] = candidate
        state['candidate'] = None
        state['count'] = 0
        return state['stable']

    if candidate == prev:
        state['candidate'] = None
        state['count'] = 0
        return prev

    if override:
        state['stable'] = candidate
        state['candidate'] = None
        state['count'] = 0
        state['cooldown'] = cooldown_days
        return state['stable']

    if state['cooldown'] > 0:
        return prev

    if state['candidate'] == candidate:
        state['count'] += 1
    else:
        state['candidate'] = candidate
        state['count'] = 1

    if state['count'] >= min_consecutive:
        state['stable'] = candidate
        state['candidate'] = None
        state['count'] = 0
        state['cooldown'] = cooldown_days

    return state['stable']


if __name__ == "__main__":
    run_adaptive_alpha()
