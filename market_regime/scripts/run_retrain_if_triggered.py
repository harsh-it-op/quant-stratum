#!/usr/bin/env python3
"""
Trigger quarterly retraining from daily ops decision output.

Reads: output/daily_operations/daily_ops_latest.json
If decision.should_retrain=true, runs:
    python scripts/quarterly_retrain.py --emergency --reason <reason> --as-of-date <date>

Cooldown logic:
    After a retrain is rejected by the validation gate, a cooldown is written to
    logs/retrain_cooldown.json. Non-emergency retrain triggers are suppressed during
    the cooldown window (default: 7 days) to avoid hammering the system every day
    when drift is persistent but new models keep failing validation.
    Emergency triggers (crash, VIX explosion) always bypass the cooldown.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


MARKET_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MARKET_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "market_regime"
LOGS_DIR = PROJECT_ROOT / "logs" / "market_regime"
COOLDOWN_FILE = LOGS_DIR / "retrain_cooldown.json"

# Triggers that bypass the cooldown entirely (market emergencies)
EMERGENCY_REASONS = {"emergency", "emergency_crash", "emergency_vix"}

# Days to wait after a rejected retrain before trying again for drift/health triggers
COOLDOWN_DAYS = 7


def append_log(payload):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "retrain_trigger_runner.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def read_cooldown():
    """Return cooldown dict if active, else None."""
    if not COOLDOWN_FILE.exists():
        return None
    try:
        doc = json.loads(COOLDOWN_FILE.read_text(encoding="utf-8"))
        until = datetime.fromisoformat(doc["cooldown_until"])
        if datetime.now() < until:
            return doc  # still in cooldown
        # Expired — clean it up
        COOLDOWN_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return None


def write_cooldown(reason, rejected_at, attempts):
    """Write a cooldown marker after a rejection."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    until = datetime.now() + timedelta(days=COOLDOWN_DAYS)
    doc = {
        "triggered_by": reason,
        "rejected_at": rejected_at,
        "cooldown_until": until.isoformat(),
        "cooldown_days": COOLDOWN_DAYS,
        "attempts": attempts,
    }
    COOLDOWN_FILE.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def clear_cooldown():
    """Remove cooldown after a successful deploy."""
    COOLDOWN_FILE.unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrain script if daily ops requested it")
    parser.add_argument("--ops-file", type=str, default=str(OUTPUT_DIR / "daily_operations" / "daily_ops_latest.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print command but do not execute")
    return parser.parse_args()


def main():
    args = parse_args()
    ops_path = Path(args.ops_file)
    ts = datetime.now().isoformat()

    if not ops_path.exists():
        payload = {"timestamp": ts, "status": "SOFT_FAIL", "error": f"Ops file not found: {ops_path}"}
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 10

    data = json.loads(ops_path.read_text(encoding="utf-8"))
    decision = data.get("decision", {})
    should_retrain = bool(decision.get("should_retrain", False))
    reason = str(decision.get("primary_reason", "trigger"))
    as_of_date = str(data.get("as_of_date") or data.get("run_date") or datetime.now().date())

    if not should_retrain:
        payload = {
            "timestamp": ts,
            "status": "SUCCESS",
            "result": "NO_TRIGGER",
            "as_of_date": as_of_date,
            "reason": reason,
        }
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 0

    # ── Cooldown check ────────────────────────────────────────────────────────
    # Emergency triggers (crash / VIX explosion) always run regardless.
    # Drift / health / quarterly triggers are suppressed during cooldown.
    is_emergency = any(reason.startswith(e) for e in EMERGENCY_REASONS)
    if not is_emergency:
        cooldown = read_cooldown()
        if cooldown:
            payload = {
                "timestamp": ts,
                "status": "SUCCESS",
                "result": "COOLDOWN_SUPPRESSED",
                "as_of_date": as_of_date,
                "reason": reason,
                "cooldown": cooldown,
                "message": (
                    f"Retrain suppressed: last attempt was rejected on "
                    f"{cooldown['rejected_at'][:10]}, cooldown active until "
                    f"{cooldown['cooldown_until'][:10]} ({COOLDOWN_DAYS} days after rejection)"
                ),
            }
            append_log(payload)
            print(json.dumps(payload, indent=2))
            return 0
    # ── End cooldown check ────────────────────────────────────────────────────

    cmd = [
        sys.executable,
        str(MARKET_DIR / "scripts" / "quarterly_retrain.py"),
        "--emergency",
        "--reason",
        reason,
        "--as-of-date",
        as_of_date,
    ]

    if args.dry_run:
        payload = {
            "timestamp": ts,
            "status": "SUCCESS",
            "result": "DRY_RUN",
            "command": cmd,
        }
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 0

    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

    # Determine if this was a "retrain rejected by validation" (expected) vs a real crash.
    # Rejection is NOT a pipeline failure — the validation gate did its job.
    retrain_rejected = False
    if proc.returncode != 0:
        if '"result": "REJECTED"' in proc.stdout or '"REJECTED"' in proc.stdout:
            retrain_rejected = True

    if retrain_rejected:
        # Count how many attempts were made (for logging)
        attempts = proc.stdout.count('"max_attempts"')
        cooldown_doc = write_cooldown(reason, ts, attempts)
        status = "SUCCESS"
        result = "RETRAIN_REJECTED"
        exit_code = 0
    elif proc.returncode == 0:
        clear_cooldown()  # Successful deploy resets cooldown
        status = "SUCCESS"
        result = "RETRAIN_DEPLOYED"
        cooldown_doc = None
        exit_code = 0
    else:
        status = "FAILED"
        result = "RETRAIN_CRASHED"
        cooldown_doc = None
        exit_code = 1

    payload = {
        "timestamp": ts,
        "status": status,
        "result": result,
        "retrain_rejected": retrain_rejected,
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if retrain_rejected and cooldown_doc:
        payload["cooldown"] = cooldown_doc
        payload["cooldown_message"] = (
            f"Drift retrain suppressed for {COOLDOWN_DAYS} days "
            f"(until {cooldown_doc['cooldown_until'][:10]}). "
            f"Emergency triggers still active."
        )
    append_log(payload)
    print(json.dumps(payload, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

