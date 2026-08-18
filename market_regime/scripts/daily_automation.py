#!/usr/bin/env python3
"""
One-command daily automation runner.

Runs in order:
1) scripts/build_features.py (optional, via --build-features)
2) scripts/daily_inference.py
3) scripts/daily_ops.py
4) scripts/run_retrain_if_triggered.py
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


MARKET_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MARKET_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "market_regime"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def append_log(payload):
    with open(LOGS_DIR / "daily_automation_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily inference + ops + retrain trigger")
    parser.add_argument("--date", type=str, default=None, help="Run date YYYY-MM-DD")
    parser.add_argument(
        "--build-features",
        action="store_true",
        help="Run scripts/build_features.py before inference",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main():
    args = parse_args()
    ts = datetime.now().isoformat()
    py = sys.executable

    commands = []
    if args.build_features:
        commands.append([py, str(MARKET_DIR / "scripts" / "build_features.py")])
    commands.extend(
        [
            [py, str(MARKET_DIR / "scripts" / "daily_inference.py")],
            [py, str(MARKET_DIR / "scripts" / "daily_ops.py")],
            [py, str(MARKET_DIR / "scripts" / "run_retrain_if_triggered.py")],
        ]
    )
    if args.date:
        inf_idx = 1 if args.build_features else 0
        ops_idx = 2 if args.build_features else 1
        commands[inf_idx] += ["--date", args.date]
        commands[ops_idx] += ["--date", args.date]
        # trigger runner reads from daily_ops output; no date arg needed.

    if args.dry_run:
        payload = {"timestamp": ts, "status": "SUCCESS", "result": "DRY_RUN", "commands": commands}
        append_log(payload)
        print(json.dumps(payload, indent=2))
        return 0

    steps = []
    for idx, cmd in enumerate(commands, start=1):
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        steps.append(
            {
                "step": idx,
                "command": cmd,
                "returncode": int(proc.returncode),
                "stdout_tail": proc.stdout[-3000:],
                "stderr_tail": proc.stderr[-3000:],
            }
        )
        # Stop only on hard failure (1+). Soft fail 10 is accepted for inference/ops.
        if proc.returncode not in (0, 10):
            payload = {
                "timestamp": ts,
                "status": "FAILED",
                "error_step": idx,
                "steps": steps,
            }
            append_log(payload)
            print(json.dumps(payload, indent=2))
            return 1

    payload = {"timestamp": ts, "status": "SUCCESS", "steps": steps}
    append_log(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
