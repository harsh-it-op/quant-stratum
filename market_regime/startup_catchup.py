"""
Startup Catch-up Script for ML Regime Classifier
Runs at system startup and checks if today's pipeline has executed.
If not, runs it automatically.
"""

import json
import sys
import subprocess
from datetime import datetime
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "daily_automation_runs.jsonl"


def get_last_run_date():
    """Get the date of the last successful run."""
    if not LOG_FILE.exists():
        return None
    
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
            
        # Read from end to find last SUCCESS
        for line in reversed(lines):
            try:
                entry = json.loads(line.strip())
                if entry.get('status') == 'SUCCESS':
                    # Parse timestamp: "2026-03-01T07:30:15"
                    timestamp = entry.get('timestamp', '')
                    run_date = datetime.fromisoformat(timestamp).date()
                    return run_date
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
                
        return None
    except Exception as e:
        print(f"Error reading log: {e}")
        return None


def should_run_today():
    """Check if we need to run today's pipeline."""
    today = datetime.now().date()
    last_run = get_last_run_date()
    
    if last_run is None:
        print(f"[CATCHUP] No previous runs found. Will execute pipeline.")
        return True
    
    if last_run < today:
        print(f"[CATCHUP] Last run: {last_run}, Today: {today}. Pipeline needs to run.")
        return True
    
    print(f"[CATCHUP] Pipeline already ran today ({today}). Skipping.")
    return False


def run_daily_pipeline():
    """Execute the daily automation pipeline."""
    print("[CATCHUP] Starting daily automation pipeline...")
    
    try:
        # Run with venv Python
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
        script = PROJECT_ROOT / "scripts" / "daily_automation.py"
        
        result = subprocess.run(
            [str(venv_python), str(script), "--build-features"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print("[CATCHUP] ✓ Pipeline completed successfully!")
            return True
        else:
            print(f"[CATCHUP] ✗ Pipeline failed with code {result.returncode}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("[CATCHUP] ✗ Pipeline timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"[CATCHUP] ✗ Error running pipeline: {e}")
        return False


def main():
    print("="*60)
    print("ML Regime Classifier - Startup Catch-up Check")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {PROJECT_ROOT}")
    print("-"*60)
    
    if should_run_today():
        success = run_daily_pipeline()
        sys.exit(0 if success else 1)
    else:
        print("[CATCHUP] Nothing to do. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
