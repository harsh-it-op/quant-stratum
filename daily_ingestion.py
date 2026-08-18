#!/usr/bin/env python3
"""
Compatibility wrapper for legacy daily ingestion entrypoint.

Canonical daily runtime is now:
    python scripts/daily_inference.py [args]
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.daily_inference import main


if __name__ == '__main__':
    print('INFO: scripts/daily_ingestion.py is deprecated; forwarding to scripts/daily_inference.py')
    sys.exit(main())
