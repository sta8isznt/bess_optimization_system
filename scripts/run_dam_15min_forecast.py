"""CLI wrapper for the Greek DAM 15-minute forecasting baseline."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from optimization.forecasting.dam_15min_forecast import main


if __name__ == "__main__":
    raise SystemExit(main())
