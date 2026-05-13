"""Thin Streamlit adapter around the shared BESS optimization services."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bess_optimization.io.degradation import (  # noqa: E402
    default_lut_for_source,
    list_lut_files,
)
from bess_optimization.models import OptimizationResult  # noqa: E402
from bess_optimization.services.optimization import (  # noqa: E402
    DashboardOptimizerError,
    available_dates,
    available_years,
    display_path,
    list_price_files,
    load_price_series,
    run_annual_optimization as _run_annual_optimization,
    run_daily_optimization as _run_daily_optimization,
    validate_parameters,
)


def run_daily_optimization(*args, **kwargs) -> OptimizationResult:
    return _run_daily_optimization(*args, **kwargs)


def run_annual_optimization(*args, **kwargs) -> OptimizationResult:
    return _run_annual_optimization(*args, **kwargs)
