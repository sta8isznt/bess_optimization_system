"""Central project paths used by package services and CLIs."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
CLEANED_DATA_DIR = DATA_DIR / "cleaned_data"
HENEX_DATA_DIR = DATA_DIR / "HeNex_data"
DEFAULT_PRICE_SIGNALS_PATH = CLEANED_DATA_DIR / "price_signals_15m.csv"
DEFAULT_DEGRADATION_LUT_PATH = CLEANED_DATA_DIR / "Reduced_LUT_Final.csv"
DEFAULT_PYBAMM_LUT_PATH = CLEANED_DATA_DIR / "Reduced_LUT_PyBaMM.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FORECAST_OUTPUT_DIR = OUTPUTS_DIR / "forecasts"
DAILY_OUTPUT_DIR = OUTPUTS_DIR / "daily"
ANNUAL_OUTPUT_DIR = OUTPUTS_DIR / "annual"
FORECAST_BACKTEST_OUTPUT_DIR = OUTPUTS_DIR / "forecast_backtests"
PYBAMM_OUTPUT_DIR = OUTPUTS_DIR / "pybamm_lut"
