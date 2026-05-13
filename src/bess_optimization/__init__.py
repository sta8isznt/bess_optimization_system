"""Source package for the BESS optimization workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models import (
    BatteryConfig,
    DegradationCurve,
    ForecastBacktestRequest,
    ForecastBacktestResult,
    ForecastRequest,
    ForecastResult,
    OptimizationRequest,
    OptimizationResult,
    OptimizationRunMode,
)

if TYPE_CHECKING:
    from .degradation.pybamm_lut import PyBaMMLutConfig


def __getattr__(name: str):
    if name == "PyBaMMLutConfig":
        from .degradation.pybamm_lut import PyBaMMLutConfig

        return PyBaMMLutConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_forecast(request: ForecastRequest) -> ForecastResult:
    from .services.forecasting import run_forecast as _run_forecast

    return _run_forecast(request)


def run_optimization(request: OptimizationRequest) -> OptimizationResult:
    from .services.optimization import run_optimization as _run_optimization

    return _run_optimization(request)


def run_forecast_strategy_backtest(request: ForecastBacktestRequest) -> ForecastBacktestResult:
    from .services.forecast_strategy_backtest import (
        run_forecast_strategy_backtest as _run_forecast_strategy_backtest,
    )

    return _run_forecast_strategy_backtest(**request.__dict__)


def build_pybamm_lut(
    config: PyBaMMLutConfig,
    diagnostics_output_dir=None,
    optimizer_output_dir=None,
) -> dict:
    from .degradation.pybamm_lut import (
        DEFAULT_DIAGNOSTICS_OUTPUT_DIR,
        DEFAULT_OPTIMIZER_OUTPUT_DIR,
        run_pipeline,
    )

    return run_pipeline(
        diagnostics_output_dir or DEFAULT_DIAGNOSTICS_OUTPUT_DIR,
        optimizer_output_dir or DEFAULT_OPTIMIZER_OUTPUT_DIR,
        config,
    )


__all__ = [
    "BatteryConfig",
    "DegradationCurve",
    "ForecastBacktestRequest",
    "ForecastBacktestResult",
    "ForecastRequest",
    "ForecastResult",
    "OptimizationRequest",
    "OptimizationResult",
    "OptimizationRunMode",
    "PyBaMMLutConfig",
    "build_pybamm_lut",
    "run_forecast",
    "run_forecast_strategy_backtest",
    "run_optimization",
]
