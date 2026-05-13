"""Shared request and result models for the BESS workflows.

This module is the data contract layer of the project. It defines the
configuration, request, and result objects that move between the forecasting,
optimization, backtesting, CLI, and dashboard code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from enum import StrEnum

import pandas as pd


class TerminalSoCMode(StrEnum):
    """
    Allowed final state-of-charge rules for an optimization run.

    EQUAL_INITIAL forces the final SoC to equal the initial SoC.
    FREE leaves the final SoC unconstrained inside the normal SoC bounds.
    """

    EQUAL_INITIAL = "equal_initial"
    FREE = "free"


class OptimizationRunMode(StrEnum):
    """Supported optimization horizons."""

    DAILY = "daily"
    ANNUAL = "annual"


@dataclass(frozen=True)
class BatteryConfig:
    """
    Physical and operational battery parameters used by the optimizer.

    The defaults describe a 1 MW / 2 MWh battery, so the default duration is
    two hours. The workflow can later scale results to a larger park size
    without changing the optimization model itself.

    Attributes:
        p_max: Maximum charge/discharge power in MW.
        e_max: Maximum energy capacity in MWh.
        eta_ch: Charging efficiency.
        eta_dis: Discharging efficiency.
        soc_min: Minimum allowed state of charge as a fraction of e_max (%).
        soc_max: Maximum allowed state of charge as a fraction of e_max (%).
        soc_init: Initial state of charge as a fraction of e_max (%).
        dt: Time step duration in hours. For 15-minute intervals, dt = 0.25.
        terminal_soc_mode: Rule for final SoC at the end of the horizon.
    """
    p_max: float = 1.0
    e_max: float = 2.0
    eta_ch: float = 0.92
    eta_dis: float = 0.92
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_init: float = 0.50
    dt: float = 0.25
    terminal_soc_mode: TerminalSoCMode = TerminalSoCMode.EQUAL_INITIAL

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "terminal_soc_mode",
            TerminalSoCMode(str(self.terminal_soc_mode).strip().lower()),
        )

    def as_dict(self) -> dict:
        """
        Return optimizer parameters as a plain dictionary.

        The enum field is converted to its string value so the result is easy
        to pass into existing dictionary-based optimizer code and reports.
        """

        data = asdict(self)
        data["terminal_soc_mode"] = str(self.terminal_soc_mode)
        return data


@dataclass(frozen=True)
class DegradationCurve:
    """
    Piecewise degradation cost curve passed to the MILP optimizer.

    energy_points and cost_points define matching x/y breakpoints for the
    dispatch-energy-to-degradation-cost relationship. source_label records
    whether the curve came from PyBaMM, a dummy fallback, or another source.
    warnings carries non-fatal loading or validation messages.
    """

    energy_points: tuple[float, ...]
    cost_points: tuple[float, ...]
    source_label: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastRequest:
    """
    Input contract for DAM price forecasting.

    The forecast service reads input_file, identifies the timestamp and price
    columns, builds a forecast for target_date, and optionally writes an
    optimizer-ready CSV. window_days controls how much recent history the
    baseline model uses.
    """

    input_file: Path
    output_file: Path | None = None
    target_date: str | pd.Timestamp | None = None
    window_days: int = 30
    model: str = "seasonal"
    timestamp_col: str | None = None
    price_col: str | None = None
    allow_hourly_upsampling: bool = False
    write_optimizer_input: bool = True


@dataclass(frozen=True)
class ForecastResult:
    """
    Output contract returned by the forecasting workflow.

    forecast contains the forecasted price intervals.
    output_path points to the human-readable forecast file if one was written, and
    optimizer_input_path points to the optimizer-shaped file if requested.
    warnings lists any non-fatal issues encountered during the run, and
    metadata stores model-specific details that should not become fixed fields.
    """

    forecast: pd.DataFrame
    output_path: Path | None
    optimizer_input_path: Path | None
    warnings: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationRequest:
    """
    Input contract for BESS optimization.

    run_mode selects a daily or annual workflow. battery stores the physical
    battery parameters. The remaining fields select market data, degradation
    inputs, scale-up target, and solver options.
    """

    run_mode: OptimizationRunMode = OptimizationRunMode.DAILY
    target_date: str = "2025-11-01"
    year: int = 2025
    price_file: Path | None = None
    degradation_source: str = "pybamm"
    degradation_lut_file: Path | None = None
    temperature_c: float = 25.0
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    scale_capacity_mw: float | None = 50.0
    degradation_cost_multiplier: float = 1.0
    solver_msg: bool = False


@dataclass(frozen=True)
class OptimizationResult:
    """
    Output contract returned by the optimization workflow.

    dispatch_df contains interval-level decisions and economics. summary_dict
    contains run-level KPIs. params_used and files_used record the resolved
    configuration for reproducibility. Annual runs may also include daily
    statistics and benchmark comparison tables.
    """

    status: str
    dispatch_df: pd.DataFrame
    summary_dict: dict
    params_used: dict
    files_used: dict
    warnings: tuple[str, ...] = ()
    daily_stats_df: pd.DataFrame | None = None
    benchmark_comparison_df: pd.DataFrame | None = None


@dataclass(frozen=True)
class ForecastBacktestRequest:
    """
    Input contract for forecast-strategy backtesting.

    The workflow repeatedly forecasts prices, optimizes against those forecast
    prices, then settles the resulting strategy against realized market prices.
    The date range can be explicit, or backtest_days can select a recent fixed
    horizon from the input data.
    """

    input_file: Path
    start_date: str | None = None
    end_date: str | None = None
    backtest_days: int | None = 30
    window_days: int = 30
    model: str = "seasonal"
    degradation_source: str = "pybamm"
    temperature_c: float = 25.0
    installed_capacity_mw: float = 50.0
    timestamp_col: str | None = None
    price_col: str | None = None
    allow_hourly_upsampling: bool = False
    solver_msg: bool = False


@dataclass(frozen=True)
class ForecastBacktestResult:
    """
    Output contract returned by the forecast-strategy backtest workflow.

    interval_schedule stores the interval-level simulated dispatch and settlement results.
    daily_stats and trade_stats summarize performance at
    higher levels, summary contains overall KPIs, and warnings lists non-fatal
    issues encountered during the run.
    """

    interval_schedule: pd.DataFrame
    daily_stats: pd.DataFrame
    trade_stats: pd.DataFrame
    summary: pd.DataFrame
    warnings: tuple[str, ...]
