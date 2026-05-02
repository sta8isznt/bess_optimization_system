"""Shared request/result models for BESS workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BatteryConfig:
    p_max: float = 1.0
    e_max: float = 2.0
    eta_ch: float = 0.92
    eta_dis: float = 0.92
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_init: float = 0.50
    dt: float = 0.25
    terminal_soc_mode: str = "equal_initial"

    def as_dict(self) -> dict:
        return {
            "p_max": self.p_max,
            "e_max": self.e_max,
            "eta_ch": self.eta_ch,
            "eta_dis": self.eta_dis,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "soc_init": self.soc_init,
            "dt": self.dt,
            "terminal_soc_mode": self.terminal_soc_mode,
        }


@dataclass(frozen=True)
class DegradationCurve:
    energy_points: list[float]
    cost_points: list[float]
    source_label: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ForecastRequest:
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
    forecast: pd.DataFrame
    output_path: Path | None
    optimizer_input_path: Path | None
    warnings: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationRequest:
    run_mode: str = "daily"
    target_date: str = "2025-11-01"
    year: int = 2025
    price_file: Path | None = None
    degradation_source: str = "pybamm_only"
    degradation_lut_file: Path | None = None
    temperature_c: float = 25.0
    params_override: dict = field(default_factory=dict)
    scale_capacity_mw: float | None = 50.0
    terminal_soc_mode: str = "equal_initial"
    degradation_cost_multiplier: float = 1.0
    solver_msg: bool = False


@dataclass(frozen=True)
class OptimizationResult:
    status: str
    dispatch_df: pd.DataFrame
    summary_dict: dict
    params_used: dict
    files_used: dict
    warnings: list[str] = field(default_factory=list)
    daily_stats_df: pd.DataFrame | None = None
    benchmark_comparison_df: pd.DataFrame | None = None

    def to_legacy_dict(self) -> dict:
        out = {
            "status": self.status,
            "dispatch_df": self.dispatch_df,
            "summary_dict": self.summary_dict,
            "params_used": self.params_used,
            "files_used": self.files_used,
            "warnings": self.warnings,
        }
        if self.daily_stats_df is not None:
            out["daily_stats_df"] = self.daily_stats_df
        if self.benchmark_comparison_df is not None:
            out["benchmark_comparison_df"] = self.benchmark_comparison_df
        return out


@dataclass(frozen=True)
class ForecastBacktestRequest:
    input_file: Path
    start_date: str | None = None
    end_date: str | None = None
    backtest_days: int | None = 30
    window_days: int = 30
    model: str = "seasonal"
    degradation_source: str = "pybamm_only"
    temperature_c: float = 25.0
    installed_capacity_mw: float = 50.0
    timestamp_col: str | None = None
    price_col: str | None = None
    allow_hourly_upsampling: bool = False
    solver_msg: bool = False


@dataclass(frozen=True)
class ForecastBacktestResult:
    interval_schedule: pd.DataFrame
    daily_stats: pd.DataFrame
    trade_stats: pd.DataFrame
    summary: pd.DataFrame
    warnings: tuple[str, ...]
