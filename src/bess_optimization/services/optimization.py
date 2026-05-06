"""Shared optimization services used by CLI wrappers and the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pulp as pl


from bess_optimization.io.degradation import (
    default_lut_for_source as shared_default_lut_for_source,
    list_lut_files as shared_list_lut_files,
    load_degradation_curve as load_shared_degradation_curve,
)
from bess_optimization.io.prices import (
    available_dates as shared_available_dates,
    available_years as shared_available_years,
    list_price_files as shared_list_price_files,
    load_price_series as shared_load_price_series,
    load_price_signal_day as shared_load_price_signal_day,
    load_price_signal_year as shared_load_price_signal_year,
)
from bess_optimization.models import OptimizationRequest as PublicOptimizationRequest
from bess_optimization.models import OptimizationResult
from bess_optimization.optimization.backtest_utils import (
    build_schedule,
    safe_divide,
    summarize,
    validate_schedule,
)
from bess_optimization.optimization.benchmarking import BENCHMARK_LABELS, run_benchmark_model
from bess_optimization.optimization.config import params as BASE_PARAMS
from bess_optimization.optimization.engine import bess_order
from bess_optimization.paths import (
    CLEANED_DATA_DIR,
    DEFAULT_PRICE_SIGNALS_PATH,
    PROJECT_ROOT,
)
from bess_optimization.reporting.scaling import apply_scale, summary_value


BASE_BENCHMARK_MODEL_ID = "degradation_aware_milp"
BASE_BENCHMARK_MODEL_LABEL = "Degradation-aware MILP (PyBaMM/LUT cost)"
BENCHMARK_MODEL_IDS = ("perfect", "naive")
DASHBOARD_NAIVE_ALPHA_FACTOR = 0.05
DASHBOARD_NAIVE_MARGIN = 0.05

class DashboardOptimizerError(RuntimeError):
    """Raised for user-facing dashboard optimizer errors."""


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def list_price_files() -> list[Path]:
    return shared_list_price_files(CLEANED_DATA_DIR)


def list_lut_files() -> list[Path]:
    return shared_list_lut_files(CLEANED_DATA_DIR)


def default_lut_for_source(source: str, lut_files: Iterable[Path] | None = None) -> Path | None:
    return shared_default_lut_for_source(source, lut_files or list_lut_files())


def load_price_series(csv_path: Path) -> pd.Series:
    try:
        return shared_load_price_series(Path(csv_path))
    except Exception as exc:
        raise DashboardOptimizerError(str(exc)) from exc


def available_dates(csv_path: Path) -> list[pd.Timestamp]:
    try:
        return shared_available_dates(Path(csv_path))
    except Exception as exc:
        raise DashboardOptimizerError(str(exc)) from exc


def available_years(csv_path: Path) -> list[int]:
    try:
        return shared_available_years(Path(csv_path))
    except Exception as exc:
        raise DashboardOptimizerError(str(exc)) from exc


def load_price_signal_day(csv_path: Path, target_date: str, dt: float) -> pd.Series:
    try:
        return shared_load_price_signal_day(Path(csv_path), target_date, dt, fill_missing=True)
    except Exception as exc:
        raise DashboardOptimizerError(str(exc)) from exc


def load_price_signal_year(csv_path: Path, year: int, dt: float) -> pd.Series:
    try:
        return shared_load_price_signal_year(Path(csv_path), int(year), dt)
    except Exception as exc:
        raise DashboardOptimizerError(str(exc)) from exc


def validate_parameters(params: dict) -> list[str]:
    errors = []
    p_max = float(params.get("p_max", 0.0))
    e_max = float(params.get("e_max", 0.0))
    eta_ch = float(params.get("eta_ch", 0.0))
    eta_dis = float(params.get("eta_dis", 0.0))
    soc_min = float(params.get("soc_min", 0.0))
    soc_max = float(params.get("soc_max", 0.0))
    soc_init = float(params.get("soc_init", 0.0))

    if p_max <= 0:
        errors.append("Battery power must be positive.")
    if e_max <= 0:
        errors.append("Battery energy capacity must be positive.")
    if not 0 < eta_ch <= 1:
        errors.append("Charge efficiency must be between 0 and 1.")
    if not 0 < eta_dis <= 1:
        errors.append("Discharge efficiency must be between 0 and 1.")
    if not 0 <= soc_min < soc_max <= 1:
        errors.append("SoC minimum must be lower than SoC maximum, both inside 0-100%.")
    if not soc_min < soc_init < soc_max:
        errors.append("Initial SoC must be strictly between the minimum and maximum SoC.")
    if str(params.get("terminal_soc_mode", "equal_initial")).lower() not in {"equal_initial", "free"}:
        errors.append('Terminal SoC mode must be "equal_initial" or "free".')
    return errors


def build_params(params_override: dict | None = None, terminal_soc_mode: str = "equal_initial") -> dict:
    params = dict(BASE_PARAMS)
    params.update(params_override or {})
    params["dt"] = float(params.get("dt", params.get("DT", 0.25)))
    params["terminal_soc_mode"] = terminal_soc_mode
    errors = validate_parameters(params)
    if errors:
        raise DashboardOptimizerError(" ".join(errors))
    return params


def load_degradation_curve(
    source: str,
    params: dict,
    lut_file: Path | None,
    temperature_c: float,
    multiplier: float = 1.0,
) -> tuple[list[float], list[float], str, list[str]]:
    try:
        curve = load_shared_degradation_curve(
            source=source,
            params=params,
            lut_file=Path(lut_file) if lut_file else default_lut_for_source(source),
            temperature_c=temperature_c,
            multiplier=multiplier,
            allow_nearest_temperature=True,
        )
        return curve.energy_points, curve.cost_points, curve.source_label, list(curve.warnings)
    except Exception as exc:
        raise DashboardOptimizerError(str(exc)) from exc


def _apply_scale(schedule: pd.DataFrame, summary: dict, params: dict, scale_capacity_mw: float | None) -> tuple[pd.DataFrame, dict]:
    try:
        return apply_scale(schedule, summary, params, scale_capacity_mw)
    except Exception as exc:
        raise DashboardOptimizerError(str(exc)) from exc


def _summary_value(summary: dict, base_key: str, park_key: str) -> float:
    return summary_value(summary, base_key, park_key)


def _benchmark_note(model_id: str, summary: dict | None = None) -> str:
    if model_id == BASE_BENCHMARK_MODEL_ID:
        return "Selected degradation-aware optimizer."
    if model_id == "perfect":
        return "Perfect foresight with zero degradation cost."
    if model_id == "naive":
        alpha = float((summary or {}).get("naive_alpha_factor", DASHBOARD_NAIVE_ALPHA_FACTOR))
        margin = float((summary or {}).get("naive_margin", DASHBOARD_NAIVE_MARGIN))
        return f"EMA threshold rule, alpha {alpha:g}, margin {margin:g}; zero degradation cost."
    return ""


def _benchmark_comparison_row(
    model_id: str,
    model_label: str,
    summary: dict | None,
    base_summary: dict,
    available: bool = True,
    note: str = "",
) -> dict:
    base_net = _summary_value(base_summary, "net_profit_eur", "park_net_profit_eur")
    if not available or summary is None:
        return {
            "model_id": model_id,
            "available": False,
            "Model": model_label,
            "Status": "Unavailable",
            "Net profit EUR": np.nan,
            "Delta vs degradation-aware MILP EUR": np.nan,
            "Delta vs degradation-aware MILP %": np.nan,
            "Revenue EUR": np.nan,
            "Purchase cost EUR": np.nan,
            "Degradation cost EUR": np.nan,
            "Bought MWh": np.nan,
            "Sold MWh": np.nan,
            "Throughput MWh": np.nan,
            "Equivalent cycles": np.nan,
            "Final SoC %": np.nan,
            "Note": note,
        }

    net = _summary_value(summary, "net_profit_eur", "park_net_profit_eur")
    delta = net - base_net
    return {
        "model_id": model_id,
        "available": True,
        "Model": model_label,
        "Status": str(summary.get("solver_status", "Unknown")),
        "Net profit EUR": net,
        "Delta vs degradation-aware MILP EUR": delta,
        "Delta vs degradation-aware MILP %": safe_divide(delta, abs(base_net)) * 100.0,
        "Revenue EUR": _summary_value(summary, "gross_revenue_eur", "park_gross_revenue_eur"),
        "Purchase cost EUR": _summary_value(summary, "gross_purchase_eur", "park_gross_purchase_eur"),
        "Degradation cost EUR": _summary_value(summary, "degradation_cost_eur", "park_degradation_cost_eur"),
        "Bought MWh": _summary_value(summary, "buy_energy_mwh", "park_buy_energy_mwh"),
        "Sold MWh": _summary_value(summary, "sell_energy_mwh", "park_sell_energy_mwh"),
        "Throughput MWh": _summary_value(summary, "total_throughput_mwh", "park_total_throughput_mwh"),
        "Equivalent cycles": float(summary.get("equivalent_discharge_cycles", 0.0)),
        "Final SoC %": float(summary.get("final_soc_pct", 0.0)) * 100.0,
        "Note": note or _benchmark_note(model_id, summary),
    }


def _annual_summary_from_schedule(
    schedule: pd.DataFrame,
    params: dict,
    year: int,
    days: int,
    optimal_days: int,
    solver_status: str,
    degradation_source: str,
) -> dict:
    gross_revenue = float(schedule["gross_revenue_eur"].sum())
    gross_purchase = float(schedule["gross_purchase_eur"].sum())
    degradation_cost = float(schedule["degradation_cost_eur"].sum())
    net_profit = float(schedule["interval_profit_eur"].sum())
    buy_energy = float(schedule["buy_energy_mwh"].sum())
    sell_energy = float(schedule["sell_energy_mwh"].sum())
    return {
        "solver_status": solver_status,
        "year": int(year),
        "days": int(days),
        "intervals": int(schedule.shape[0]),
        "optimal_days": int(optimal_days),
        "gross_revenue_eur": gross_revenue,
        "gross_purchase_eur": gross_purchase,
        "degradation_cost_eur": degradation_cost,
        "net_profit_eur": net_profit,
        "buy_energy_mwh": buy_energy,
        "sell_energy_mwh": sell_energy,
        "total_throughput_mwh": buy_energy + sell_energy,
        "equivalent_discharge_cycles": safe_divide(sell_energy, params["e_max"]),
        "buy_intervals": int((schedule["operation"] == "buy").sum()),
        "sell_intervals": int((schedule["operation"] == "sell").sum()),
        "idle_intervals": int((schedule["operation"] == "idle").sum()),
        "final_soc_pct": float(schedule["soc_pct"].iloc[-1]),
        "min_soc_pct": float(schedule["soc_pct"].min()),
        "max_soc_pct": float(schedule["soc_pct"].max()),
        "price_min_eur_mwh": float(schedule["price_eur_mwh"].min()),
        "price_max_eur_mwh": float(schedule["price_eur_mwh"].max()),
        "price_mean_eur_mwh": float(schedule["price_eur_mwh"].mean()),
        "battery_power_mw": float(params["p_max"]),
        "battery_capacity_mwh": float(params["e_max"]),
        "battery_duration_h": safe_divide(float(params["e_max"]), float(params["p_max"])),
        "terminal_soc_mode": params.get("terminal_soc_mode", "equal_initial"),
        "degradation_cost_source": degradation_source,
    }


def _run_daily_benchmark_comparison(
    prices: pd.Series,
    params: dict,
    base_summary: dict,
    scale_capacity_mw: float | None,
    solver_msg: bool,
) -> tuple[pd.DataFrame, list[str]]:
    rows = [
        _benchmark_comparison_row(
            BASE_BENCHMARK_MODEL_ID,
            BASE_BENCHMARK_MODEL_LABEL,
            base_summary,
            base_summary,
            note=_benchmark_note(BASE_BENCHMARK_MODEL_ID),
        )
    ]
    warnings = []
    for model_id in BENCHMARK_MODEL_IDS:
        model_label = BENCHMARK_LABELS[model_id]
        try:
            schedule, summary = run_benchmark_model(
                model_id=model_id,
                prices=prices,
                test_params=params,
                solver_msg=solver_msg,
                naive_alpha_factor=DASHBOARD_NAIVE_ALPHA_FACTOR,
                naive_margin=DASHBOARD_NAIVE_MARGIN,
            )
            _, summary = _apply_scale(schedule, summary, params, scale_capacity_mw)
            rows.append(
                _benchmark_comparison_row(
                    model_id,
                    model_label,
                    summary,
                    base_summary,
                    note=_benchmark_note(model_id, summary),
                )
            )
        except Exception as exc:
            message = f"{model_label} benchmark unavailable: {exc}"
            warnings.append(message)
            rows.append(
                _benchmark_comparison_row(
                    model_id,
                    model_label,
                    None,
                    base_summary,
                    available=False,
                    note=str(exc),
                )
            )
    return pd.DataFrame(rows), warnings


def _run_annual_benchmark_comparison(
    prices_year: pd.Series,
    grouped_days: list[tuple[pd.Timestamp, pd.Series]],
    params: dict,
    base_summary: dict,
    scale_capacity_mw: float | None,
    solver_msg: bool,
    year: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows = [
        _benchmark_comparison_row(
            BASE_BENCHMARK_MODEL_ID,
            BASE_BENCHMARK_MODEL_LABEL,
            base_summary,
            base_summary,
            note=_benchmark_note(BASE_BENCHMARK_MODEL_ID),
        )
    ]
    warnings = []
    days = len(grouped_days)
    for model_id in BENCHMARK_MODEL_IDS:
        model_label = BENCHMARK_LABELS[model_id]
        try:
            benchmark_note = ""
            if model_id == "naive":
                schedule, metadata_summary = run_benchmark_model(
                    model_id=model_id,
                    prices=prices_year,
                    test_params=params,
                    solver_msg=solver_msg,
                    naive_alpha_factor=DASHBOARD_NAIVE_ALPHA_FACTOR,
                    naive_margin=DASHBOARD_NAIVE_MARGIN,
                )
                daily_summaries = []
                optimal_days = 0
                solver_status = str(metadata_summary.get("solver_status", "Heuristic"))
            else:
                schedules = []
                daily_summaries = []
                day_failures = []
                for day, day_prices in grouped_days:
                    try:
                        day_schedule, day_summary = run_benchmark_model(
                            model_id=model_id,
                            prices=day_prices,
                            test_params=params,
                            solver_msg=solver_msg,
                            naive_alpha_factor=DASHBOARD_NAIVE_ALPHA_FACTOR,
                            naive_margin=DASHBOARD_NAIVE_MARGIN,
                        )
                        schedules.append(day_schedule)
                        daily_summaries.append(day_summary)
                    except Exception as day_exc:
                        day_failures.append(f"{pd.Timestamp(day).date().isoformat()}: {day_exc}")
                if not schedules:
                    raise RuntimeError("; ".join(day_failures) or "No daily benchmark schedules were solved.")
                schedule = pd.concat(schedules, ignore_index=True)
                metadata_summary = daily_summaries[0]
                optimal_days = int(sum(item.get("solver_status") == "Optimal" for item in daily_summaries))
                solver_status = "Optimal" if optimal_days == days and not day_failures else "Partial"
                if day_failures:
                    benchmark_note = f"{len(day_failures)} daily benchmark runs unavailable; completed days aggregated."
                    warnings.append(f"{model_label}: {benchmark_note}")

            summary = _annual_summary_from_schedule(
                schedule=schedule,
                params=params,
                year=year,
                days=days,
                optimal_days=optimal_days,
                solver_status=solver_status,
                degradation_source=str(metadata_summary.get("degradation_cost_source", "")),
            )
            summary["benchmark_model"] = model_id
            summary["model_label"] = model_label
            if "naive_alpha_factor" in metadata_summary:
                summary["naive_alpha_factor"] = metadata_summary["naive_alpha_factor"]
                summary["naive_margin"] = metadata_summary["naive_margin"]
            _, summary = _apply_scale(schedule, summary, params, scale_capacity_mw)
            rows.append(
                _benchmark_comparison_row(
                    model_id,
                    model_label,
                    summary,
                    base_summary,
                    note=benchmark_note or _benchmark_note(model_id, summary),
                )
            )
        except Exception as exc:
            message = f"{model_label} benchmark unavailable: {exc}"
            warnings.append(message)
            rows.append(
                _benchmark_comparison_row(
                    model_id,
                    model_label,
                    None,
                    base_summary,
                    available=False,
                    note=str(exc),
                )
            )
    return pd.DataFrame(rows), warnings


def _optimize_price_series(
    prices: pd.Series,
    params: dict,
    energy_points: list[float],
    cost_points: list[float],
    degradation_label: str,
    price_file: Path,
    lut_file: Path | None,
    scale_capacity_mw: float | None,
    solver_msg: bool,
    warnings: list[str] | None = None,
) -> dict:
    run_warnings = list(warnings or [])
    problem, p_buy, p_sell, soc, degradation_cost = bess_order(
        prices=prices.to_numpy(dtype=float),
        battery_params=params,
        degradation_energy_points=energy_points,
        degradation_cost_points=cost_points,
        solver_msg=solver_msg,
        terminal_soc_mode=params.get("terminal_soc_mode", "equal_initial"),
    )
    status_name = pl.LpStatus[problem.status]
    objective_value = float(pl.value(problem.objective) or 0.0)
    schedule = build_schedule(prices, p_buy, p_sell, soc, degradation_cost, params)
    summary = summarize(schedule, objective_value, status_name, params)
    summary.update(
        {
            "date": prices.index.min().date().isoformat(),
            "price_min_eur_mwh": float(prices.min()),
            "price_max_eur_mwh": float(prices.max()),
            "price_mean_eur_mwh": float(prices.mean()),
            "intervals": int(schedule.shape[0]),
            "battery_power_mw": float(params["p_max"]),
            "battery_capacity_mwh": float(params["e_max"]),
            "battery_duration_h": safe_divide(float(params["e_max"]), float(params["p_max"])),
            "terminal_soc_mode": params.get("terminal_soc_mode", "equal_initial"),
            "degradation_cost_source": degradation_label,
        }
    )

    try:
        validate_schedule(schedule, summary, params)
        summary["validation_passed"] = True
    except RuntimeError as exc:
        summary["validation_passed"] = False
        run_warnings.append(str(exc))

    schedule, summary = _apply_scale(schedule, summary, params, scale_capacity_mw)
    schedule["mode"] = schedule["operation"].str.upper()
    schedule["net_power_mw"] = schedule["p_sell_mw"] - schedule["p_buy_mw"]

    return {
        "status": status_name,
        "dispatch_df": schedule,
        "summary_dict": summary,
        "params_used": dict(params),
        "files_used": {
            "price_file": str(price_file),
            "degradation_lut_file": str(lut_file) if lut_file else None,
            "degradation_source": degradation_label,
        },
        "warnings": run_warnings,
    }


def run_daily_optimization(
    target_date=None,
    price_file=None,
    degradation_lut_file=None,
    params_override=None,
    degradation_source="pybamm",
    scale_capacity_mw=None,
    temperature_c=25.0,
    terminal_soc_mode="equal_initial",
    degradation_cost_multiplier=1.0,
    solver_msg=False,
) -> OptimizationResult:
    if isinstance(target_date, PublicOptimizationRequest):
        request = target_date
        target_date = request.target_date
        price_file = request.price_file
        degradation_lut_file = request.degradation_lut_file
        params_override = request.params_override
        degradation_source = request.degradation_source
        scale_capacity_mw = request.scale_capacity_mw
        temperature_c = request.temperature_c
        terminal_soc_mode = request.terminal_soc_mode
        degradation_cost_multiplier = request.degradation_cost_multiplier
        solver_msg = request.solver_msg

    price_file = Path(price_file or DEFAULT_PRICE_SIGNALS_PATH)
    params_override = params_override or {}
    params = build_params(params_override, terminal_soc_mode=terminal_soc_mode)
    energy_points, cost_points, degradation_label, warnings = load_degradation_curve(
        degradation_source,
        params,
        degradation_lut_file,
        temperature_c,
        multiplier=degradation_cost_multiplier,
    )
    prices = load_price_signal_day(price_file, str(target_date), params["dt"])
    result = _optimize_price_series(
        prices=prices,
        params=params,
        energy_points=energy_points,
        cost_points=cost_points,
        degradation_label=degradation_label,
        price_file=price_file,
        lut_file=Path(degradation_lut_file) if degradation_lut_file else None,
        scale_capacity_mw=scale_capacity_mw,
        solver_msg=solver_msg,
        warnings=warnings,
    )
    result["summary_dict"]["price_start"] = str(prices.index.min())
    result["summary_dict"]["price_end"] = str(prices.index.max())
    comparison, benchmark_warnings = _run_daily_benchmark_comparison(
        prices=prices,
        params=params,
        base_summary=result["summary_dict"],
        scale_capacity_mw=scale_capacity_mw,
        solver_msg=solver_msg,
    )
    result["benchmark_comparison_df"] = comparison
    result["warnings"].extend(benchmark_warnings)
    return OptimizationResult(
        status=result["status"],
        dispatch_df=result["dispatch_df"],
        summary_dict=result["summary_dict"],
        params_used=result["params_used"],
        files_used=result["files_used"],
        warnings=result["warnings"],
        benchmark_comparison_df=result.get("benchmark_comparison_df"),
    )


def run_annual_optimization(
    year=None,
    price_file=None,
    degradation_lut_file=None,
    params_override=None,
    degradation_source="pybamm",
    scale_capacity_mw=None,
    temperature_c=25.0,
    terminal_soc_mode="equal_initial",
    degradation_cost_multiplier=1.0,
    solver_msg=False,
) -> OptimizationResult:
    if isinstance(year, PublicOptimizationRequest):
        request = year
        year = request.year
        price_file = request.price_file
        degradation_lut_file = request.degradation_lut_file
        params_override = request.params_override
        degradation_source = request.degradation_source
        scale_capacity_mw = request.scale_capacity_mw
        temperature_c = request.temperature_c
        terminal_soc_mode = request.terminal_soc_mode
        degradation_cost_multiplier = request.degradation_cost_multiplier
        solver_msg = request.solver_msg

    price_file = Path(price_file or DEFAULT_PRICE_SIGNALS_PATH)
    params_override = params_override or {}
    params = build_params(params_override, terminal_soc_mode=terminal_soc_mode)
    energy_points, cost_points, degradation_label, warnings = load_degradation_curve(
        degradation_source,
        params,
        degradation_lut_file,
        temperature_c,
        multiplier=degradation_cost_multiplier,
    )
    prices_year = load_price_signal_year(price_file, int(year), params["dt"])

    schedules = []
    daily_summaries = []
    run_warnings = list(warnings)
    grouped_days = list(prices_year.groupby(prices_year.index.normalize()))
    for _, day_prices in grouped_days:
        result = _optimize_price_series(
            prices=day_prices,
            params=params,
            energy_points=energy_points,
            cost_points=cost_points,
            degradation_label=degradation_label,
                price_file=price_file,
            lut_file=Path(degradation_lut_file) if degradation_lut_file else None,
            scale_capacity_mw=scale_capacity_mw,
            solver_msg=solver_msg,
        )
        schedules.append(result["dispatch_df"])
        daily_summaries.append(result["summary_dict"])
        run_warnings.extend(result["warnings"])

    annual_schedule = pd.concat(schedules, ignore_index=True)
    daily_stats = pd.DataFrame(daily_summaries)
    gross_revenue = float(annual_schedule["gross_revenue_eur"].sum())
    gross_purchase = float(annual_schedule["gross_purchase_eur"].sum())
    degradation_cost = float(annual_schedule["degradation_cost_eur"].sum())
    net_profit = float(annual_schedule["interval_profit_eur"].sum())
    buy_energy = float(annual_schedule["buy_energy_mwh"].sum())
    sell_energy = float(annual_schedule["sell_energy_mwh"].sum())
    status = "Optimal" if (daily_stats["solver_status"] == "Optimal").all() else "Partial"

    summary = {
        "solver_status": status,
        "year": int(year),
        "days": int(daily_stats.shape[0]),
        "intervals": int(annual_schedule.shape[0]),
        "optimal_days": int((daily_stats["solver_status"] == "Optimal").sum()),
        "gross_revenue_eur": gross_revenue,
        "gross_purchase_eur": gross_purchase,
        "degradation_cost_eur": degradation_cost,
        "net_profit_eur": net_profit,
        "buy_energy_mwh": buy_energy,
        "sell_energy_mwh": sell_energy,
        "total_throughput_mwh": buy_energy + sell_energy,
        "equivalent_discharge_cycles": safe_divide(sell_energy, params["e_max"]),
        "buy_intervals": int((annual_schedule["operation"] == "buy").sum()),
        "sell_intervals": int((annual_schedule["operation"] == "sell").sum()),
        "idle_intervals": int((annual_schedule["operation"] == "idle").sum()),
        "final_soc_pct": float(annual_schedule["soc_pct"].iloc[-1]),
        "min_soc_pct": float(annual_schedule["soc_pct"].min()),
        "max_soc_pct": float(annual_schedule["soc_pct"].max()),
        "price_min_eur_mwh": float(annual_schedule["price_eur_mwh"].min()),
        "price_max_eur_mwh": float(annual_schedule["price_eur_mwh"].max()),
        "price_mean_eur_mwh": float(annual_schedule["price_eur_mwh"].mean()),
        "battery_power_mw": float(params["p_max"]),
        "battery_capacity_mwh": float(params["e_max"]),
        "battery_duration_h": safe_divide(float(params["e_max"]), float(params["p_max"])),
        "terminal_soc_mode": params.get("terminal_soc_mode", "equal_initial"),
        "degradation_cost_source": degradation_label,
        "validation_passed": not run_warnings,
    }
    annual_schedule, summary = _apply_scale(annual_schedule, summary, params, scale_capacity_mw)
    comparison, benchmark_warnings = _run_annual_benchmark_comparison(
        prices_year=prices_year,
        grouped_days=grouped_days,
        params=params,
        base_summary=summary,
        scale_capacity_mw=scale_capacity_mw,
        solver_msg=solver_msg,
        year=int(year),
    )
    run_warnings.extend(benchmark_warnings)

    return OptimizationResult(
        status=status,
        dispatch_df=annual_schedule,
        summary_dict=summary,
        daily_stats_df=daily_stats,
        benchmark_comparison_df=comparison,
        params_used=dict(params),
        files_used={
            "price_file": str(price_file),
            "degradation_lut_file": str(degradation_lut_file) if degradation_lut_file else None,
            "degradation_source": degradation_label,
        },
        warnings=run_warnings,
    )
