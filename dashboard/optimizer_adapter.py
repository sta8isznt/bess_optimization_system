"""Adapter layer between Streamlit and the existing BESS optimizer."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pulp as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.backtest_utils import (  # noqa: E402
    build_dummy_cost_curve,
    build_schedule,
    safe_divide,
    summarize,
    validate_schedule,
)
from optimization.backtesting_loaders import (  # noqa: E402
    DEFAULT_DEGRADATION_LUT_PATH,
    DEFAULT_PRICE_SIGNALS_PATH,
    DEFAULT_PYBAMM_ONLY_LUT_PATH,
)
from optimization.benchmarking import BENCHMARK_LABELS, run_benchmark_model  # noqa: E402
from optimization.config import params as BASE_PARAMS  # noqa: E402
from optimization.engine import bess_order  # noqa: E402


CLEANED_DATA_DIR = PROJECT_ROOT / "optimization" / "data" / "cleaned_data"
BASE_BENCHMARK_MODEL_ID = "degradation_aware_milp"
BASE_BENCHMARK_MODEL_LABEL = "Degradation-aware MILP (PyBaMM/LUT cost)"
BENCHMARK_MODEL_IDS = ("perfect", "naive")
DASHBOARD_NAIVE_ALPHA_FACTOR = 0.05
DASHBOARD_NAIVE_MARGIN = 0.05

TIME_COLUMNS = ("DELIVERY_MTU", "timestamp", "datetime", "time")
PRICE_COLUMNS = (
    "DAM_Price_EUR_MWh",
    "price_eur_mwh",
    "price_eur_per_MWh",
    "price_eur_per_mwh",
    "MCP",
)
LUT_ENERGY_COLUMNS = ("energy", "energy_MWh", "discharge_energy_mwh")
LUT_TEMP_COLUMNS = ("temperature_c", "temperature_C", "temp_c", "T_cell_C")
LUT_COST_COLUMNS = (
    "deg_cost_eur_per_MWh_throughput",
    "deg_cost_final",
    "deg_cost_smoothed",
    "deg_cost_surface_full",
    "hybrid_cost_final_eur_per_MWh",
    "aggressive_cost_final_surface_eur_per_MWh",
    "degradation_cost_eur",
)


class DashboardOptimizerError(RuntimeError):
    """Raised for user-facing dashboard optimizer errors."""


@dataclass(frozen=True)
class OptimizationRequest:
    run_mode: str
    target_date: str
    year: int
    price_file: Path
    degradation_source: str
    degradation_lut_file: Path | None
    temperature_c: float
    params_override: dict
    scale_capacity_mw: float | None
    terminal_soc_mode: str = "equal_initial"
    degradation_cost_multiplier: float = 1.0
    solver_msg: bool = False


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def list_price_files() -> list[Path]:
    files = []
    if DEFAULT_PRICE_SIGNALS_PATH.exists():
        files.append(DEFAULT_PRICE_SIGNALS_PATH)
    files.extend(
        p for p in sorted(CLEANED_DATA_DIR.glob("*.csv"))
        if p not in files and ("price" in p.name.lower() or "signal" in p.name.lower())
    )
    return files


def list_lut_files() -> list[Path]:
    files = []
    for candidate in (DEFAULT_PYBAMM_ONLY_LUT_PATH, DEFAULT_DEGRADATION_LUT_PATH):
        if candidate.exists() and candidate not in files:
            files.append(candidate)
    files.extend(
        p for p in sorted(CLEANED_DATA_DIR.glob("*.csv"))
        if p not in files and ("lut" in p.name.lower() or "degradation" in p.name.lower())
    )
    return files


def default_lut_for_source(source: str, lut_files: Iterable[Path] | None = None) -> Path | None:
    files = list(lut_files or list_lut_files())
    source = source.lower()
    if source == "pybamm_only":
        for path in files:
            if "pybamm" in path.name.lower():
                return path
    if source == "lut":
        for path in files:
            if "pybamm" not in path.name.lower():
                return path
    return files[0] if files else None


def _find_column(df: pd.DataFrame, candidates: Iterable[str], purpose: str) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    raise DashboardOptimizerError(
        f"Could not infer {purpose} column. Available columns: {list(df.columns)}"
    )


def _infer_time_column(df: pd.DataFrame) -> str:
    try:
        return _find_column(df, TIME_COLUMNS, "timestamp")
    except DashboardOptimizerError:
        pass

    for column in df.columns:
        lower = str(column).lower()
        if any(token in lower for token in ("time", "date", "mtu")):
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().mean() > 0.8:
                return column
    raise DashboardOptimizerError(
        f"Could not infer timestamp column. Available columns: {list(df.columns)}"
    )


def _infer_price_column(df: pd.DataFrame, time_column: str) -> str:
    try:
        return _find_column(df, PRICE_COLUMNS, "price")
    except DashboardOptimizerError:
        pass

    numeric_candidates = []
    for column in df.columns:
        if column == time_column:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().mean() > 0.8:
            numeric_candidates.append(column)
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise DashboardOptimizerError(
        f"Could not infer price column. Available columns: {list(df.columns)}"
    )


def load_price_series(csv_path: Path) -> pd.Series:
    path = Path(csv_path)
    if not path.exists():
        raise DashboardOptimizerError(f"Price input file does not exist: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise DashboardOptimizerError(f"Price input file is empty: {path.name}")

    time_col = _infer_time_column(df)
    price_col = _infer_price_column(df, time_col)
    work = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[time_col], errors="coerce"),
            "price": pd.to_numeric(df[price_col], errors="coerce"),
        }
    ).dropna()
    if work.empty:
        raise DashboardOptimizerError(f"No valid timestamp/price rows found in {path.name}")

    prices = work.groupby("timestamp")["price"].mean().sort_index()
    prices.name = "price_eur_mwh"
    return prices


def available_dates(csv_path: Path) -> list[pd.Timestamp]:
    prices = load_price_series(csv_path)
    return [pd.Timestamp(d) for d in sorted(pd.Series(prices.index.date).unique())]


def available_years(csv_path: Path) -> list[int]:
    prices = load_price_series(csv_path)
    return sorted(int(year) for year in pd.Series(prices.index.year).unique())


def load_price_signal_day(csv_path: Path, target_date: str, dt: float) -> pd.Series:
    prices = load_price_series(csv_path)
    target_day = pd.Timestamp(target_date).floor("D")
    same_day = prices[prices.index.normalize() == target_day]
    if same_day.empty:
        raise DashboardOptimizerError(f"No price data exists for {target_day.date().isoformat()}.")

    step_minutes = int(round(float(dt) * 60))
    periods = int(round(24 * 60 / step_minutes))
    target_index = pd.date_range(target_day, periods=periods, freq=f"{step_minutes}min")
    day_prices = (
        same_day.reindex(same_day.index.union(target_index))
        .sort_index()
        .ffill()
        .bfill()
        .reindex(target_index)
    )
    if day_prices.isna().any():
        raise DashboardOptimizerError(f"Could not build a complete daily price vector for {target_day.date()}.")
    day_prices.name = "price_eur_mwh"
    return day_prices


def load_price_signal_year(csv_path: Path, year: int, dt: float) -> pd.Series:
    prices = load_price_series(csv_path)
    year = int(year)
    step_minutes = int(round(float(dt) * 60))
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year + 1, month=1, day=1)
    target_index = pd.date_range(
        start=year_start,
        end=year_end - pd.Timedelta(minutes=step_minutes),
        freq=f"{step_minutes}min",
    )
    year_prices = prices.reindex(target_index)
    if year_prices.isna().any():
        missing = int(year_prices.isna().sum())
        raise DashboardOptimizerError(f"{missing} 15-minute prices are missing for {year}.")
    year_prices.name = "price_eur_mwh"
    return year_prices


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


def _read_lut_curve(
    csv_path: Path,
    temperature_c: float,
    max_interval_energy_mwh: float,
    multiplier: float,
) -> tuple[list[float], list[float], str, list[str]]:
    path = Path(csv_path)
    if not path.exists():
        raise DashboardOptimizerError(f"Degradation LUT file does not exist: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise DashboardOptimizerError(f"Degradation LUT is empty: {path.name}")

    energy_col = _find_column(df, LUT_ENERGY_COLUMNS, "LUT energy")
    cost_col = _find_column(df, LUT_COST_COLUMNS, "LUT degradation cost")
    warnings = []

    filtered = df.copy()
    temp_label = "all temperatures"
    try:
        temp_col = _find_column(df, LUT_TEMP_COLUMNS, "LUT temperature")
        filtered[temp_col] = pd.to_numeric(filtered[temp_col], errors="coerce")
        available = filtered[temp_col].dropna().sort_values().unique()
        if len(available) == 0:
            raise DashboardOptimizerError("No valid temperature rows found in LUT.")
        nearest = float(available[np.argmin(np.abs(available - float(temperature_c)))])
        if abs(nearest - float(temperature_c)) > 1e-9:
            warnings.append(
                f"LUT has no exact {temperature_c:g} C row; using nearest {nearest:g} C."
            )
        filtered = filtered[np.isclose(filtered[temp_col], nearest)].copy()
        temp_label = f"{nearest:g} C"
    except DashboardOptimizerError:
        warnings.append("LUT has no temperature column; using all rows.")

    filtered[energy_col] = pd.to_numeric(filtered[energy_col], errors="coerce")
    filtered[cost_col] = pd.to_numeric(filtered[cost_col], errors="coerce")
    filtered = filtered.dropna(subset=[energy_col, cost_col])
    filtered = filtered[filtered[energy_col] > 0].sort_values(energy_col)
    if filtered.empty:
        raise DashboardOptimizerError(f"No valid energy/cost rows found in {path.name}.")

    energy_points = filtered[energy_col].astype(float).tolist()
    cost_values = filtered[cost_col].astype(float)
    if "cost_eur_per" in cost_col.lower() or cost_col.lower().startswith("deg_cost"):
        cost_points = (filtered[energy_col].astype(float) * cost_values).tolist()
    else:
        cost_points = cost_values.tolist()

    multiplier = float(multiplier)
    if multiplier < 0:
        raise DashboardOptimizerError("Degradation cost multiplier cannot be negative.")
    cost_points = [float(value) * multiplier for value in cost_points]

    if max_interval_energy_mwh > max(energy_points) + 1e-9:
        warnings.append(
            "Selected power can discharge more energy per interval than the LUT covers. "
            f"The MILP will be effectively capped at {max(energy_points):.3f} MWh per interval."
        )

    energy_points.insert(0, 0.0)
    cost_points.insert(0, 0.0)
    label = f"{path.name} at {temp_label}"
    if abs(multiplier - 1.0) > 1e-12:
        label += f" x {multiplier:g}"
    return energy_points, cost_points, label, warnings


def load_degradation_curve(
    source: str,
    params: dict,
    lut_file: Path | None,
    temperature_c: float,
    multiplier: float = 1.0,
) -> tuple[list[float], list[float], str, list[str]]:
    source = source.strip().lower()
    max_interval_energy = float(params["p_max"]) * float(params["dt"])

    if source == "dummy":
        energy, cost = build_dummy_cost_curve(params)
        cost = [float(value) * float(multiplier) for value in cost]
        return energy, cost, "synthetic dummy degradation curve", []

    if source == "zero":
        energy = np.linspace(0.0, max_interval_energy, 5).tolist()
        return energy, [0.0 for _ in energy], "zero degradation cost - comparison only", []

    if source not in {"lut", "pybamm_only"}:
        raise DashboardOptimizerError('Degradation source must be "lut", "pybamm_only", "dummy", or "zero".')

    selected_lut = Path(lut_file) if lut_file else default_lut_for_source(source)
    if selected_lut is None:
        raise DashboardOptimizerError(f"No LUT file is available for degradation source: {source}")
    return _read_lut_curve(selected_lut, temperature_c, max_interval_energy, multiplier)


def _apply_scale(schedule: pd.DataFrame, summary: dict, params: dict, scale_capacity_mw: float | None) -> tuple[pd.DataFrame, dict]:
    p_max = float(params["p_max"])
    installed_mw = float(scale_capacity_mw) if scale_capacity_mw else p_max
    if installed_mw <= 0:
        raise DashboardOptimizerError("Installed capacity must be positive.")
    scale_factor = installed_mw / p_max

    scaled = schedule.copy()
    columns_to_scale = {
        "p_buy_mw": "park_p_buy_mw",
        "p_sell_mw": "park_p_sell_mw",
        "net_export_mw": "park_net_export_mw",
        "buy_energy_mwh": "park_buy_energy_mwh",
        "sell_energy_mwh": "park_sell_energy_mwh",
        "soc_mwh": "park_soc_mwh",
        "gross_revenue_eur": "park_gross_revenue_eur",
        "gross_purchase_eur": "park_gross_purchase_eur",
        "degradation_cost_eur": "park_degradation_cost_eur",
        "interval_profit_eur": "park_interval_profit_eur",
    }
    for source_col, target_col in columns_to_scale.items():
        scaled[target_col] = scaled[source_col] * scale_factor

    out = dict(summary)
    out["installed_capacity_mw"] = installed_mw
    out["installed_energy_capacity_mwh"] = float(params["e_max"]) * scale_factor
    out["scale_factor"] = scale_factor
    out["park_net_profit_eur"] = float(out["net_profit_eur"] * scale_factor)
    out["park_gross_revenue_eur"] = float(out["gross_revenue_eur"] * scale_factor)
    out["park_gross_purchase_eur"] = float(out["gross_purchase_eur"] * scale_factor)
    out["park_degradation_cost_eur"] = float(out["degradation_cost_eur"] * scale_factor)
    out["park_buy_energy_mwh"] = float(out["buy_energy_mwh"] * scale_factor)
    out["park_sell_energy_mwh"] = float(out["sell_energy_mwh"] * scale_factor)
    out["park_total_throughput_mwh"] = float(out["total_throughput_mwh"] * scale_factor)
    return scaled, out


def _summary_value(summary: dict, base_key: str, park_key: str) -> float:
    if float(summary.get("scale_factor", 1.0)) != 1.0 and park_key in summary:
        return float(summary[park_key])
    return float(summary.get(base_key, 0.0))


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
    target_date,
    price_file,
    degradation_lut_file,
    params_override,
    degradation_source,
    scale_capacity_mw=None,
    temperature_c=25.0,
    terminal_soc_mode="equal_initial",
    degradation_cost_multiplier=1.0,
    solver_msg=False,
) -> dict:
    params = build_params(params_override, terminal_soc_mode=terminal_soc_mode)
    energy_points, cost_points, degradation_label, warnings = load_degradation_curve(
        degradation_source,
        params,
        degradation_lut_file,
        temperature_c,
        multiplier=degradation_cost_multiplier,
    )
    prices = load_price_signal_day(Path(price_file), str(target_date), params["dt"])
    result = _optimize_price_series(
        prices=prices,
        params=params,
        energy_points=energy_points,
        cost_points=cost_points,
        degradation_label=degradation_label,
        price_file=Path(price_file),
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
    return result


def run_annual_optimization(
    year,
    price_file,
    degradation_lut_file,
    params_override,
    degradation_source,
    scale_capacity_mw=None,
    temperature_c=25.0,
    terminal_soc_mode="equal_initial",
    degradation_cost_multiplier=1.0,
    solver_msg=False,
) -> dict:
    params = build_params(params_override, terminal_soc_mode=terminal_soc_mode)
    energy_points, cost_points, degradation_label, warnings = load_degradation_curve(
        degradation_source,
        params,
        degradation_lut_file,
        temperature_c,
        multiplier=degradation_cost_multiplier,
    )
    prices_year = load_price_signal_year(Path(price_file), int(year), params["dt"])

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
            price_file=Path(price_file),
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

    return {
        "status": status,
        "dispatch_df": annual_schedule,
        "summary_dict": summary,
        "daily_stats_df": daily_stats,
        "benchmark_comparison_df": comparison,
        "params_used": dict(params),
        "files_used": {
            "price_file": str(price_file),
            "degradation_lut_file": str(degradation_lut_file) if degradation_lut_file else None,
            "degradation_source": degradation_label,
        },
        "warnings": run_warnings,
    }
