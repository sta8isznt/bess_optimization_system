"""Backtest forecast-driven BESS dispatch against actual DAM prices.

This module intentionally wraps the existing optimization engine instead of
changing it. The MILP still sees one price series. The wrapper makes that price
series a forecast, then settles the resulting schedule on actual prices.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pulp as pl

from bess_optimization.io.degradation import load_degradation_curve as load_shared_degradation_curve
from bess_optimization.models import ForecastBacktestResult as PublicForecastBacktestResult
from bess_optimization.optimization.backtest_utils import (
    build_schedule,
    safe_divide,
    summarize,
    validate_schedule,
)
from bess_optimization.optimization.config import params as BASE_PARAMS
from bess_optimization.optimization.engine import bess_order
from bess_optimization.forecasting.dam_15min_forecast import (
    EXPECTED_PERIODS_PER_DAY,
    ForecastingError,
    forecast_next_day,
    load_price_history,
    utc_created_at,
)
from bess_optimization.paths import (
    DEFAULT_DEGRADATION_LUT_PATH,
    DEFAULT_PRICE_SIGNALS_PATH,
    DEFAULT_PYBAMM_ONLY_LUT_PATH,
    FORECAST_BACKTEST_OUTPUT_DIR,
)
from bess_optimization.settlement.cashflows import settle_schedule_on_actual_prices


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = FORECAST_BACKTEST_OUTPUT_DIR
DEFAULT_BACKTEST_DAYS = 30
DEFAULT_WINDOW_DAYS = 30
DEFAULT_INSTALLED_CAPACITY_MW = 50.0
DEFAULT_DEGRADATION_SOURCE = "pybamm_only"
DEFAULT_LUT_TEMPERATURE_C = 25.0
PROFIT_TOLERANCE_EUR = 1e-7


@dataclass(frozen=True)
class ForecastStrategyBacktestResult(PublicForecastBacktestResult):
    """Container for forecast strategy backtest outputs."""

    interval_schedule: pd.DataFrame
    daily_stats: pd.DataFrame
    trade_stats: pd.DataFrame
    summary: pd.DataFrame
    warnings: tuple[str, ...]


def build_test_params() -> dict:
    """Return optimizer battery parameters with a normalized ``dt`` key."""

    test_params = dict(BASE_PARAMS)
    test_params["dt"] = test_params.get("dt", test_params.get("DT", 0.25))
    return test_params


def load_degradation_curve(
    source: str,
    test_params: dict,
    temperature_c: float = DEFAULT_LUT_TEMPERATURE_C,
    lut_csv: Path = DEFAULT_DEGRADATION_LUT_PATH,
    pybamm_only_lut_csv: Path = DEFAULT_PYBAMM_ONLY_LUT_PATH,
) -> tuple[list[float], list[float], str]:
    """Load the same degradation-curve inputs expected by the optimizer."""

    source = str(source).strip().lower()
    lut_file = pybamm_only_lut_csv if source == "pybamm_only" else lut_csv
    curve = load_shared_degradation_curve(
        source=source,
        params=test_params,
        lut_file=lut_file,
        temperature_c=temperature_c,
    )
    return curve.energy_points, curve.cost_points, curve.source_label


def _with_date(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["date"] = out["timestamp"].dt.normalize()
    return out


def complete_15min_days(history: pd.DataFrame) -> list[pd.Timestamp]:
    """Return days that have exactly 96 unique 15-minute timestamps."""

    featured = _with_date(history)
    counts = featured.groupby("date")["timestamp"].nunique()
    complete = counts[counts == EXPECTED_PERIODS_PER_DAY]
    return [pd.Timestamp(day) for day in complete.index.sort_values()]


def select_backtest_days(
    history: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
    backtest_days: int | None = DEFAULT_BACKTEST_DAYS,
) -> list[pd.Timestamp]:
    """Select complete historical target days for forecast-settlement testing."""

    days = complete_15min_days(history)
    if not days:
        raise ValueError("No complete 96-slot days are available for backtesting.")

    min_timestamp = pd.to_datetime(history["timestamp"]).min()
    days = [day for day in days if day > min_timestamp.floor("D")]

    if start_date:
        start = pd.Timestamp(start_date).floor("D")
        days = [day for day in days if day >= start]
    if end_date:
        end = pd.Timestamp(end_date).floor("D")
        days = [day for day in days if day <= end]
    if backtest_days is not None and int(backtest_days) > 0:
        days = days[-int(backtest_days) :]

    if not days:
        raise ValueError("No complete target days remain after date filtering.")
    return days


def extract_actual_day_prices(
    history: pd.DataFrame,
    target_day: pd.Timestamp,
    dt: float,
) -> pd.Series:
    """Load exact actual prices for one target day without forward filling."""

    target_day = pd.Timestamp(target_day).floor("D")
    step_minutes = int(round(float(dt) * 60))
    target_index = pd.date_range(
        target_day,
        periods=EXPECTED_PERIODS_PER_DAY,
        freq=f"{step_minutes}min",
    )
    prices = (
        history.assign(timestamp=pd.to_datetime(history["timestamp"]))
        .groupby("timestamp")["price_eur_mwh"]
        .mean()
        .sort_index()
    )
    actual = prices.reindex(target_index)
    if actual.isna().any():
        missing = int(actual.isna().sum())
        raise ValueError(
            f"{missing} actual price slots are missing for {target_day.date()}."
        )
    actual.name = "actual_price_eur_mwh"
    return actual


def forecast_series_for_day(
    history: pd.DataFrame,
    target_day: pd.Timestamp,
    window_days: int = DEFAULT_WINDOW_DAYS,
    model: str = "seasonal",
) -> tuple[pd.Series, pd.DataFrame, list[str]]:
    """Forecast one day and return an optimizer-ready price series."""

    forecast, warnings = forecast_next_day(
        history=history,
        target_date=target_day,
        window_days=window_days,
        model=model,
        created_at_utc=utc_created_at(),
    )
    prices = pd.Series(
        forecast["forecast_price_eur_mwh"].to_numpy(dtype=float),
        index=pd.to_datetime(forecast["timestamp"]),
        name="price_eur_mwh",
    )
    return prices, forecast, warnings


def optimize_dispatch_on_prices(
    prices: pd.Series,
    test_params: dict,
    energy_points: list[float],
    cost_points: list[float],
    solver_msg: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the existing MILP against the supplied price signal."""

    problem, p_buy, p_sell, soc, degradation_cost = bess_order(
        prices=prices.to_numpy(dtype=float),
        battery_params=test_params,
        degradation_energy_points=energy_points,
        degradation_cost_points=cost_points,
        solver_msg=solver_msg,
    )
    status_name = pl.LpStatus[problem.status]
    objective_value = float(pl.value(problem.objective) or 0.0)
    schedule = build_schedule(
        prices=prices,
        p_buy=p_buy,
        p_sell=p_sell,
        soc=soc,
        degradation_cost=degradation_cost,
        test_params=test_params,
    )
    summary = summarize(schedule, objective_value, status_name, test_params)
    summary["date"] = prices.index.min().date().isoformat()
    summary["price_min_eur_mwh"] = float(prices.min())
    summary["price_max_eur_mwh"] = float(prices.max())
    summary["price_mean_eur_mwh"] = float(prices.mean())
    validate_schedule(schedule, summary, test_params)
    return schedule, summary


def _forecast_error_metrics(settled: pd.DataFrame) -> dict:
    actual = settled["actual_price_eur_mwh"].to_numpy(dtype=float)
    forecast = settled["forecast_price_eur_mwh"].to_numpy(dtype=float)
    error = forecast - actual
    abs_actual = np.abs(actual)
    mape_mask = abs_actual > 1e-9
    smape_denominator = abs_actual + np.abs(forecast)
    smape_mask = smape_denominator > 1e-9
    return {
        "forecast_mae_eur_mwh": float(np.mean(np.abs(error))),
        "forecast_rmse_eur_mwh": float(np.sqrt(np.mean(error**2))),
        "forecast_bias_eur_mwh": float(np.mean(error)),
        "forecast_mape_percent": (
            float(np.mean(np.abs(error[mape_mask]) / abs_actual[mape_mask]) * 100.0)
            if mape_mask.any()
            else 0.0
        ),
        "forecast_smape_percent": (
            float(
                np.mean(
                    2.0
                    * np.abs(error[smape_mask])
                    / smape_denominator[smape_mask]
                )
                * 100.0
            )
            if smape_mask.any()
            else 0.0
        ),
    }


def _operation_blocks(settled: pd.DataFrame) -> pd.DataFrame:
    active = settled[settled["operation"].isin(["buy", "sell"])].copy()
    if active.empty:
        return pd.DataFrame()

    active["operation_block"] = (
        active["operation"].ne(active["operation"].shift()).cumsum()
    )
    rows = []
    for block_id, block in active.groupby("operation_block", sort=True):
        rows.append(
            {
                "block_id": int(block_id),
                "operation": str(block["operation"].iloc[0]),
                "start_timestamp": block["timestamp"].iloc[0],
                "end_timestamp": block["timestamp"].iloc[-1],
                "intervals": int(block.shape[0]),
                "buy_energy_mwh": float(block["buy_energy_mwh"].sum()),
                "sell_energy_mwh": float(block["sell_energy_mwh"].sum()),
                "forecast_gross_revenue_eur": float(
                    block["forecast_gross_revenue_eur"].sum()
                ),
                "forecast_gross_purchase_eur": float(
                    block["forecast_gross_purchase_eur"].sum()
                ),
                "actual_gross_revenue_eur": float(
                    block["actual_gross_revenue_eur"].sum()
                ),
                "actual_gross_purchase_eur": float(
                    block["actual_gross_purchase_eur"].sum()
                ),
                "degradation_cost_eur": float(block["degradation_cost_eur"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_trade_ledger(
    settled: pd.DataFrame,
    target_day: pd.Timestamp | str,
) -> pd.DataFrame:
    """Build a simple block-pair trade ledger for profitable/failed ratios.

    A trade is defined as one active operation block paired with the next block
    of the opposite operation, for example buy->sell or sell->buy. This mirrors
    the daily arbitrage schedule without requiring changes to the MILP.
    """

    blocks = _operation_blocks(settled)
    target_day = pd.Timestamp(target_day).date().isoformat()
    if blocks.empty:
        return pd.DataFrame(
            columns=[
                "target_date",
                "trade_id",
                "sequence",
                "trade_type",
                "is_completed_buy_sell",
                "start_timestamp",
                "end_timestamp",
                "intervals",
                "buy_energy_mwh",
                "sell_energy_mwh",
                "forecast_net_profit_eur",
                "actual_net_profit_eur",
                "actual_profit_class",
            ]
        )

    rows = []
    trade_id = 1
    index = 0
    while index < len(blocks):
        current = blocks.iloc[index]
        if (
            index + 1 < len(blocks)
            and str(blocks.iloc[index + 1]["operation"]) != str(current["operation"])
        ):
            trade_blocks = blocks.iloc[[index, index + 1]]
            index += 2
        else:
            trade_blocks = blocks.iloc[[index]]
            index += 1

        sequence = "->".join(trade_blocks["operation"].astype(str).tolist())
        if sequence == "buy->sell":
            trade_type = "completed_buy_sell"
        elif sequence == "sell->buy":
            trade_type = "sell_then_buy_cycle_closure"
        else:
            trade_type = f"unpaired_{sequence}"

        forecast_net_profit = float(
            trade_blocks["forecast_gross_revenue_eur"].sum()
            - trade_blocks["forecast_gross_purchase_eur"].sum()
            - trade_blocks["degradation_cost_eur"].sum()
        )
        actual_net_profit = float(
            trade_blocks["actual_gross_revenue_eur"].sum()
            - trade_blocks["actual_gross_purchase_eur"].sum()
            - trade_blocks["degradation_cost_eur"].sum()
        )
        if actual_net_profit > PROFIT_TOLERANCE_EUR:
            profit_class = "profitable"
        elif actual_net_profit < -PROFIT_TOLERANCE_EUR:
            profit_class = "failed"
        else:
            profit_class = "break_even"

        rows.append(
            {
                "target_date": target_day,
                "trade_id": trade_id,
                "sequence": sequence,
                "trade_type": trade_type,
                "is_completed_buy_sell": bool(sequence == "buy->sell"),
                "start_timestamp": trade_blocks["start_timestamp"].min(),
                "end_timestamp": trade_blocks["end_timestamp"].max(),
                "intervals": int(trade_blocks["intervals"].sum()),
                "buy_energy_mwh": float(trade_blocks["buy_energy_mwh"].sum()),
                "sell_energy_mwh": float(trade_blocks["sell_energy_mwh"].sum()),
                "forecast_net_profit_eur": forecast_net_profit,
                "actual_net_profit_eur": actual_net_profit,
                "actual_profit_class": profit_class,
            }
        )
        trade_id += 1

    return pd.DataFrame(rows)


def _trade_ratio_stats(trades: pd.DataFrame, prefix: str = "") -> dict:
    if trades.empty:
        return {
            f"{prefix}trade_count": 0,
            f"{prefix}profitable_trades": 0,
            f"{prefix}failed_trades": 0,
            f"{prefix}break_even_trades": 0,
            f"{prefix}profitable_trade_ratio": 0.0,
            f"{prefix}failed_trade_ratio": 0.0,
            f"{prefix}profit_factor": 0.0,
        }

    profits = trades["actual_net_profit_eur"].astype(float)
    profitable = int((profits > PROFIT_TOLERANCE_EUR).sum())
    failed = int((profits < -PROFIT_TOLERANCE_EUR).sum())
    break_even = int(len(trades) - profitable - failed)
    gross_profit = float(profits[profits > 0.0].sum())
    gross_loss = float(-profits[profits < 0.0].sum())
    return {
        f"{prefix}trade_count": int(len(trades)),
        f"{prefix}profitable_trades": profitable,
        f"{prefix}failed_trades": failed,
        f"{prefix}break_even_trades": break_even,
        f"{prefix}profitable_trade_ratio": safe_divide(profitable, len(trades)),
        f"{prefix}failed_trade_ratio": safe_divide(failed, len(trades)),
        f"{prefix}profit_factor": safe_divide(gross_profit, gross_loss),
    }


def _completed_buy_sell_stats(trades: pd.DataFrame, prefix: str = "") -> dict:
    """Return ratios for strict buy->sell trades only."""

    if trades.empty or "sequence" not in trades.columns:
        completed = trades.iloc[0:0].copy()
    else:
        completed = trades[trades["sequence"] == "buy->sell"].copy()

    base = _trade_ratio_stats(completed, prefix=f"{prefix}completed_buy_sell_")
    base[f"{prefix}excluded_non_buy_sell_ledger_rows"] = int(
        len(trades) - len(completed)
    )
    return base


def summarize_settled_day(
    settled: pd.DataFrame,
    forecast_summary: dict,
    trades: pd.DataFrame,
    target_day: pd.Timestamp,
    test_params: dict,
    degradation_source: str,
    installed_capacity_mw: float,
) -> dict:
    """Summarize one forecast-driven dispatch settled on actual prices."""

    planned_net = float(settled["forecast_interval_profit_eur"].sum())
    actual_net = float(settled["actual_interval_profit_eur"].sum())
    actual_gross_revenue = float(settled["actual_gross_revenue_eur"].sum())
    actual_gross_purchase = float(settled["actual_gross_purchase_eur"].sum())
    degradation = float(settled["degradation_cost_eur"].sum())
    buy_energy = float(settled["buy_energy_mwh"].sum())
    sell_energy = float(settled["sell_energy_mwh"].sum())
    scale_factor = safe_divide(installed_capacity_mw, test_params["p_max"])

    out = {
        "target_date": pd.Timestamp(target_day).date().isoformat(),
        "solver_status": str(forecast_summary["solver_status"]),
        "planned_objective_profit_eur": float(
            forecast_summary["objective_profit_eur"]
        ),
        "planned_net_profit_eur": planned_net,
        "actual_net_profit_eur": actual_net,
        "profit_capture_ratio": safe_divide(actual_net, planned_net),
        "actual_minus_planned_profit_eur": actual_net - planned_net,
        "actual_gross_revenue_eur": actual_gross_revenue,
        "actual_gross_purchase_eur": actual_gross_purchase,
        "degradation_cost_eur": degradation,
        "buy_energy_mwh": buy_energy,
        "sell_energy_mwh": sell_energy,
        "equivalent_discharge_cycles": safe_divide(sell_energy, test_params["e_max"]),
        "actual_net_profit_eur_per_mwh_sold": safe_divide(actual_net, sell_energy),
        "actual_degradation_cost_eur_per_mwh_sold": safe_divide(
            degradation,
            sell_energy,
        ),
        "buy_intervals": int((settled["operation"] == "buy").sum()),
        "sell_intervals": int((settled["operation"] == "sell").sum()),
        "idle_intervals": int((settled["operation"] == "idle").sum()),
        "actual_profitable_day": bool(actual_net > PROFIT_TOLERANCE_EUR),
        "actual_failed_day": bool(actual_net < -PROFIT_TOLERANCE_EUR),
        "min_soc_pct": float(settled["soc_pct"].min()),
        "max_soc_pct": float(settled["soc_pct"].max()),
        "final_soc_pct": float(settled["soc_pct"].iloc[-1]),
        "actual_price_min_eur_mwh": float(settled["actual_price_eur_mwh"].min()),
        "actual_price_max_eur_mwh": float(settled["actual_price_eur_mwh"].max()),
        "actual_price_mean_eur_mwh": float(settled["actual_price_eur_mwh"].mean()),
        "forecast_price_min_eur_mwh": float(
            settled["forecast_price_eur_mwh"].min()
        ),
        "forecast_price_max_eur_mwh": float(
            settled["forecast_price_eur_mwh"].max()
        ),
        "forecast_price_mean_eur_mwh": float(
            settled["forecast_price_eur_mwh"].mean()
        ),
        "battery_power_mw": float(test_params["p_max"]),
        "battery_capacity_mwh": float(test_params["e_max"]),
        "installed_capacity_mw": float(installed_capacity_mw),
        "installed_energy_capacity_mwh": float(test_params["e_max"] * scale_factor),
        "scale_factor": float(scale_factor),
        "park_actual_net_profit_eur": float(actual_net * scale_factor),
        "park_planned_net_profit_eur": float(planned_net * scale_factor),
        "park_actual_gross_revenue_eur": float(actual_gross_revenue * scale_factor),
        "park_actual_gross_purchase_eur": float(actual_gross_purchase * scale_factor),
        "park_degradation_cost_eur": float(degradation * scale_factor),
        "degradation_cost_source": degradation_source,
    }
    out.update(_forecast_error_metrics(settled))
    out.update(_trade_ratio_stats(trades))
    out.update(_completed_buy_sell_stats(trades))
    return out


def summarize_backtest(
    daily_stats: pd.DataFrame,
    trade_stats: pd.DataFrame,
    interval_schedule: pd.DataFrame,
    warnings: Sequence[str],
) -> pd.DataFrame:
    """Aggregate all daily and trade statistics into a one-row summary."""

    actual_net = float(daily_stats["actual_net_profit_eur"].sum())
    planned_net = float(daily_stats["planned_net_profit_eur"].sum())
    profitable_days = int(daily_stats["actual_profitable_day"].sum())
    failed_days = int(daily_stats["actual_failed_day"].sum())
    break_even_days = int(len(daily_stats) - profitable_days - failed_days)
    out = {
        "start_date": str(daily_stats["target_date"].min()),
        "end_date": str(daily_stats["target_date"].max()),
        "days": int(len(daily_stats)),
        "optimal_days": int((daily_stats["solver_status"] == "Optimal").sum()),
        "planned_net_profit_eur": planned_net,
        "actual_net_profit_eur": actual_net,
        "actual_minus_planned_profit_eur": actual_net - planned_net,
        "profit_capture_ratio": safe_divide(actual_net, planned_net),
        "profitable_days": profitable_days,
        "failed_days": failed_days,
        "break_even_days": break_even_days,
        "profitable_day_ratio": safe_divide(profitable_days, len(daily_stats)),
        "failed_day_ratio": safe_divide(failed_days, len(daily_stats)),
        "buy_energy_mwh": float(daily_stats["buy_energy_mwh"].sum()),
        "sell_energy_mwh": float(daily_stats["sell_energy_mwh"].sum()),
        "degradation_cost_eur": float(daily_stats["degradation_cost_eur"].sum()),
        "actual_gross_revenue_eur": float(
            daily_stats["actual_gross_revenue_eur"].sum()
        ),
        "actual_gross_purchase_eur": float(
            daily_stats["actual_gross_purchase_eur"].sum()
        ),
        "actual_net_profit_eur_per_mwh_sold": safe_divide(
            actual_net,
            float(daily_stats["sell_energy_mwh"].sum()),
        ),
        "mean_daily_actual_net_profit_eur": float(
            daily_stats["actual_net_profit_eur"].mean()
        ),
        "median_daily_actual_net_profit_eur": float(
            daily_stats["actual_net_profit_eur"].median()
        ),
        "worst_daily_actual_net_profit_eur": float(
            daily_stats["actual_net_profit_eur"].min()
        ),
        "best_daily_actual_net_profit_eur": float(
            daily_stats["actual_net_profit_eur"].max()
        ),
        "forecast_mae_eur_mwh": float(
            interval_schedule["price_error_eur_mwh"].abs().mean()
        ),
        "forecast_rmse_eur_mwh": float(
            np.sqrt(np.mean(interval_schedule["price_error_eur_mwh"] ** 2))
        ),
        "forecast_bias_eur_mwh": float(interval_schedule["price_error_eur_mwh"].mean()),
        "warnings": " | ".join(dict.fromkeys(warnings)),
    }
    out.update(_trade_ratio_stats(trade_stats))
    out.update(_completed_buy_sell_stats(trade_stats))

    for col in [
        "park_actual_net_profit_eur",
        "park_planned_net_profit_eur",
        "park_actual_gross_revenue_eur",
        "park_actual_gross_purchase_eur",
        "park_degradation_cost_eur",
    ]:
        if col in daily_stats.columns:
            out[col] = float(daily_stats[col].sum())

    metadata_cols = [
        "installed_capacity_mw",
        "installed_energy_capacity_mwh",
        "scale_factor",
        "battery_power_mw",
        "battery_capacity_mwh",
        "degradation_cost_source",
    ]
    for col in metadata_cols:
        if col in daily_stats.columns:
            out[col] = daily_stats[col].iloc[0]

    return pd.DataFrame([out])


def run_forecast_strategy_backtest(
    input_file: Path = DEFAULT_PRICE_SIGNALS_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
    backtest_days: int | None = DEFAULT_BACKTEST_DAYS,
    window_days: int = DEFAULT_WINDOW_DAYS,
    model: str = "seasonal",
    degradation_source: str = DEFAULT_DEGRADATION_SOURCE,
    temperature_c: float = DEFAULT_LUT_TEMPERATURE_C,
    installed_capacity_mw: float = DEFAULT_INSTALLED_CAPACITY_MW,
    timestamp_col: str | None = None,
    price_col: str | None = None,
    allow_hourly_upsampling: bool = False,
    solver_msg: bool = False,
) -> ForecastStrategyBacktestResult:
    """Run a historical backtest where forecast dispatch is settled on actuals."""

    price_history = load_price_history(
        input_file=input_file,
        timestamp_col=timestamp_col,
        price_col=price_col,
        allow_hourly_upsampling=allow_hourly_upsampling,
    )
    warnings: list[str] = list(price_history.warnings)
    test_params = build_test_params()
    energy_points, cost_points, degradation_source_label = load_degradation_curve(
        source=degradation_source,
        test_params=test_params,
        temperature_c=temperature_c,
    )
    target_days = select_backtest_days(
        price_history.frame,
        start_date=start_date,
        end_date=end_date,
        backtest_days=backtest_days,
    )

    schedules = []
    daily_rows = []
    trade_frames = []
    for target_day in target_days:
        actual_prices = extract_actual_day_prices(
            price_history.frame,
            target_day=target_day,
            dt=test_params["dt"],
        )
        forecast_prices, forecast_frame, forecast_warnings = forecast_series_for_day(
            price_history.frame,
            target_day=target_day,
            window_days=window_days,
            model=model,
        )
        warnings.extend(f"{target_day.date()}: {item}" for item in forecast_warnings)

        forecast_schedule, forecast_summary = optimize_dispatch_on_prices(
            prices=forecast_prices,
            test_params=test_params,
            energy_points=energy_points,
            cost_points=cost_points,
            solver_msg=solver_msg,
        )
        settled = settle_schedule_on_actual_prices(
            forecast_schedule=forecast_schedule,
            actual_prices=actual_prices,
            forecast_frame=forecast_frame,
        )
        settled.insert(0, "target_date", target_day.date().isoformat())
        trades = build_trade_ledger(settled, target_day=target_day)
        daily_summary = summarize_settled_day(
            settled=settled,
            forecast_summary=forecast_summary,
            trades=trades,
            target_day=target_day,
            test_params=test_params,
            degradation_source=degradation_source_label,
            installed_capacity_mw=installed_capacity_mw,
        )

        schedules.append(settled)
        daily_rows.append(daily_summary)
        if not trades.empty:
            trade_frames.append(trades)

    interval_schedule = pd.concat(schedules, ignore_index=True)
    daily_stats = pd.DataFrame(daily_rows)
    trade_stats = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else build_trade_ledger(interval_schedule.iloc[0:0], target_days[0])
    )
    summary = summarize_backtest(
        daily_stats=daily_stats,
        trade_stats=trade_stats,
        interval_schedule=interval_schedule,
        warnings=warnings,
    )
    return ForecastStrategyBacktestResult(
        interval_schedule=interval_schedule,
        daily_stats=daily_stats,
        trade_stats=trade_stats,
        summary=summary,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def write_report(
    result: ForecastStrategyBacktestResult,
    report_path: Path,
    schedule_path: Path,
    daily_path: Path,
    trades_path: Path,
    summary_path: Path,
) -> None:
    summary = result.summary.iloc[0].to_dict()
    lines = [
        "# Forecast Strategy Backtest",
        "",
        "The optimizer was run on forecast prices. The resulting dispatch was settled on actual prices.",
        "",
        "## Outputs",
        f"- Interval schedule: `{schedule_path.name}`",
        f"- Daily statistics: `{daily_path.name}`",
        f"- Trade statistics: `{trades_path.name}`",
        f"- Summary: `{summary_path.name}`",
        "",
        "## Key Metrics",
        f"- Date range: {summary['start_date']} to {summary['end_date']}",
        f"- Days: {int(summary['days'])}",
        f"- Actual net profit: {summary['actual_net_profit_eur']:.2f} EUR",
        f"- Planned net profit: {summary['planned_net_profit_eur']:.2f} EUR",
        f"- Profit capture ratio: {summary['profit_capture_ratio']:.3f}",
        (
            "- Completed buy->sell profitable / failed trades: "
            f"{int(summary['completed_buy_sell_profitable_trades'])} / "
            f"{int(summary['completed_buy_sell_failed_trades'])} "
            f"({summary['completed_buy_sell_profitable_trade_ratio']:.3f} / "
            f"{summary['completed_buy_sell_failed_trade_ratio']:.3f})"
        ),
        (
            "- All ledger profitable / failed rows: "
            f"{int(summary['profitable_trades'])} / {int(summary['failed_trades'])} "
            f"({summary['profitable_trade_ratio']:.3f} / "
            f"{summary['failed_trade_ratio']:.3f})"
        ),
        (
            "- Non buy->sell ledger rows excluded from strict trade ratio: "
            f"{int(summary['excluded_non_buy_sell_ledger_rows'])}"
        ),
        (
            "- Profitable / failed days: "
            f"{int(summary['profitable_days'])} / {int(summary['failed_days'])} "
            f"({summary['profitable_day_ratio']:.3f} / "
            f"{summary['failed_day_ratio']:.3f})"
        ),
        f"- Forecast MAE: {summary['forecast_mae_eur_mwh']:.2f} EUR/MWh",
        f"- Forecast RMSE: {summary['forecast_rmse_eur_mwh']:.2f} EUR/MWh",
        "",
        "## Trade Definition",
        (
            "The strict completed-trade ratio uses only buy->sell rows. The "
            "ledger also keeps sell->buy and unpaired buy/sell rows because "
            "they are real parts of the MILP dispatch and daily P&L, but those "
            "rows are excluded from the completed buy->sell ratio."
        ),
        "",
        "## Full Summary",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.6f}")
        else:
            lines.append(f"- {key}: {value}")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_backtest_outputs(
    result: ForecastStrategyBacktestResult,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_prefix: str = "forecast_strategy_backtest",
) -> dict[str, Path]:
    """Write interval, daily, trade, summary, and report outputs."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "schedule": output_dir / f"{output_prefix}_schedule.csv",
        "daily_stats": output_dir / f"{output_prefix}_daily_stats.csv",
        "trade_stats": output_dir / f"{output_prefix}_trade_stats.csv",
        "summary": output_dir / f"{output_prefix}_summary.csv",
        "report": output_dir / f"{output_prefix}_report.md",
    }
    result.interval_schedule.to_csv(paths["schedule"], index=False)
    result.daily_stats.to_csv(paths["daily_stats"], index=False)
    result.trade_stats.to_csv(paths["trade_stats"], index=False)
    result.summary.to_csv(paths["summary"], index=False)
    write_report(
        result=result,
        report_path=paths["report"],
        schedule_path=paths["schedule"],
        daily_path=paths["daily_stats"],
        trades_path=paths["trade_stats"],
        summary_path=paths["summary"],
    )
    return paths


def _print_warnings(warnings: Iterable[str]) -> None:
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest forecast-driven BESS dispatch by optimizing on predicted "
            "prices and settling the dispatch on actual prices."
        ),
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_PRICE_SIGNALS_PATH),
        help="CSV/XLSX file with actual historical DAM prices.",
    )
    parser.add_argument("--start-date", help="First target day, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Last target day, YYYY-MM-DD.")
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=DEFAULT_BACKTEST_DAYS,
        help="Use the last N complete days after date filtering. Set 0 to use all.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Recent days used by the forecasting baseline.",
    )
    parser.add_argument(
        "--model",
        choices=["seasonal"],
        default="seasonal",
        help="Forecast model to run.",
    )
    parser.add_argument(
        "--degradation-source",
        choices=["lut", "pybamm_only", "dummy"],
        default=DEFAULT_DEGRADATION_SOURCE,
        help="Degradation curve used by the existing optimizer.",
    )
    parser.add_argument(
        "--temperature-c",
        type=float,
        default=DEFAULT_LUT_TEMPERATURE_C,
        help="Temperature slice for LUT-based degradation costs.",
    )
    parser.add_argument(
        "--installed-capacity-mw",
        type=float,
        default=DEFAULT_INSTALLED_CAPACITY_MW,
        help="Park scale-up capacity used for scaled economic statistics.",
    )
    parser.add_argument("--timestamp-col", help="Timestamp column name.")
    parser.add_argument("--price-col", help="Price column name.")
    parser.add_argument(
        "--allow-hourly-upsampling",
        action="store_true",
        help="Allow hourly input to be forward-filled into 15-minute slots.",
    )
    parser.add_argument("--solver-msg", action="store_true", help="Show CBC logs.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for backtest outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="forecast_strategy_backtest",
        help="Filename prefix for backtest outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    font_cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "bess_optimization_cache"
    font_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(font_cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(font_cache_root / "xdg"))

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    backtest_days = None if int(args.backtest_days) == 0 else int(args.backtest_days)

    try:
        result = run_forecast_strategy_backtest(
            input_file=Path(args.input_file),
            start_date=args.start_date,
            end_date=args.end_date,
            backtest_days=backtest_days,
            window_days=args.window_days,
            model=args.model,
            degradation_source=args.degradation_source,
            temperature_c=args.temperature_c,
            installed_capacity_mw=args.installed_capacity_mw,
            timestamp_col=args.timestamp_col,
            price_col=args.price_col,
            allow_hourly_upsampling=args.allow_hourly_upsampling,
            solver_msg=args.solver_msg,
        )
        paths = write_backtest_outputs(
            result,
            output_dir=Path(args.output_dir),
            output_prefix=args.output_prefix,
        )
        _print_warnings(result.warnings)
        summary = result.summary.iloc[0]
        print("Forecast strategy backtest completed.")
        print(f"Date range: {summary['start_date']} to {summary['end_date']}")
        print(f"Days: {int(summary['days'])}")
        print(f"Actual net profit: {summary['actual_net_profit_eur']:.2f} EUR")
        print(f"Planned net profit: {summary['planned_net_profit_eur']:.2f} EUR")
        print(f"Profit capture ratio: {summary['profit_capture_ratio']:.3f}")
        print(
            "Completed buy->sell profitable/failed trades: "
            f"{int(summary['completed_buy_sell_profitable_trades'])}/"
            f"{int(summary['completed_buy_sell_failed_trades'])}"
        )
        print(
            "All ledger profitable/failed rows: "
            f"{int(summary['profitable_trades'])}/"
            f"{int(summary['failed_trades'])}"
        )
        for label, path in paths.items():
            print(f"{label}: {path}")
        return 0
    except (ForecastingError, ValueError, RuntimeError) as exc:
        print(f"Forecast strategy backtest error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
