"""Benchmark strategy adapters for optimization backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp as pl

from .backtest_utils import (
    build_schedule_from_arrays,
    summarize,
    validate_schedule,
    values_by_step,
)
from .naive import bess_ema
from .perfect import bess_order_perfect


BENCHMARK_LABELS = {
    "perfect": "Perfect-foresight MILP (no degradation cost)",
    "naive": "EMA threshold heuristic (no degradation cost)",
}
ZERO_DEGRADATION_SOURCE = "zero degradation cost benchmark"


def normalize_benchmark_id(model_id: str) -> str:
    normalized = str(model_id).strip().lower()
    aliases = {
        "perfect_no_degradation_milp": "perfect",
        "perfect_milp": "perfect",
        "no_degradation": "perfect",
        "naive_ema_heuristic": "naive",
        "ema": "naive",
    }
    return aliases.get(normalized, normalized)


def _benchmark_params(test_params: dict, terminal_soc_mode: str) -> dict:
    benchmark_params = dict(test_params)
    benchmark_params["dt"] = benchmark_params.get("dt", benchmark_params.get("DT", 0.25))
    benchmark_params["terminal_soc_mode"] = terminal_soc_mode
    return benchmark_params


def _add_benchmark_metadata(
    summary: dict,
    schedule: pd.DataFrame,
    prices: pd.Series,
    test_params: dict,
    model_id: str,
    model_label: str,
) -> tuple[pd.DataFrame, dict]:
    out = dict(summary)
    out["date"] = prices.index.min().date().isoformat()
    out["price_min_eur_mwh"] = float(prices.min())
    out["price_max_eur_mwh"] = float(prices.max())
    out["price_mean_eur_mwh"] = float(prices.mean())
    out["battery_power_mw"] = float(test_params["p_max"])
    out["battery_capacity_mwh"] = float(test_params["e_max"])
    out["benchmark_model"] = model_id
    out["model_label"] = model_label
    out["zero_degradation_benchmark"] = True
    out["degradation_cost_source"] = ZERO_DEGRADATION_SOURCE

    annotated = schedule.copy()
    annotated["benchmark_model"] = model_id
    annotated["model_label"] = model_label
    return annotated, out


def _run_perfect_benchmark(
    prices: pd.Series,
    test_params: dict,
    solver_msg: bool,
) -> tuple[pd.DataFrame, dict]:
    benchmark_params = _benchmark_params(test_params, terminal_soc_mode="equal_initial")
    problem, p_buy, p_sell, soc = bess_order_perfect(
        prices=prices.to_numpy(dtype=float),
        battery_params=benchmark_params,
        solver_msg=solver_msg,
    )
    periods = len(prices)
    schedule = build_schedule_from_arrays(
        prices=prices,
        p_buy_mw=values_by_step(p_buy, periods),
        p_sell_mw=values_by_step(p_sell, periods),
        soc_mwh=values_by_step(soc, periods),
        degradation_cost_eur=np.zeros(periods, dtype=float),
        test_params=benchmark_params,
    )
    status_name = pl.LpStatus[problem.status]
    objective_value = float(pl.value(problem.objective) or 0.0)
    summary = summarize(schedule, objective_value, status_name, benchmark_params)
    validate_schedule(schedule, summary, benchmark_params)
    return _add_benchmark_metadata(
        summary,
        schedule,
        prices,
        benchmark_params,
        "perfect",
        BENCHMARK_LABELS["perfect"],
    )


def _run_naive_benchmark(
    prices: pd.Series,
    test_params: dict,
    alpha_factor: float,
    margin: float,
    initial_soc_mwh: float | None,
) -> tuple[pd.DataFrame, dict]:
    benchmark_params = _benchmark_params(test_params, terminal_soc_mode="free")
    _, p_buy, p_sell, soc = bess_ema(
        prices=prices.to_numpy(dtype=float),
        alpha_factor=alpha_factor,
        margin=margin,
        battery_params=benchmark_params,
        initial_soc_mwh=initial_soc_mwh,
    )
    periods = len(prices)
    schedule = build_schedule_from_arrays(
        prices=prices,
        p_buy_mw=p_buy,
        p_sell_mw=p_sell,
        soc_mwh=soc,
        degradation_cost_eur=np.zeros(periods, dtype=float),
        test_params=benchmark_params,
    )
    objective_value = float(schedule["interval_profit_eur"].sum())
    summary = summarize(schedule, objective_value, "Heuristic", benchmark_params)
    summary["naive_alpha_factor"] = float(alpha_factor)
    summary["naive_margin"] = float(margin)
    validate_schedule(
        schedule,
        summary,
        benchmark_params,
        require_optimal=False,
    )
    return _add_benchmark_metadata(
        summary,
        schedule,
        prices,
        benchmark_params,
        "naive",
        BENCHMARK_LABELS["naive"],
    )


def run_benchmark_model(
    model_id: str,
    prices: pd.Series,
    test_params: dict,
    solver_msg: bool = False,
    naive_alpha_factor: float = 0.3,
    naive_margin: float = 0.05,
    initial_soc_mwh: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run one benchmark strategy and return a normalized schedule and summary."""
    normalized = normalize_benchmark_id(model_id)
    if normalized == "perfect":
        return _run_perfect_benchmark(
            prices=prices,
            test_params=test_params,
            solver_msg=solver_msg,
        )
    if normalized == "naive":
        return _run_naive_benchmark(
            prices=prices,
            test_params=test_params,
            alpha_factor=naive_alpha_factor,
            margin=naive_margin,
            initial_soc_mwh=initial_soc_mwh,
        )
    raise ValueError(f"Unknown benchmark model: {model_id}")
