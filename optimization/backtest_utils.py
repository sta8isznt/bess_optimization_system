"""Shared helpers for BESS optimization backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pulp as pl


def safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def dummy_degradation_cost(discharge_mwh: np.ndarray) -> np.ndarray:
    """Simple convex dummy cost curve in EUR per interval discharge energy."""
    return 0.25 * discharge_mwh + 2.0 * discharge_mwh**2


def build_dummy_cost_curve(
    test_params: dict,
    points: int = 8,
) -> tuple[list[float], list[float]]:
    max_interval_energy = test_params["p_max"] * test_params["dt"]
    energy_points = np.linspace(0.0, max_interval_energy, points)
    cost_points = dummy_degradation_cost(energy_points)
    return energy_points.tolist(), cost_points.tolist()


def values_by_step(variables: dict, periods: int) -> np.ndarray:
    return np.array([pl.value(variables[t]) or 0.0 for t in range(periods)], dtype=float)


def build_schedule(
    prices: pd.Series,
    p_buy: dict,
    p_sell: dict,
    soc: dict,
    degradation_cost: dict,
    test_params: dict,
) -> pd.DataFrame:
    periods = len(prices)
    p_buy_mw = values_by_step(p_buy, periods)
    p_sell_mw = values_by_step(p_sell, periods)
    soc_mwh = values_by_step(soc, periods)
    degradation_cost_eur = values_by_step(degradation_cost, periods)
    dt = test_params["dt"]

    schedule = pd.DataFrame(
        {
            "timestamp": prices.index,
            "price_eur_mwh": prices.to_numpy(dtype=float),
            "p_buy_mw": p_buy_mw,
            "p_sell_mw": p_sell_mw,
            "net_export_mw": p_sell_mw - p_buy_mw,
            "buy_energy_mwh": p_buy_mw * dt,
            "sell_energy_mwh": p_sell_mw * dt,
            "soc_mwh": soc_mwh,
            "soc_pct": soc_mwh / test_params["e_max"],
            "degradation_cost_eur": degradation_cost_eur,
        }
    )
    schedule["gross_revenue_eur"] = (
        schedule["price_eur_mwh"] * schedule["sell_energy_mwh"]
    )
    schedule["gross_purchase_eur"] = (
        schedule["price_eur_mwh"] * schedule["buy_energy_mwh"]
    )
    schedule["interval_profit_eur"] = (
        schedule["gross_revenue_eur"]
        - schedule["gross_purchase_eur"]
        - schedule["degradation_cost_eur"]
    )
    schedule["operation"] = np.select(
        [schedule["p_sell_mw"] > 1e-7, schedule["p_buy_mw"] > 1e-7],
        ["sell", "buy"],
        default="idle",
    )
    return schedule


def summarize(
    schedule: pd.DataFrame,
    objective_value: float,
    status_name: str,
    test_params: dict,
) -> dict:
    gross_revenue = float(schedule["gross_revenue_eur"].sum())
    gross_purchase = float(schedule["gross_purchase_eur"].sum())
    degradation_cost = float(schedule["degradation_cost_eur"].sum())
    net_profit = float(schedule["interval_profit_eur"].sum())
    buy_energy = float(schedule["buy_energy_mwh"].sum())
    sell_energy = float(schedule["sell_energy_mwh"].sum())

    return {
        "solver_status": status_name,
        "objective_profit_eur": float(objective_value),
        "net_profit_eur": net_profit,
        "gross_revenue_eur": gross_revenue,
        "gross_purchase_eur": gross_purchase,
        "degradation_cost_eur": degradation_cost,
        "buy_energy_mwh": buy_energy,
        "sell_energy_mwh": sell_energy,
        "equivalent_discharge_cycles": safe_divide(sell_energy, test_params["e_max"]),
        "gross_revenue_eur_per_mwh_sold": safe_divide(gross_revenue, sell_energy),
        "gross_purchase_eur_per_mwh_bought": safe_divide(gross_purchase, buy_energy),
        "net_profit_eur_per_mwh_sold": safe_divide(net_profit, sell_energy),
        "degradation_cost_eur_per_mwh_sold": safe_divide(
            degradation_cost,
            sell_energy,
        ),
        "net_profit_eur_per_mwh_capacity": safe_divide(
            net_profit,
            test_params["e_max"],
        ),
        "buy_intervals": int((schedule["p_buy_mw"] > 1e-7).sum()),
        "sell_intervals": int((schedule["p_sell_mw"] > 1e-7).sum()),
        "idle_intervals": int((schedule["operation"] == "idle").sum()),
        "min_soc_pct": float(schedule["soc_pct"].min()),
        "max_soc_pct": float(schedule["soc_pct"].max()),
        "final_soc_pct": float(schedule["soc_pct"].iloc[-1]),
        "total_throughput_mwh": float(buy_energy + sell_energy),
    }


def validate_schedule(schedule: pd.DataFrame, summary: dict, test_params: dict) -> None:
    tolerance = 1e-5
    errors = []

    if summary["solver_status"] != "Optimal":
        errors.append(f"solver status is {summary['solver_status']}, expected Optimal")

    simultaneous = (schedule["p_buy_mw"] > tolerance) & (schedule["p_sell_mw"] > tolerance)
    if simultaneous.any():
        errors.append(f"{int(simultaneous.sum())} intervals buy and sell simultaneously")

    soc_min = test_params["soc_min"] * test_params["e_max"]
    soc_max = test_params["soc_max"] * test_params["e_max"]
    if schedule["soc_mwh"].lt(soc_min - tolerance).any():
        errors.append("SoC falls below configured minimum")
    if schedule["soc_mwh"].gt(soc_max + tolerance).any():
        errors.append("SoC exceeds configured maximum")

    terminal_mode = str(test_params.get("terminal_soc_mode", "equal_initial")).strip().lower()
    initial_soc = test_params["soc_init"] * test_params["e_max"]
    if terminal_mode != "free":
        terminal_soc = schedule["soc_mwh"].iloc[-1]
        if abs(terminal_soc - initial_soc) > tolerance:
            errors.append(
                f"terminal SoC {terminal_soc:.6f} MWh does not match initial "
                f"{initial_soc:.6f} MWh"
            )

    previous_soc = np.r_[initial_soc, schedule["soc_mwh"].to_numpy()[:-1]]
    expected_soc = previous_soc + (
        schedule["p_buy_mw"].to_numpy() * test_params["eta_ch"]
        - schedule["p_sell_mw"].to_numpy() / test_params["eta_dis"]
    ) * test_params["dt"]
    balance_error = np.max(np.abs(schedule["soc_mwh"].to_numpy() - expected_soc))
    if balance_error > tolerance:
        errors.append(f"SoC balance max error is {balance_error:.6e} MWh")

    if errors:
        raise RuntimeError("Optimization validation failed:\n- " + "\n- ".join(errors))
