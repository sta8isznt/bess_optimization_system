from typing import Iterable, Optional, Sequence

import pulp as pl

try:
    from .config import params as DEFAULT_PARAMS
except ImportError:  # Allows running this file from inside optimization/.
    from config import params as DEFAULT_PARAMS


DEFAULT_ENERGY_POINTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]  # MWh
DEFAULT_COST_POINTS = [0.0, 0.10, 0.25, 0.50, 0.90, 1.50]  # EUR


def _normalise_params(battery_params: Optional[dict]) -> dict:
    normalized = dict(DEFAULT_PARAMS if battery_params is None else battery_params)
    if "dt" not in normalized and "DT" in normalized:
        normalized["dt"] = normalized["DT"]

    required = [
        "p_max",
        "e_max",
        "eta_ch",
        "eta_dis",
        "soc_min",
        "soc_max",
        "soc_init",
        "dt",
    ]
    missing = [key for key in required if key not in normalized]
    if missing:
        raise ValueError(f"Missing BESS parameter(s): {', '.join(missing)}")

    return normalized


def _validate_curve(
    energy_points: Sequence[float],
    cost_points: Sequence[float],
    max_interval_energy: float,
) -> tuple[list[float], list[float]]:
    energy = [float(point) for point in energy_points]
    cost = [float(point) for point in cost_points]

    if len(energy) != len(cost):
        raise ValueError("Degradation energy and cost curves must have the same length.")
    if len(energy) < 2:
        raise ValueError("Degradation curve needs at least two points.")
    if energy[0] != 0.0:
        raise ValueError("Degradation energy curve must start at 0.0 MWh.")
    if any(right <= left for left, right in zip(energy, energy[1:])):
        raise ValueError("Degradation energy curve must be strictly increasing.")
    if any(point < 0 for point in cost):
        raise ValueError("Degradation cost curve cannot contain negative costs.")
    if energy[-1] < max_interval_energy:
        raise ValueError(
            "Degradation energy curve must cover the maximum discharge energy per interval."
        )

    return energy, cost


def _default_solver(msg: bool) -> pl.LpSolver:
    solver = pl.PULP_CBC_CMD(msg=msg)
    if solver.available():
        return solver

    available_solvers = pl.listSolvers(onlyAvailable=True)
    if available_solvers:
        return pl.getSolver(available_solvers[0], msg=msg)

    raise RuntimeError(
        "No MILP solver is available. Install CBC or use PuLP's bundled CBC solver."
    )


def bess_order(
    prices: Iterable[float],
    battery_params: Optional[dict] = None,
    degradation_energy_points: Optional[Sequence[float]] = None,
    degradation_cost_points: Optional[Sequence[float]] = None,
    solver: Optional[pl.LpSolver] = None,
    solver_msg: bool = False,
):
    price_values = [float(price) for price in prices]
    if not price_values:
        raise ValueError("prices must contain at least one timestep.")

    bess_params = _normalise_params(battery_params)
    time_steps = range(len(price_values))  # 96 for a 24h, 15-minute horizon.

    energy_points, cost_points = _validate_curve(
        DEFAULT_ENERGY_POINTS
        if degradation_energy_points is None
        else degradation_energy_points,
        DEFAULT_COST_POINTS if degradation_cost_points is None else degradation_cost_points,
        bess_params["p_max"] * bess_params["dt"],
    )
    curve_points = range(len(energy_points))

    obj_f = pl.LpProblem("BESS_Optimization_Engine", pl.LpMaximize)

    # Variables
    p_buy = pl.LpVariable.dicts(
        "P_buy",
        time_steps,
        lowBound=0,
        upBound=bess_params["p_max"],
        cat=pl.LpContinuous,
    )
    p_sell = pl.LpVariable.dicts(
        "P_sell",
        time_steps,
        lowBound=0,
        upBound=bess_params["p_max"],
        cat=pl.LpContinuous,
    )
    soc = pl.LpVariable.dicts(
        "SoC",
        time_steps,
        lowBound=bess_params["soc_min"] * bess_params["e_max"],
        upBound=bess_params["soc_max"] * bess_params["e_max"],
        cat=pl.LpContinuous,
    )
    u = pl.LpVariable.dicts("u", time_steps, cat=pl.LpBinary)

    lambdas = pl.LpVariable.dicts(
        "Lambda",
        (time_steps, curve_points),
        lowBound=0,
        upBound=1,
        cat=pl.LpContinuous,
    )
    dynamic_deg_cost = pl.LpVariable.dicts(
        "Degradation_Cost",
        time_steps,
        lowBound=0,
        cat=pl.LpContinuous,
    )

    # Constraints
    for t in time_steps:
        # No Simultaneous Operation Exclusion
        obj_f += p_buy[t] <= bess_params["p_max"] * u[t], f"Max_Charge_{t}"
        obj_f += (
            p_sell[t] <= bess_params["p_max"] * (1 - u[t]),
            f"Max_Discharge_{t}",
        )

        # Energy Balance Constraint
        if t == 0:
            prev_soc = bess_params["soc_init"] * bess_params["e_max"]  # MWh
        else:
            prev_soc = soc[t-1]

        obj_f += (
            soc[t]
            == prev_soc
            + (
                p_buy[t] * bess_params["eta_ch"]
                - p_sell[t] / bess_params["eta_dis"]
            )
            * bess_params["dt"],
            f"SoC_Balance_{t}",
        )

        # Convex combination over the degradation-cost lookup curve.
        obj_f += (
            pl.lpSum([lambdas[t][k] for k in curve_points]) == 1,
            f"Lambda_Sum_{t}",
        )
        obj_f += (
            (p_sell[t] * bess_params["dt"])
            == pl.lpSum([lambdas[t][k] * energy_points[k] for k in curve_points]),
            f"Energy_Interp_{t}",
        )
        obj_f += (
            dynamic_deg_cost[t]
            == pl.lpSum([lambdas[t][k] * cost_points[k] for k in curve_points]),
            f"Cost_Interp_{t}",
        )

    # Cycle Continuity Constraint
    last_step = time_steps[-1]
    obj_f += (
        soc[last_step] == bess_params["soc_init"] * bess_params["e_max"],
        "Cycle_Continuity",
    )

    # Objective Function: Maximize Total Profit
    obj_f += (
        pl.lpSum(
            [
                (price_values[t] * (p_sell[t] - p_buy[t]) * bess_params["dt"])
                - dynamic_deg_cost[t]
                for t in time_steps
            ]
        ),
        "Total_Profit",
    )

    obj_f.solve(solver or _default_solver(msg=solver_msg))

    return obj_f, p_buy, p_sell, soc, dynamic_deg_cost
