#!/usr/bin/env python3
"""
BESS runtime pipeline matched to the PuLP optimization engine abstraction.

Purpose
-------
This version deliberately simplifies the BESS pipeline so that its degradation
input matches the optimization engine in engine.py:

    degradation_energy_points -> MWh discharged per timestep
    degradation_cost_points   -> EUR degradation cost per timestep

It does NOT run PyBaMM. PyBaMM remains an offline calibration layer that creates:

    degradation_lut_pybamm_calibrated.csv

This script converts that calibrated LUT into optimizer-ready piecewise-linear
points and, if PuLP is installed, solves the same MILP structure as the engine:

    maximize sum_t price_t * (P_sell_t - P_buy_t) * dt - degradation_cost_t

Main simplifications versus the previous runtime BESS pipeline
--------------------------------------------------------------
1. Degradation cost depends only on discharged energy per timestep.
2. The engine is not temperature-aware, so a single reference temperature slice
   is used, default 25°C.
3. The engine uses EUR/timestep degradation cost points, not EUR/MWh directly.
4. SoH is reported diagnostically after solving; it is not an optimization state.
5. PyBaMM is not imported or executed at runtime.

Run
---
python bess_pipeline_engine_matched.py --input-dir . --output-dir .

If PuLP/CBC is not installed, the script still writes the degradation points and
market input, then reports that the MILP solve was skipped. To solve locally:

    pip install pulp

Outputs
-------
engine_degradation_points.csv
engine_degradation_points.json
bess_market_input_engine_matched.csv
bess_dispatch_engine_matched.csv               # only if solve succeeds
bess_financial_result_engine_matched.csv       # only if solve succeeds
bess_financial_summary_engine_matched.csv      # only if solve succeeds
bess_engine_matched_manifest.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineMatchedConfig:
    # Same battery dimensions used by the optimizer engine.
    E_nom_MWh: float = 2.0
    P_max_MW: float = 1.0
    dt_h: float = 0.25

    soc_min: float = 0.10
    soc_max: float = 0.90
    soc_init: float = 0.50

    eta_charge: float = 0.92
    eta_discharge: float = 0.92

    # For 2 MWh / 1 MW this gives 0.5C and max energy per 15 min = 0.25 MWh.
    c_rate_max: float = 0.50

    # Engine simplification: choose one temperature slice from the calibrated LUT.
    engine_temperature_c: float = 25.0

    # Engine-friendly low-dimensional breakpoints.
    # For E=2 MWh and DoD max=0.125, max discharged energy per step is 0.25 MWh.
    n_engine_breakpoints: int = 6

    # Optional SoH diagnostic conversion after the MILP solve.
    replacement_cost_eur_per_MWh_capacity: float = 30_000.0
    soh_eol: float = 0.80

    # The engine objective uses raw EUR. These normalized columns are exported
    # for ML/diagnostics, but the MILP uses EUR to stay consistent with prices.
    export_normalized_cost: bool = True
    degradation_cost_scale: float = 1.0

    @property
    def physical_c_rate_from_power(self) -> float:
        return self.P_max_MW / self.E_nom_MWh

    @property
    def effective_c_rate_max(self) -> float:
        return min(self.c_rate_max, self.physical_c_rate_from_power)

    @property
    def p_max_effective_MW(self) -> float:
        return min(self.P_max_MW, self.effective_c_rate_max * self.E_nom_MWh)

    @property
    def max_dod_per_step(self) -> float:
        return self.effective_c_rate_max * self.dt_h

    @property
    def max_energy_per_step_MWh(self) -> float:
        return self.max_dod_per_step * self.E_nom_MWh

    @property
    def replacement_total_cost_eur(self) -> float:
        return self.replacement_cost_eur_per_MWh_capacity * self.E_nom_MWh

    def as_engine_params(self) -> Dict[str, float]:
        """Return the param shape expected by engine.py."""
        return {
            "e_max": self.E_nom_MWh,
            "p_max": self.p_max_effective_MW,
            "dt": self.dt_h,
            "eta_ch": self.eta_charge,
            "eta_dis": self.eta_discharge,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "soc_init": self.soc_init,
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _find_first_existing(input_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for name in candidates:
        p = input_dir / name
        if p.exists():
            return p
    return None


def load_market_input(input_dir: Path) -> Tuple[pd.DataFrame, str]:
    source = _find_first_existing(
        input_dir,
        [
            "market_input.csv",
            "bess_market_input.csv",
            "bess_market_input_realistic.csv",
            "bess_market_input_pybamm_calibrated.csv",
            "bess_dispatch_fixed.csv",
            "bess_financial_result_fixed.csv",
            "bess_dispatch.csv",
            "bess_financial_result.csv",
        ],
    )

    if source is not None:
        df = pd.read_csv(source)
        if "timestamp" not in df.columns:
            raise ValueError(f"{source.name} must contain a timestamp column.")
        if "price_eur_per_MWh" not in df.columns:
            raise ValueError(f"{source.name} must contain price_eur_per_MWh.")

        out = pd.DataFrame()
        out["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
        out["price_eur_per_MWh"] = pd.to_numeric(df["price_eur_per_MWh"], errors="raise")

        if "T_amb_C" in df.columns:
            out["T_amb_C"] = pd.to_numeric(df["T_amb_C"], errors="coerce").fillna(25.0)
        elif "T_cell_C_used" in df.columns:
            out["T_amb_C"] = pd.to_numeric(df["T_cell_C_used"], errors="coerce").fillna(25.0)
        elif "T_cell_C" in df.columns:
            out["T_amb_C"] = pd.to_numeric(df["T_cell_C"], errors="coerce").fillna(25.0)
        else:
            out["T_amb_C"] = 25.0

        out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return out, source.name

    # Deterministic fallback market input, 96 x 15-minute steps.
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2030-01-01", periods=96, freq="15min")
    price = 80.0 + 35.0 * np.sin(np.linspace(-np.pi, np.pi, len(timestamps))) + 10.0 * rng.normal(size=len(timestamps))
    return pd.DataFrame({"timestamp": timestamps, "price_eur_per_MWh": price, "T_amb_C": 25.0}), "synthetic_generated_market"


def load_calibrated_lut(input_dir: Path) -> Tuple[pd.DataFrame, str]:
    source = _find_first_existing(
        input_dir,
        [
            "degradation_lut_pybamm_calibrated.csv",
            "degradation_lut_market_realistic.csv",
            "degradation_lut_market_realistic_dense_dod.csv",
            "degradation_lut_fixed.csv",
            "degradation_lut.csv",
        ],
    )
    if source is None:
        raise FileNotFoundError(
            "No degradation LUT found. Run offline_pybamm_benchmark_calibrator.py first "
            "or place degradation_lut_pybamm_calibrated.csv in the input folder."
        )
    return pd.read_csv(source), source.name


def _select_cost_column(lut: pd.DataFrame) -> str:
    for col in [
        "deg_cost_final",
        "deg_cost_eur_per_MWh_throughput",
        "deg_cost_surface_full",
        "deg_cost_smoothed",
    ]:
        if col in lut.columns:
            return col
    raise ValueError("LUT must contain one of: deg_cost_final, deg_cost_eur_per_MWh_throughput, deg_cost_surface_full, deg_cost_smoothed")


# ---------------------------------------------------------------------------
# LUT -> engine piecewise-linear cost points
# ---------------------------------------------------------------------------

def _nearest_temperature_slice(lut: pd.DataFrame, target_temperature_c: float) -> Tuple[pd.DataFrame, float]:
    if "temperature_c" not in lut.columns:
        out = lut.copy()
        out["temperature_c"] = target_temperature_c
        return out, target_temperature_c

    temps = np.array(sorted(pd.to_numeric(lut["temperature_c"], errors="coerce").dropna().unique()), dtype=float)
    if temps.size == 0:
        out = lut.copy()
        out["temperature_c"] = target_temperature_c
        return out, target_temperature_c

    chosen = float(temps[np.argmin(np.abs(temps - target_temperature_c))])
    return lut[np.isclose(pd.to_numeric(lut["temperature_c"], errors="coerce"), chosen)].copy(), chosen


def build_engine_degradation_points(
    lut: pd.DataFrame,
    cfg: EngineMatchedConfig,
    use_dense_lut_points: bool = False,
) -> pd.DataFrame:
    """
    Convert calibrated LUT rows into the exact arrays expected by engine.py.

    LUT convention:
        deg_cost_final or deg_cost_surface_full = EUR/MWh throughput
        dod = fraction of nominal battery energy

    Engine convention:
        energy_points[k] = discharged MWh in one timestep
        cost_points[k]   = EUR degradation cost for that timestep

    Therefore:
        energy_MWh = dod * E_nom_MWh
        cost_EUR   = deg_cost_EUR_per_MWh * energy_MWh
    """
    required = {"dod"}
    missing = required - set(lut.columns)
    if missing:
        raise ValueError(f"LUT is missing required columns: {sorted(missing)}")

    cost_col = _select_cost_column(lut)
    temp_slice, actual_temp = _nearest_temperature_slice(lut, cfg.engine_temperature_c)

    temp_slice["dod"] = pd.to_numeric(temp_slice["dod"], errors="coerce")
    temp_slice[cost_col] = pd.to_numeric(temp_slice[cost_col], errors="coerce")

    valid = temp_slice.dropna(subset=["dod", cost_col]).copy()
    if "valid" in valid.columns:
        valid = valid[pd.to_numeric(valid["valid"], errors="coerce").fillna(0) > 0]

    valid = valid[(valid["dod"] >= 0) & (valid["dod"] <= cfg.max_dod_per_step + 1e-12)]
    valid = valid[valid[cost_col] >= 0]
    valid = valid.sort_values("dod")

    if valid.empty:
        raise ValueError("No valid LUT rows remain after selecting temperature and DoD/C-rate limits.")

    valid["energy_mwh"] = valid["dod"] * cfg.E_nom_MWh
    valid["deg_cost_eur_per_MWh_engine"] = valid[cost_col] * cfg.degradation_cost_scale
    valid["cost_eur"] = valid["deg_cost_eur_per_MWh_engine"] * valid["energy_mwh"]

    # Include exact origin so the engine can choose zero discharge at zero degradation.
    origin = pd.DataFrame(
        {
            "dod": [0.0],
            "temperature_c": [actual_temp],
            "energy_mwh": [0.0],
            "cost_eur": [0.0],
            cost_col: [0.0],
            "deg_cost_eur_per_MWh_engine": [0.0],
        }
    )

    if use_dense_lut_points:
        points = pd.concat([origin, valid[["dod", "temperature_c", "energy_mwh", "cost_eur", cost_col, "deg_cost_eur_per_MWh_engine"]]], ignore_index=True)
        points = points.drop_duplicates(subset=["energy_mwh"]).sort_values("energy_mwh")
    else:
        # Match the current engine style: few breakpoints, by default 6.
        target_energy = np.linspace(0.0, cfg.max_energy_per_step_MWh, cfg.n_engine_breakpoints)
        dense_energy = pd.concat([origin, valid], ignore_index=True).drop_duplicates(subset=["energy_mwh"]).sort_values("energy_mwh")

        interp_cost = np.interp(target_energy, dense_energy["energy_mwh"].to_numpy(), dense_energy["cost_eur"].to_numpy())
        interp_dod = target_energy / cfg.E_nom_MWh
        interp_cost_per_mwh = np.divide(interp_cost, target_energy, out=np.zeros_like(interp_cost), where=target_energy > 1e-12)
        interp_original_cost_per_mwh = np.interp(target_energy, dense_energy["energy_mwh"].to_numpy(), dense_energy[cost_col].to_numpy())

        points = pd.DataFrame(
            {
                "dod": interp_dod,
                "temperature_c": actual_temp,
                "energy_mwh": target_energy,
                "cost_eur": interp_cost,
                cost_col: interp_original_cost_per_mwh,
                "deg_cost_eur_per_MWh_engine": interp_cost_per_mwh,
            }
        )

    # Convex MILP cost interpolation needs nondecreasing cost and preferably nondecreasing slopes.
    points = points.sort_values("energy_mwh").reset_index(drop=True)
    points["cost_eur"] = np.maximum.accumulate(points["cost_eur"].to_numpy(dtype=float))

    if cfg.export_normalized_cost:
        max_cost = max(float(points["cost_eur"].max()), 1e-12)
        max_energy = max(float(points["energy_mwh"].max()), 1e-12)
        points["cost_normalized"] = points["cost_eur"] / max_cost
        points["energy_normalized"] = points["energy_mwh"] / max_energy

    points["degradation_cost_scale"] = cfg.degradation_cost_scale
    points["source_cost_column"] = cost_col
    points["engine_temperature_c_requested"] = cfg.engine_temperature_c
    points["engine_temperature_c_used"] = actual_temp
    points["max_energy_per_step_MWh"] = cfg.max_energy_per_step_MWh
    return points


# ---------------------------------------------------------------------------
# Engine-matched MILP solve
# ---------------------------------------------------------------------------

def _default_solver(msg: bool = False):
    import pulp as pl

    # Match the current engine style but be robust to PuLP versions.
    try:
        homebrew_solver = pl.COIN_CMD(path="/opt/homebrew/bin/cbc", msg=msg)
        if homebrew_solver.available():
            return homebrew_solver
    except Exception:
        pass

    try:
        if hasattr(pl, "PULP_CBC_CMD"):
            bundled_solver = pl.PULP_CBC_CMD(msg=msg)
            if bundled_solver.available():
                return bundled_solver
    except Exception:
        pass

    available_solvers = pl.listSolvers(onlyAvailable=True)
    if available_solvers:
        return pl.getSolver(available_solvers[0], msg=msg)

    raise RuntimeError("No MILP solver is available. Install CBC or run `pip install pulp[cbc]` if supported by your PuLP version.")


def solve_engine_formulation(
    prices: Sequence[float],
    cfg: EngineMatchedConfig,
    energy_points: Sequence[float],
    cost_points: Sequence[float],
    solver_msg: bool = False,
) -> Tuple[object, Dict[int, object], Dict[int, object], Dict[int, object], Dict[int, object]]:
    """Same mathematical structure as engine.py, kept standalone."""
    try:
        import pulp as pl
    except Exception as exc:
        raise RuntimeError("PuLP is not installed. Install it with `pip install pulp` to solve the MILP.") from exc

    prices = [float(x) for x in prices]
    if not prices:
        raise ValueError("prices must contain at least one timestep")

    if len(energy_points) != len(cost_points):
        raise ValueError("energy_points and cost_points must have equal length")
    if len(energy_points) < 2:
        raise ValueError("At least two degradation breakpoints are required")

    TIME_STEPS = range(len(prices))
    K = range(len(energy_points))
    params = cfg.as_engine_params()

    prob = pl.LpProblem("BESS_Optimization_Engine_Matched", pl.LpMaximize)

    p_buy = pl.LpVariable.dicts("P_buy", TIME_STEPS, lowBound=0, upBound=params["p_max"], cat=pl.LpContinuous)
    p_sell = pl.LpVariable.dicts("P_sell", TIME_STEPS, lowBound=0, upBound=params["p_max"], cat=pl.LpContinuous)
    soc = pl.LpVariable.dicts(
        "SoC",
        TIME_STEPS,
        lowBound=params["soc_min"] * params["e_max"],
        upBound=params["soc_max"] * params["e_max"],
        cat=pl.LpContinuous,
    )
    u = pl.LpVariable.dicts("u", TIME_STEPS, cat=pl.LpBinary)

    lambdas = pl.LpVariable.dicts("Lambda", (TIME_STEPS, K), lowBound=0, upBound=1, cat=pl.LpContinuous)
    dynamic_deg_cost = pl.LpVariable.dicts("Degradation_Cost", TIME_STEPS, lowBound=0, cat=pl.LpContinuous)

    for t in TIME_STEPS:
        prob += p_buy[t] <= params["p_max"] * u[t], f"Max_Charge_{t}"
        prob += p_sell[t] <= params["p_max"] * (1 - u[t]), f"Max_Discharge_{t}"

        prev_soc = params["soc_init"] * params["e_max"] if t == 0 else soc[t - 1]
        prob += (
            soc[t]
            == prev_soc + (p_buy[t] * params["eta_ch"] - p_sell[t] / params["eta_dis"]) * params["dt"]
        ), f"SoC_Balance_{t}"

        # This is the same lambda interpolation abstraction as engine.py.
        prob += pl.lpSum(lambdas[t][k] for k in K) == 1, f"Lambda_Sum_{t}"
        prob += p_sell[t] * params["dt"] == pl.lpSum(lambdas[t][k] * float(energy_points[k]) for k in K), f"Energy_Interp_{t}"
        prob += dynamic_deg_cost[t] == pl.lpSum(lambdas[t][k] * float(cost_points[k]) for k in K), f"Cost_Interp_{t}"

    last_step = list(TIME_STEPS)[-1]
    prob += soc[last_step] == params["soc_init"] * params["e_max"], "Cycle_Continuity"

    prob += pl.lpSum(
        prices[t] * (p_sell[t] - p_buy[t]) * params["dt"] - dynamic_deg_cost[t]
        for t in TIME_STEPS
    ), "Total_Profit"

    prob.solve(_default_solver(msg=solver_msg))
    return prob, p_buy, p_sell, soc, dynamic_deg_cost


# ---------------------------------------------------------------------------
# Result extraction and economics
# ---------------------------------------------------------------------------

def _var_value(x) -> float:
    try:
        import pulp as pl
        v = pl.value(x)
    except Exception:
        v = getattr(x, "varValue", np.nan)
    return float(v) if v is not None and np.isfinite(v) else np.nan


def soh_drop_from_degradation_cost(cost_eur: float, cfg: EngineMatchedConfig) -> float:
    return float(cost_eur * (1.0 - cfg.soh_eol) / cfg.replacement_total_cost_eur)


def build_result_tables(
    market: pd.DataFrame,
    cfg: EngineMatchedConfig,
    prob,
    p_buy,
    p_sell,
    soc,
    dynamic_deg_cost,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    soh = 1.0

    for t, row in market.reset_index(drop=True).iterrows():
        P_buy = _var_value(p_buy[t])
        P_sell = _var_value(p_sell[t])
        SoC_MWh = _var_value(soc[t])
        deg_cost = _var_value(dynamic_deg_cost[t])
        price = float(row["price_eur_per_MWh"])

        charge_cost = P_buy * cfg.dt_h * price
        discharge_revenue = P_sell * cfg.dt_h * price
        gross_margin = discharge_revenue - charge_cost
        net_profit = gross_margin - deg_cost
        discharge_energy = P_sell * cfg.dt_h
        charge_energy = P_buy * cfg.dt_h
        dod_discharge = discharge_energy / cfg.E_nom_MWh
        c_rate_charge = P_buy / cfg.E_nom_MWh
        c_rate_discharge = P_sell / cfg.E_nom_MWh
        delta_soh = soh_drop_from_degradation_cost(deg_cost, cfg)
        soh_next = max(0.0, soh - delta_soh)

        rows.append(
            {
                "timestamp": row["timestamp"],
                "price_eur_per_MWh": price,
                "P_buy_MW": P_buy,
                "P_sell_MW": P_sell,
                "SoC_MWh": SoC_MWh,
                "SoC_fraction": SoC_MWh / cfg.E_nom_MWh,
                "charge_energy_MWh": charge_energy,
                "discharge_energy_MWh": discharge_energy,
                "DoD_discharge_step": dod_discharge,
                "C_rate_charge": c_rate_charge,
                "C_rate_discharge": c_rate_discharge,
                "degradation_cost_eur": deg_cost,
                "charge_cost_eur": charge_cost,
                "discharge_revenue_eur": discharge_revenue,
                "gross_margin_eur": gross_margin,
                "net_profit_eur": net_profit,
                "SoH": soh,
                "SoH_next": soh_next,
                "delta_SoH": delta_soh,
                "is_engine_matched": True,
            }
        )
        soh = soh_next

    result = pd.DataFrame(rows)
    dispatch = result[
        [
            "timestamp",
            "price_eur_per_MWh",
            "P_buy_MW",
            "P_sell_MW",
            "SoC_MWh",
            "SoC_fraction",
            "charge_energy_MWh",
            "discharge_energy_MWh",
            "DoD_discharge_step",
            "degradation_cost_eur",
            "net_profit_eur",
        ]
    ].copy()

    status = getattr(prob, "status", None)
    try:
        import pulp as pl
        status_name = pl.LpStatus.get(status, str(status))
        objective_value = float(pl.value(prob.objective))
    except Exception:
        status_name = str(status)
        objective_value = np.nan

    summary = pd.DataFrame(
        [
            {
                "solver_status": status_name,
                "objective_value_eur": objective_value,
                "rows": len(result),
                "E_nom_MWh": cfg.E_nom_MWh,
                "P_max_MW": cfg.P_max_MW,
                "P_max_effective_MW": cfg.p_max_effective_MW,
                "dt_h": cfg.dt_h,
                "effective_c_rate_max": cfg.effective_c_rate_max,
                "max_dod_per_step": cfg.max_dod_per_step,
                "max_energy_per_step_MWh": cfg.max_energy_per_step_MWh,
                "total_charge_MWh": float(result["charge_energy_MWh"].sum()),
                "total_discharge_MWh": float(result["discharge_energy_MWh"].sum()),
                "total_gross_margin_eur": float(result["gross_margin_eur"].sum()),
                "total_degradation_cost_eur": float(result["degradation_cost_eur"].sum()),
                "total_net_profit_eur": float(result["net_profit_eur"].sum()),
                "initial_SoH": float(result["SoH"].iloc[0]),
                "final_SoH": float(result["SoH_next"].iloc[-1]),
                "total_delta_SoH": float(result["delta_SoH"].sum()),
                "max_DoD_discharge_step": float(result["DoD_discharge_step"].max()),
                "max_C_rate_discharge": float(result["C_rate_discharge"].max()),
                "engine_degradation_cost_depends_on": "P_sell_MW * dt_h only",
            }
        ]
    )
    return dispatch, result, summary


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    cfg: EngineMatchedConfig,
    use_dense_lut_points: bool = False,
    skip_solve: bool = False,
    solver_msg: bool = False,
) -> Dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    market, market_source = load_market_input(input_dir)
    lut, lut_source = load_calibrated_lut(input_dir)
    points = build_engine_degradation_points(lut, cfg, use_dense_lut_points=use_dense_lut_points)

    points_path = output_dir / "engine_degradation_points.csv"
    points_json_path = output_dir / "engine_degradation_points.json"
    market_path = output_dir / "bess_market_input_engine_matched.csv"
    manifest_path = output_dir / "bess_engine_matched_manifest.json"

    points.to_csv(points_path, index=False)
    points_json_path.write_text(
        json.dumps(
            {
                "degradation_energy_points": points["energy_mwh"].round(12).tolist(),
                "degradation_cost_points": points["cost_eur"].round(12).tolist(),
                "units": {
                    "degradation_energy_points": "MWh discharged per timestep",
                    "degradation_cost_points": "EUR degradation cost per timestep",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    market.to_csv(market_path, index=False)

    outputs = {
        "engine_degradation_points_csv": str(points_path),
        "engine_degradation_points_json": str(points_json_path),
        "market": str(market_path),
    }

    solve_status = "skipped_by_user"
    summary_payload: Dict[str, object] = {}

    if not skip_solve:
        try:
            prob, p_buy, p_sell, soc, deg = solve_engine_formulation(
                prices=market["price_eur_per_MWh"].tolist(),
                cfg=cfg,
                energy_points=points["energy_mwh"].tolist(),
                cost_points=points["cost_eur"].tolist(),
                solver_msg=solver_msg,
            )
            dispatch, result, summary = build_result_tables(market, cfg, prob, p_buy, p_sell, soc, deg)

            dispatch_path = output_dir / "bess_dispatch_engine_matched.csv"
            result_path = output_dir / "bess_financial_result_engine_matched.csv"
            summary_path = output_dir / "bess_financial_summary_engine_matched.csv"
            dispatch.to_csv(dispatch_path, index=False)
            result.to_csv(result_path, index=False)
            summary.to_csv(summary_path, index=False)
            outputs.update(
                {
                    "dispatch": str(dispatch_path),
                    "result": str(result_path),
                    "summary": str(summary_path),
                }
            )
            solve_status = str(summary.loc[0, "solver_status"])
            summary_payload = summary.iloc[0].to_dict()
        except Exception as exc:
            solve_status = f"solve_failed_or_unavailable: {exc}"

    manifest = {
        "pipeline": "bess_pipeline_engine_matched",
        "purpose": "Convert PyBaMM-calibrated LUT into the same piecewise-linear degradation points used by the optimization engine.",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "market_source": market_source,
        "lut_source": lut_source,
        "config": asdict(cfg),
        "engine_params": cfg.as_engine_params(),
        "degradation_point_count": int(len(points)),
        "degradation_energy_points": points["energy_mwh"].round(12).tolist(),
        "degradation_cost_points": points["cost_eur"].round(12).tolist(),
        "solve_status": solve_status,
        "outputs": outputs,
        "summary": summary_payload,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    outputs["manifest"] = str(manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BESS pipeline matched to the PuLP optimization engine.")
    p.add_argument("--input-dir", default=".")
    p.add_argument("--output-dir", default=".")

    p.add_argument("--E-nom-MWh", type=float, default=2.0)
    p.add_argument("--P-max-MW", type=float, default=1.0)
    p.add_argument("--dt-h", type=float, default=0.25)
    p.add_argument("--c-rate-max", type=float, default=0.50)
    p.add_argument("--soc-min", type=float, default=0.10)
    p.add_argument("--soc-max", type=float, default=0.90)
    p.add_argument("--soc-init", type=float, default=0.50)
    p.add_argument("--eta-charge", type=float, default=0.92)
    p.add_argument("--eta-discharge", type=float, default=0.92)
    p.add_argument("--replacement-cost", type=float, default=30_000.0)
    p.add_argument("--engine-temperature-c", type=float, default=25.0)
    p.add_argument("--n-engine-breakpoints", type=int, default=6)
    p.add_argument("--degradation-cost-scale", type=float, default=1.0, help="Multiplier applied when converting LUT EUR/MWh into engine EUR/timestep cost points.")

    p.add_argument("--use-dense-lut-points", action="store_true", help="Use all valid LUT points instead of the engine-style 6 breakpoints.")
    p.add_argument("--skip-solve", action="store_true", help="Only export engine degradation points; do not solve the MILP.")
    p.add_argument("--solver-msg", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EngineMatchedConfig(
        E_nom_MWh=args.E_nom_MWh,
        P_max_MW=args.P_max_MW,
        dt_h=args.dt_h,
        c_rate_max=args.c_rate_max,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        soc_init=args.soc_init,
        eta_charge=args.eta_charge,
        eta_discharge=args.eta_discharge,
        replacement_cost_eur_per_MWh_capacity=args.replacement_cost,
        engine_temperature_c=args.engine_temperature_c,
        n_engine_breakpoints=args.n_engine_breakpoints,
        degradation_cost_scale=args.degradation_cost_scale,
    )

    manifest = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        cfg=cfg,
        use_dense_lut_points=args.use_dense_lut_points,
        skip_solve=args.skip_solve,
        solver_msg=args.solver_msg,
    )

    print("\nBESS engine-matched pipeline completed.")
    print(f"Market source: {manifest['market_source']}")
    print(f"LUT source: {manifest['lut_source']}")
    print(f"Effective C-rate max: {cfg.effective_c_rate_max:.4g}C")
    print(f"Max DoD per step: {cfg.max_dod_per_step:.4g}")
    print(f"Max energy per step: {cfg.max_energy_per_step_MWh:.4g} MWh")
    print("Engine degradation energy points MWh:")
    print(manifest["degradation_energy_points"])
    print("Engine degradation cost points EUR:")
    print(manifest["degradation_cost_points"])
    print(f"Solve status: {manifest['solve_status']}")
    print("\nFiles written:")
    for k, v in manifest["outputs"].items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
