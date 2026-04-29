"""Battery digital twin physical layer using PyBaMM (SPM + thermal + degradation)."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pybamm
import matplotlib.pyplot as plt


@dataclass
class SimulationConfig:
    """Simulation settings for the physical layer."""

    capacity_ah: float = 100.0
    voltage_nominal: float = 48.0
    ambient_temp_c: float = 25.0
    profile_minutes: float = 15.0
    replace_cost_eur_per_mwh: float = 60000.0
    initial_internal_resistance_ohm: float = 0.01
    sei_resistance_coeff: float = 5.0e4
    soh_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)


def _build_model() -> pybamm.BaseModel:
    """Create an SPM with thermal coupling and degradation physics."""
    options = {
        "thermal": "lumped",
        "sei": "reaction limited",
        "lithium plating": "reversible",
    }
    return pybamm.lithium_ion.SPM(options=options)


def _build_parameter_values(ambient_temp_c: float) -> pybamm.ParameterValues:
    """Initialize parameter values and set ambient temperature."""
    params = pybamm.ParameterValues("Chen2020")
    params.update({"Ambient temperature [K]": ambient_temp_c + 273.15})
    return params


def _current_for_dod(capacity_ah: float, dod: float, profile_minutes: float) -> float:
    """Compute a constant current to reach DoD in the profile time."""
    hours = profile_minutes / 60.0
    if hours <= 0:
        raise ValueError("profile_minutes must be positive")
    return (dod * capacity_ah) / hours


def _energy_throughput_mwh(current_a: float, voltage_v: float, minutes: float) -> float:
    """Compute energy throughput in MWh for a constant current profile."""
    hours = minutes / 60.0
    power_kw = (current_a * voltage_v) / 1000.0
    return power_kw * hours / 1000.0


def _get_variable(solution: pybamm.Solution, names: List[str]) -> Optional[np.ndarray]:
    """Return the first available variable series from a list of candidates."""
    for name in names:
        if name in solution:
            return solution[name].entries
    return None


def extract_degradation_metrics(solution: pybamm.Solution) -> Dict[str, np.ndarray]:
    """Extract degradation-related variables from the PyBaMM solution.

    Variables used (candidates):
    - LLI: "Loss of lithium inventory [%]" or mol-based variants
    - LAM (anode): "Loss of active material in negative electrode [%]"
    - LAM (cathode): "Loss of active material in positive electrode [%]"
    - SEI thickness: "X-averaged SEI thickness [m]"
    - SEI film resistance: "SEI film resistance [Ohm]"
    """
    lli = _get_variable(
        solution,
        [
            "Loss of lithium inventory [%]",
            "Loss of lithium inventory [mol]",
            "Loss of lithium inventory [mol.m-3]",
        ],
    )

    lam_anode = _get_variable(
        solution,
        [
            "Loss of active material in negative electrode [%]",
            "Loss of active material in negative electrode [mol]",
        ],
    )

    lam_cathode = _get_variable(
        solution,
        [
            "Loss of active material in positive electrode [%]",
            "Loss of active material in positive electrode [mol]",
        ],
    )

    sei_thickness = _get_variable(
        solution,
        [
            "X-averaged SEI thickness [m]",
            "SEI thickness [m]",
        ],
    )

    sei_resistance = _get_variable(
        solution,
        [
            "SEI film resistance [Ohm]",
            "SEI resistance [Ohm]",
        ],
    )

    return {
        "lli": lli,
        "lam_anode": lam_anode,
        "lam_cathode": lam_cathode,
        "sei_thickness": sei_thickness,
        "sei_resistance": sei_resistance,
    }


def compute_internal_resistance(
    degradation: Dict[str, np.ndarray],
    cfg: SimulationConfig,
) -> np.ndarray:
    """Compute internal resistance from SEI growth variables.

    Internal_Resistance(t) = R0 + k * SEI_thickness(t),
    with optional direct SEI film resistance if available.
    """
    if degradation.get("sei_resistance") is not None:
        return cfg.initial_internal_resistance_ohm + degradation["sei_resistance"]

    if degradation.get("sei_thickness") is None:
        raise KeyError("SEI thickness or SEI resistance is required to model resistance growth")

    return cfg.initial_internal_resistance_ohm + cfg.sei_resistance_coeff * degradation["sei_thickness"]


def compute_composite_soh(
    capacity_series_ah: np.ndarray,
    degradation: Dict[str, np.ndarray],
    resistance_series_ohm: np.ndarray,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Compute a composite SoH metric.

    SoH(t) = w1 * Capacity_retention + w2 * Resistance_retention + w3 * Lithium_inventory_ratio
    Each term is normalized to [0, 1].
    """
    w1, w2, w3 = cfg.soh_weights

    capacity_retention = capacity_series_ah / capacity_series_ah[0]
    capacity_retention = np.clip(capacity_retention, 0.0, 1.0)

    resistance_retention = resistance_series_ohm[0] / resistance_series_ohm
    resistance_retention = np.clip(resistance_retention, 0.0, 1.0)

    lli = degradation.get("lli")
    if lli is None:
        raise KeyError("LLI variable not found in the solution")

    if np.nanmax(lli) > 1.0:
        lithium_inventory_ratio = 1.0 - (lli / 100.0)
    else:
        lithium_inventory_ratio = 1.0 - lli
    lithium_inventory_ratio = np.clip(lithium_inventory_ratio, 0.0, 1.0)

    return w1 * capacity_retention + w2 * resistance_retention + w3 * lithium_inventory_ratio


def plot_degradation(time_s: np.ndarray, metrics: Dict[str, np.ndarray]):
    """Plot SoH, resistance, and degradation metrics over time."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    if metrics.get("composite_soh") is not None:
        axes[0].plot(time_s, metrics["composite_soh"], label="Composite SoH")
        axes[0].set_ylabel("SoH")
        axes[0].legend()

    if metrics.get("internal_resistance_ohm") is not None:
        axes[1].plot(time_s, metrics["internal_resistance_ohm"], label="Internal Resistance")
        axes[1].set_ylabel("Ohm")
        axes[1].legend()

    if metrics.get("degradation") is not None:
        deg = metrics["degradation"]
        if deg.get("lli") is not None:
            axes[2].plot(time_s, deg["lli"], label="LLI")
        if deg.get("lam_anode") is not None:
            axes[2].plot(time_s, deg["lam_anode"], label="LAM Anode")
        if deg.get("lam_cathode") is not None:
            axes[2].plot(time_s, deg["lam_cathode"], label="LAM Cathode")
        axes[2].set_ylabel("Degradation")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()

    plt.tight_layout()
    return fig


def simulate_profile(
    dod: float,
    temp_c: float,
    cfg: SimulationConfig,
    return_details: bool = False,
) -> Union[Tuple[float, float, float, float], Tuple[float, float, float, float, Dict[str, object]]]:
    """Run a single 15-minute profile and return SoH metrics.

    Returns:
        soh_start, soh_end, soh_drop, energy_mwh
        If return_details=True, also returns a dict with degradation time series and composite SoH.
    """
    model = _build_model()
    params = _build_parameter_values(temp_c)

    current_a = _current_for_dod(cfg.capacity_ah, dod, cfg.profile_minutes)
    profile = pybamm.Experiment([
        f"Discharge at {current_a:.4f} A for {cfg.profile_minutes} minutes",
    ])

    sim = pybamm.Simulation(model, parameter_values=params, experiment=profile)
    solution = sim.solve()

    # SoH estimation based on capacity fade
    capacity = solution["Capacity [A.h]"]
    capacity_entries = capacity.entries
    soh_start = float(capacity_entries[0] / cfg.capacity_ah)
    soh_end = float(capacity_entries[-1] / cfg.capacity_ah)
    soh_drop = max(0.0, soh_start - soh_end)

    # Composite SoH based on degradation physics
    degradation = extract_degradation_metrics(solution)
    resistance = compute_internal_resistance(degradation, cfg)
    composite_soh = compute_composite_soh(capacity_entries, degradation, resistance, cfg)
    composite_drop = max(0.0, float(composite_soh[0] - composite_soh[-1]))

    energy_mwh = _energy_throughput_mwh(current_a, cfg.voltage_nominal, cfg.profile_minutes)
    if return_details:
        return (
            soh_start,
            soh_end,
            soh_drop,
            energy_mwh,
            {
                "degradation": degradation,
                "internal_resistance_ohm": resistance,
                "composite_soh": composite_soh,
                "composite_drop": composite_drop,
                "time_s": solution.t,
            },
        )

    return soh_start, soh_end, soh_drop, energy_mwh


def compute_marginal_degradation_cost(soh_drop: float, energy_mwh: float, cfg: SimulationConfig) -> float:
    """
    Compute marginal degradation cost in EUR/MWh.

    Formula: V_deg = C_EOL * |SoHdot|
    Here we interpret SoHdot per MWh as: SoH_drop / energy_mwh
    
    """
    if energy_mwh <= 0:
        return 0.0
    soh_per_mwh = abs(soh_drop / energy_mwh)
    return cfg.replace_cost_eur_per_mwh * soh_per_mwh


def run_lut_grid(dod_values: Iterable[float], temp_values_c: Iterable[float], cfg: SimulationConfig) -> pd.DataFrame:
    """
    Run simulations for a DoD x Temperature grid and build a LUT.

    """
    rows: List[dict] = []
    for dod in dod_values:
        for temp_c in temp_values_c:
            try:
                soh_start, soh_end, soh_drop, energy_mwh, details = simulate_profile(
                    dod, temp_c, cfg, return_details=True
                )
                v_deg = compute_marginal_degradation_cost(soh_drop, energy_mwh, cfg)
                rows.append({
                    "dod": float(dod),
                    "temperature_c": float(temp_c),
                    "soh_start": soh_start,
                    "soh_end": soh_end,
                    "soh_drop": soh_drop,
                    "composite_soh_drop": details["composite_drop"],
                    "energy_mwh": energy_mwh,
                    "v_deg_eur_per_mwh": v_deg,
                })
            except Exception as exc:
                rows.append({
                    "dod": float(dod),
                    "temperature_c": float(temp_c),
                    "soh_start": np.nan,
                    "soh_end": np.nan,
                    "soh_drop": np.nan,
                    "composite_soh_drop": np.nan,
                    "energy_mwh": np.nan,
                    "v_deg_eur_per_mwh": np.nan,
                    "error": str(exc),
                })

    return pd.DataFrame(rows)
