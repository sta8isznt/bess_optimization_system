"""Battery digital twin physical layer using PyBaMM (SPM + thermal + degradation)."""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import pybamm


@dataclass
class SimulationConfig:
    """Simulation settings for the physical layer."""

    capacity_ah: float = 100.0
    voltage_nominal: float = 48.0
    ambient_temp_c: float = 25.0
    profile_minutes: float = 15.0
    replace_cost_eur_per_mwh: float = 60000.0


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


def simulate_profile(dod: float, temp_c: float, cfg: SimulationConfig) -> Tuple[float, float, float, float]:
    """Run a single 15-minute profile and return SoH metrics.

    Returns:
        soh_start, soh_end, soh_drop, energy_mwh
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
    soh_start = float(capacity.entries[0] / cfg.capacity_ah)
    soh_end = float(capacity.entries[-1] / cfg.capacity_ah)
    soh_drop = max(0.0, soh_start - soh_end)

    energy_mwh = _energy_throughput_mwh(current_a, cfg.voltage_nominal, cfg.profile_minutes)
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
                soh_start, soh_end, soh_drop, energy_mwh = simulate_profile(dod, temp_c, cfg)
                v_deg = compute_marginal_degradation_cost(soh_drop, energy_mwh, cfg)
                rows.append({
                    "dod": float(dod),
                    "temperature_c": float(temp_c),
                    "soh_start": soh_start,
                    "soh_end": soh_end,
                    "soh_drop": soh_drop,
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
                    "energy_mwh": np.nan,
                    "v_deg_eur_per_mwh": np.nan,
                    "error": str(exc),
                })

    return pd.DataFrame(rows)
