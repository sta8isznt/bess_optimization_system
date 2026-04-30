#!/usr/bin/env python3
"""
Build a PyBaMM-only DoD degradation-cost LUT.

The pipeline runs a rest-only calendar baseline and representative 15-minute
market-step cycling points at fixed cell temperature. Calendar SoH loss is
subtracted from gross cycling SoH loss, and the incremental cycling loss is
converted to EUR/MWh for the optimizer.

The optimizer-facing output keeps the existing schema:
    energy,temperature_c,deg_cost_eur_per_MWh_throughput

In this schema, `energy` is one-timestep discharged energy in MWh:
    energy = DoD * nominal_energy_mwh

The cost column name is retained for downstream compatibility. Its value is
computed per MWh discharged.

Run:
    python pybamm_only_dod_degradation_lut.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BESS_SYSTEM_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DIAGNOSTICS_OUTPUT_DIR = SCRIPT_DIR / "training_statistics"
DEFAULT_OPTIMIZER_OUTPUT_DIR = BESS_SYSTEM_ROOT / "optimization" / "data" / "cleaned_data"


class PyBaMMDegradationError(RuntimeError):
    """Raised when PyBaMM cannot produce a valid degradation-cost LUT."""


@dataclass
class PyBaMMLutConfig:
    # Base BESS module solved by the DAM optimizer.
    nominal_energy_mwh: float = 2.0
    max_power_mw: float = 1.0
    timestep_hours: float = 0.25
    max_c_rate: float = 0.50

    # Degradation-to-cost conversion.
    cell_temperature_c: float = 25.0
    end_of_life_soh_fraction: float = 0.80
    replacement_cost_eur_per_mwh_capacity: float = 60_000.0

    # DAM optimizer breakpoints. For 2 MWh / 1 MW / 15 min, DoD <= 0.125 is feasible.
    dod_breakpoints: Tuple[float, ...] = (0.05, 0.08, 0.10, 0.125)

    # PyBaMM representative 15-minute market-step run settings.
    cycles_per_dod_point: int = 12
    rest_minutes_between_cycles: float = 3.0
    pybamm_parameter_set: str = "Chen2020"
    pybamm_model_name: str = "SPMe"
    initial_soc_fraction: float = 0.50

    # Numeric guard for rejecting zero/noisy degradation signals.
    min_valid_soh_loss_fraction: float = 1e-12

    @property
    def pack_replacement_cost_eur(self) -> float:
        return self.replacement_cost_eur_per_mwh_capacity * self.nominal_energy_mwh

    @property
    def power_limited_c_rate(self) -> float:
        return self.max_power_mw / self.nominal_energy_mwh

    @property
    def dispatch_c_rate_limit(self) -> float:
        return min(self.max_c_rate, self.power_limited_c_rate)

    @property
    def max_dispatch_dod_per_step(self) -> float:
        return self.dispatch_c_rate_limit * self.timestep_hours

    @property
    def simulated_elapsed_minutes_per_point(self) -> float:
        cycle_minutes = 2.0 * self.timestep_hours * 60.0 + max(self.rest_minutes_between_cycles, 0.0)
        return self.cycles_per_dod_point * cycle_minutes


def parse_dod_breakpoints(raw_value: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())
    if not values:
        raise ValueError("DoD breakpoints must contain at least one value.")
    if any((not np.isfinite(value)) or value <= 0 for value in values):
        raise ValueError("All DoD breakpoints must be finite and positive.")
    return tuple(sorted(values))


def validate_config(config: PyBaMMLutConfig) -> None:
    numeric_fields = {
        "nominal_energy_mwh": config.nominal_energy_mwh,
        "max_power_mw": config.max_power_mw,
        "timestep_hours": config.timestep_hours,
        "max_c_rate": config.max_c_rate,
        "cell_temperature_c": config.cell_temperature_c,
        "end_of_life_soh_fraction": config.end_of_life_soh_fraction,
        "replacement_cost_eur_per_mwh_capacity": config.replacement_cost_eur_per_mwh_capacity,
        "rest_minutes_between_cycles": config.rest_minutes_between_cycles,
        "initial_soc_fraction": config.initial_soc_fraction,
        "min_valid_soh_loss_fraction": config.min_valid_soh_loss_fraction,
    }
    non_finite = [name for name, value in numeric_fields.items() if not np.isfinite(value)]
    if non_finite:
        raise ValueError(f"Configuration values must be finite: {non_finite}")

    positive_fields = {
        "nominal_energy_mwh": config.nominal_energy_mwh,
        "max_power_mw": config.max_power_mw,
        "timestep_hours": config.timestep_hours,
        "max_c_rate": config.max_c_rate,
        "replacement_cost_eur_per_mwh_capacity": config.replacement_cost_eur_per_mwh_capacity,
        "min_valid_soh_loss_fraction": config.min_valid_soh_loss_fraction,
    }
    non_positive = [name for name, value in positive_fields.items() if value <= 0]
    if non_positive:
        raise ValueError(f"Configuration values must be positive: {non_positive}")

    if config.cycles_per_dod_point <= 0:
        raise ValueError("cycles_per_dod_point must be positive.")
    if config.rest_minutes_between_cycles < 0:
        raise ValueError("rest_minutes_between_cycles cannot be negative.")
    if not 0 < config.end_of_life_soh_fraction < 1:
        raise ValueError("end_of_life_soh_fraction must be between 0 and 1.")
    if not 0 <= config.initial_soc_fraction <= 1:
        raise ValueError("initial_soc_fraction must be between 0 and 1.")
    if config.pybamm_model_name.upper() not in {"SPM", "SPME", "DFN"}:
        raise ValueError("pybamm_model_name must be one of: SPM, SPMe, DFN.")

    if not config.dod_breakpoints:
        raise ValueError("dod_breakpoints must contain at least one value.")
    invalid_dod = [
        value
        for value in config.dod_breakpoints
        if (not np.isfinite(value)) or value <= 0
    ]
    if invalid_dod:
        raise ValueError(f"DoD breakpoints must be finite and positive: {invalid_dod}")

    max_dod = max(config.dod_breakpoints)
    if max_dod > config.max_dispatch_dod_per_step + 1e-12:
        raise ValueError(
            f"Max DoD breakpoint {max_dod:g} exceeds the one-step dispatch limit "
            f"{config.max_dispatch_dod_per_step:g}."
        )


def _build_pybamm_model(config: PyBaMMLutConfig):
    import pybamm

    degradation_option_sets = [
        {"thermal": "lumped", "SEI": "reaction limited"},
        {"thermal": "lumped", "SEI": "solvent-diffusion limited"},
        {"thermal": "lumped", "sei": "reaction limited"},
        {"thermal": "lumped", "sei": "solvent-diffusion limited"},
        {"SEI": "reaction limited"},
        {"SEI": "solvent-diffusion limited"},
        {"sei": "reaction limited"},
        {"sei": "solvent-diffusion limited"},
    ]

    model_key = config.pybamm_model_name.upper()
    last_error = None
    for model_options in degradation_option_sets:
        try:
            if model_key == "DFN":
                return pybamm.lithium_ion.DFN(options=model_options), model_options
            if model_key == "SPM":
                return pybamm.lithium_ion.SPM(options=model_options), model_options
            return pybamm.lithium_ion.SPMe(options=model_options), model_options
        except Exception as exc:
            last_error = exc

    raise PyBaMMDegradationError(f"Could not build PyBaMM degradation model: {last_error}")


def _extract_solution_array(
    solution,
    variable_names: Iterable[str],
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for variable_name in variable_names:
        try:
            values = np.asarray(solution[variable_name].entries, dtype=float)
            if values.size > 1 and np.all(np.isfinite(values)):
                return variable_name, values
        except Exception:
            pass
    return None, None


def _parameter_value_as_float(params, names: Iterable[str]) -> Optional[float]:
    for parameter_name in names:
        try:
            value = float(params[parameter_name])
            if np.isfinite(value) and value > 0:
                return value
        except Exception:
            pass
    return None


def extract_soh_loss_fraction(
    solution,
    params,
    config: PyBaMMLutConfig,
) -> Tuple[float, str, Dict[str, float]]:
    """
    Extract an absolute SoH-loss fraction from PyBaMM degradation variables.

    If PyBaMM does not expose a usable degradation signal, the caller gets an
    error instead of an empirical fallback.
    """
    signal_candidates: List[Tuple[str, float]] = []
    signal_diagnostics: Dict[str, float] = {}

    variable_name, lithium_inventory_loss = _extract_solution_array(
        solution,
        [
            "Loss of lithium inventory [%]",
            "Loss of lithium inventory",
        ],
    )
    if lithium_inventory_loss is not None:
        loss_fraction = max(0.0, float(np.nanmax(lithium_inventory_loss) - np.nanmin(lithium_inventory_loss)))
        if "[%]" in variable_name or np.nanmax(np.abs(lithium_inventory_loss)) > 1.0:
            loss_fraction /= 100.0
        signal_diagnostics[f"{variable_name}__delta_fraction"] = loss_fraction
        signal_candidates.append((variable_name, loss_fraction))

    nominal_capacity_ah = _parameter_value_as_float(
        params,
        [
            "Nominal cell capacity [A.h]",
            "Cell capacity [A.h]",
        ],
    )
    variable_name, capacity_loss_ah = _extract_solution_array(
        solution,
        [
            "Loss of capacity to SEI [A.h]",
            "Loss of capacity to negative SEI [A.h]",
            "Loss of capacity to positive SEI [A.h]",
            "Total capacity lost to side reactions [A.h]",
            "Loss of capacity [A.h]",
        ],
    )
    if capacity_loss_ah is not None and nominal_capacity_ah is not None:
        loss_ah = max(0.0, float(np.nanmax(capacity_loss_ah) - np.nanmin(capacity_loss_ah)))
        loss_fraction = loss_ah / nominal_capacity_ah
        signal_diagnostics[f"{variable_name}__delta_Ah"] = loss_ah
        signal_diagnostics[f"{variable_name}__delta_fraction"] = loss_fraction
        signal_candidates.append((variable_name, loss_fraction))

    valid_candidates = [
        (source, value)
        for source, value in signal_candidates
        if value > config.min_valid_soh_loss_fraction
    ]
    if not valid_candidates:
        available_variables = sorted(getattr(solution, "all_variable_names", []) or [])
        variable_sample = ", ".join(available_variables[:25])
        raise PyBaMMDegradationError(
            "PyBaMM did not expose a positive degradation signal. "
            f"Checked LLI/capacity-loss variables. Available variable sample: {variable_sample}"
        )

    signal_source, soh_loss_fraction = max(valid_candidates, key=lambda item: item[1])
    return float(soh_loss_fraction), signal_source, signal_diagnostics


def run_pybamm_experiment_steps(steps: Sequence[str], config: PyBaMMLutConfig):
    import pybamm

    started_at = time.perf_counter()
    model, model_options = _build_pybamm_model(config)
    parameter_values = pybamm.ParameterValues(config.pybamm_parameter_set)
    try:
        parameter_values.update(
            {"Ambient temperature [K]": config.cell_temperature_c + 273.15},
            check_already_exists=False,
        )
    except Exception as exc:
        raise PyBaMMDegradationError("Could not set PyBaMM ambient temperature.") from exc

    experiment = pybamm.Experiment(list(steps))
    simulation = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
    try:
        solution = simulation.solve(initial_soc=config.initial_soc_fraction)
    except TypeError:
        solution = simulation.solve()

    return solution, parameter_values, model_options, time.perf_counter() - started_at


def run_calendar_baseline(config: PyBaMMLutConfig) -> Dict[str, object]:
    """
    Run a rest-only PyBaMM baseline for the same elapsed time as each cycling point.
    """
    steps = [f"Rest for {config.simulated_elapsed_minutes_per_point:.8g} minutes"]
    solution, params, model_options, runtime_seconds = run_pybamm_experiment_steps(steps, config)
    soh_loss_fraction, signal_source, signal_diagnostics = extract_soh_loss_fraction(solution, params, config)

    return {
        "calendar_simulated_elapsed_minutes": config.simulated_elapsed_minutes_per_point,
        "calendar_soh_loss_fraction": soh_loss_fraction,
        "calendar_degradation_signal_source": signal_source,
        "pybamm_model_name": config.pybamm_model_name,
        "pybamm_model_options": json.dumps(model_options, sort_keys=True),
        "pybamm_parameter_set": config.pybamm_parameter_set,
        "calendar_runtime_seconds": runtime_seconds,
        **{f"calendar__{key}": value for key, value in signal_diagnostics.items()},
    }


def simulate_dod_degradation_cost(
    dod_fraction: float,
    config: PyBaMMLutConfig,
    calendar_baseline: Mapping[str, object],
) -> Dict[str, object]:
    """
    Run one PyBaMM DoD point and convert incremental cycling SoH loss to EUR/MWh.
    """
    dod_fraction = float(dod_fraction)
    if dod_fraction <= 0:
        raise ValueError("DoD must be positive.")

    c_rate = dod_fraction / config.timestep_hours
    if c_rate > config.dispatch_c_rate_limit + 1e-12:
        raise ValueError(
            f"DoD={dod_fraction:g} implies {c_rate:g}C, above the effective "
            f"{config.dispatch_c_rate_limit:g}C limit."
        )

    step_minutes = config.timestep_hours * 60.0
    experiment_steps = []
    for _ in range(config.cycles_per_dod_point):
        experiment_steps.append(f"Discharge at {c_rate:.8g}C for {step_minutes:.8g} minutes")
        experiment_steps.append(f"Charge at {c_rate:.8g}C for {step_minutes:.8g} minutes")
        if config.rest_minutes_between_cycles > 0:
            experiment_steps.append(f"Rest for {config.rest_minutes_between_cycles:.8g} minutes")

    solution, params, model_options, runtime_seconds = run_pybamm_experiment_steps(experiment_steps, config)
    gross_soh_loss_fraction, signal_source, signal_diagnostics = extract_soh_loss_fraction(solution, params, config)

    calendar_soh_loss_fraction = float(calendar_baseline["calendar_soh_loss_fraction"])
    incremental_cycling_soh_loss_fraction = gross_soh_loss_fraction - calendar_soh_loss_fraction
    dispatch_energy_mwh = dod_fraction * config.nominal_energy_mwh
    total_discharged_energy_mwh = dispatch_energy_mwh * config.cycles_per_dod_point
    replacement_life_fraction_consumed = (
        incremental_cycling_soh_loss_fraction / (1.0 - config.end_of_life_soh_fraction)
    )
    degradation_cost_eur = replacement_life_fraction_consumed * config.pack_replacement_cost_eur
    cost_eur_per_mwh_discharged = degradation_cost_eur / total_discharged_energy_mwh

    return {
        "dod_fraction": dod_fraction,
        "dispatch_energy_mwh": dispatch_energy_mwh,
        "cell_temperature_c": config.cell_temperature_c,
        "timestep_hours": config.timestep_hours,
        "c_rate": c_rate,
        "cycles_per_dod_point": config.cycles_per_dod_point,
        "simulated_elapsed_minutes": config.simulated_elapsed_minutes_per_point,
        "total_discharged_energy_mwh": total_discharged_energy_mwh,
        "gross_soh_loss_fraction": gross_soh_loss_fraction,
        "calendar_soh_loss_fraction": calendar_soh_loss_fraction,
        "incremental_cycling_soh_loss_fraction": incremental_cycling_soh_loss_fraction,
        "replacement_life_fraction_consumed": replacement_life_fraction_consumed,
        "pack_replacement_cost_eur": config.pack_replacement_cost_eur,
        "degradation_cost_eur": degradation_cost_eur,
        "cost_eur_per_mwh_discharged": cost_eur_per_mwh_discharged,
        "degradation_signal_source": signal_source,
        "pybamm_model_name": config.pybamm_model_name,
        "pybamm_model_options": json.dumps(model_options, sort_keys=True),
        "pybamm_parameter_set": config.pybamm_parameter_set,
        "simulation_runtime_seconds": runtime_seconds,
        **signal_diagnostics,
    }


def build_pybamm_dod_results(config: PyBaMMLutConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    calendar_baseline = run_calendar_baseline(config)
    result_rows = [
        simulate_dod_degradation_cost(dod_fraction, config, calendar_baseline)
        for dod_fraction in config.dod_breakpoints
    ]

    return (
        pd.DataFrame(result_rows).sort_values("dod_fraction").reset_index(drop=True),
        pd.DataFrame([calendar_baseline]),
    )


def _require_columns(frame: pd.DataFrame, required_columns: Sequence[str], frame_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise PyBaMMDegradationError(f"{frame_name} missing required columns: {missing_columns}")


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def validate_optimizer_lut_inputs(
    dod_results: pd.DataFrame,
    calendar_baseline: pd.DataFrame,
    config: PyBaMMLutConfig,
) -> None:
    if dod_results.empty:
        raise PyBaMMDegradationError("No PyBaMM DoD points were produced.")
    if calendar_baseline.empty:
        raise PyBaMMDegradationError("No PyBaMM calendar baseline was produced.")

    required_result_columns = [
        "dod_fraction",
        "dispatch_energy_mwh",
        "cell_temperature_c",
        "gross_soh_loss_fraction",
        "calendar_soh_loss_fraction",
        "incremental_cycling_soh_loss_fraction",
        "degradation_cost_eur",
        "cost_eur_per_mwh_discharged",
    ]
    _require_columns(dod_results, required_result_columns, "DoD results")
    _require_columns(calendar_baseline, ["calendar_soh_loss_fraction"], "Calendar baseline")

    numeric_columns = required_result_columns + ["c_rate", "total_discharged_energy_mwh"]
    for column in numeric_columns:
        values = _numeric_column(dod_results, column)
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise PyBaMMDegradationError(f"DoD results contain invalid numeric values in '{column}'.")

    calendar_loss = _numeric_column(calendar_baseline, "calendar_soh_loss_fraction")
    if calendar_loss.isna().any() or not np.isfinite(calendar_loss.to_numpy(dtype=float)).all():
        raise PyBaMMDegradationError("Calendar baseline contains an invalid SoH loss value.")

    dod_fraction = _numeric_column(dod_results, "dod_fraction")
    if not ((dod_fraction > 0) & (dod_fraction <= config.max_dispatch_dod_per_step + 1e-12)).all():
        raise PyBaMMDegradationError(
            f"DoD points must be positive and no larger than {config.max_dispatch_dod_per_step:g}."
        )

    gross_soh_loss = _numeric_column(dod_results, "gross_soh_loss_fraction")
    calendar_soh_loss = _numeric_column(dod_results, "calendar_soh_loss_fraction")
    if not (gross_soh_loss > calendar_soh_loss).all():
        raise PyBaMMDegradationError("Gross cycling SoH loss must exceed calendar baseline SoH loss.")

    incremental_soh_loss = _numeric_column(dod_results, "incremental_cycling_soh_loss_fraction")
    if not (incremental_soh_loss > config.min_valid_soh_loss_fraction).all():
        raise PyBaMMDegradationError("Incremental cycling SoH loss is zero, negative, or below noise guard.")

    dispatch_energy = _numeric_column(dod_results, "dispatch_energy_mwh")
    total_discharged_energy = _numeric_column(dod_results, "total_discharged_energy_mwh")
    cost_per_mwh = _numeric_column(dod_results, "cost_eur_per_mwh_discharged")
    if not ((dispatch_energy > 0) & (total_discharged_energy > 0) & (cost_per_mwh > 0)).all():
        raise PyBaMMDegradationError("Dispatch energy and degradation cost per MWh must be positive.")


def build_optimizer_lut(dod_results: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "dispatch_energy_mwh",
        "cell_temperature_c",
        "cost_eur_per_mwh_discharged",
    ]
    _require_columns(dod_results, required_columns, "DoD results")

    optimizer_lut = dod_results[required_columns].rename(
        columns={
            "dispatch_energy_mwh": "energy",
            "cell_temperature_c": "temperature_c",
            "cost_eur_per_mwh_discharged": "deg_cost_eur_per_MWh_throughput",
        }
    )
    return optimizer_lut.sort_values("energy").reset_index(drop=True)


def build_report_diagnostics(dod_results: pd.DataFrame) -> Dict[str, object]:
    cost_per_mwh = _numeric_column(dod_results, "cost_eur_per_mwh_discharged")
    return {
        "cost_eur_per_mwh_non_decreasing": bool(
            (np.diff(cost_per_mwh.to_numpy(dtype=float)) >= -1e-9).all()
        ),
        "cost_eur_per_mwh_values": cost_per_mwh.round(8).tolist(),
    }


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return f"{value:.8g}"
        if pd.isna(value):
            return ""
        return str(value).replace("\n", " ").replace("|", "\\|")

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    dod_results: pd.DataFrame,
    calendar_baseline: pd.DataFrame,
    diagnostics: Mapping[str, object],
    config: PyBaMMLutConfig,
    outputs: Mapping[str, Path],
) -> None:
    lines = [
        "# PyBaMM-Only DoD Degradation LUT Report",
        "",
        "## Method",
        "- No empirical fallback curve is used.",
        "- PyBaMM runs a rest-only calendar baseline for the same elapsed time as each cycling point.",
        f"- PyBaMM runs representative market-step discharge/charge cycles at {config.cell_temperature_c:g}C.",
        "- Calendar baseline SoH loss is subtracted from gross cycling SoH loss.",
        "- Incremental cycling SoH loss is converted to replacement-cost consumption.",
        "- The optimizer LUT uses one-timestep discharged energy: `energy = DoD * nominal_energy_mwh`.",
        "",
        "Cost conversion:",
        "",
        "```text",
        "incremental_cycling_soh_loss = gross_cycling_soh_loss - calendar_baseline_soh_loss",
        "replacement_life_fraction_consumed = incremental_cycling_soh_loss / (1 - end_of_life_soh_fraction)",
        "degradation_cost_eur = replacement_life_fraction_consumed * pack_replacement_cost_eur",
        "EUR_per_MWh_discharged = degradation_cost_eur / total_discharged_energy_mwh",
        "```",
        "",
        "## Configuration",
    ]
    for key, value in asdict(config).items():
        lines.append(f"- {key}: {value}")

    baseline_columns = [
        "calendar_simulated_elapsed_minutes",
        "calendar_soh_loss_fraction",
        "calendar_degradation_signal_source",
        "calendar_runtime_seconds",
    ]
    existing_baseline_columns = [column for column in baseline_columns if column in calendar_baseline.columns]
    lines.extend(["", "## Calendar Baseline", "", _markdown_table(calendar_baseline[existing_baseline_columns])])

    point_columns = [
        "dod_fraction",
        "dispatch_energy_mwh",
        "c_rate",
        "gross_soh_loss_fraction",
        "calendar_soh_loss_fraction",
        "incremental_cycling_soh_loss_fraction",
        "degradation_cost_eur",
        "cost_eur_per_mwh_discharged",
        "degradation_signal_source",
        "simulation_runtime_seconds",
    ]
    existing_point_columns = [column for column in point_columns if column in dod_results.columns]
    lines.extend(["", "## DoD Points", "", _markdown_table(dod_results[existing_point_columns])])

    lines.extend(["", "## Diagnostics", "", "| Metric | Value |", "|---|---|"])
    for key, value in diagnostics.items():
        lines.append(f"| `{key}` | {value} |")

    lines.extend(["", "## Outputs"])
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    diagnostics_output_dir: Path,
    optimizer_output_dir: Path,
    config: PyBaMMLutConfig,
) -> Dict[str, object]:
    validate_config(config)

    diagnostics_output_dir = diagnostics_output_dir.resolve()
    optimizer_output_dir = optimizer_output_dir.resolve()
    diagnostics_output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_output_dir.mkdir(parents=True, exist_ok=True)

    dod_results, calendar_baseline = build_pybamm_dod_results(config)
    validate_optimizer_lut_inputs(dod_results, calendar_baseline, config)

    calendar_path = diagnostics_output_dir / "pybamm_only_calendar_baseline.csv"
    points_path = diagnostics_output_dir / "pybamm_only_dod_points.csv"
    full_lut_path = diagnostics_output_dir / "pybamm_only_dod_degradation_lut.csv"
    optimizer_lut_path = optimizer_output_dir / "Reduced_LUT_PyBaMM_Only.csv"
    report_path = diagnostics_output_dir / "pybamm_only_dod_report.md"
    manifest_path = diagnostics_output_dir / "pybamm_only_dod_manifest.json"

    optimizer_lut = build_optimizer_lut(dod_results)
    diagnostics = build_report_diagnostics(dod_results)

    calendar_baseline.to_csv(calendar_path, index=False)
    dod_results.to_csv(points_path, index=False)
    dod_results.to_csv(full_lut_path, index=False)
    optimizer_lut.to_csv(optimizer_lut_path, index=False)

    outputs = {
        "calendar_baseline": calendar_path,
        "dod_points": points_path,
        "full_lut": full_lut_path,
        "optimizer_lut": optimizer_lut_path,
        "report": report_path,
        "manifest": manifest_path,
    }
    write_report(report_path, dod_results, calendar_baseline, diagnostics, config, outputs)

    manifest = {
        "pipeline": "pybamm_only_dod_degradation_lut",
        "strict_pybamm_only": True,
        "validation_status": "passed",
        "diagnostics_output_dir": str(diagnostics_output_dir),
        "optimizer_output_dir": str(optimizer_output_dir),
        "config": asdict(config),
        "calendar_baseline": calendar_baseline.iloc[0].to_dict(),
        "diagnostics": diagnostics,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PyBaMM-only DoD degradation-cost LUT.")
    parser.add_argument(
        "--diagnostics-output-dir",
        help=f"Folder for PyBaMM diagnostics. Default: {DEFAULT_DIAGNOSTICS_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--optimizer-output-dir",
        help=f"Folder for Reduced_LUT_PyBaMM_Only.csv. Default: {DEFAULT_OPTIMIZER_OUTPUT_DIR}",
    )
    parser.add_argument("--nominal-energy-mwh", type=float, default=2.0)
    parser.add_argument("--max-power-mw", type=float, default=1.0)
    parser.add_argument("--timestep-hours", type=float, default=0.25)
    parser.add_argument("--max-c-rate", type=float, default=0.50)
    parser.add_argument("--cell-temperature-c", type=float, default=25.0)
    parser.add_argument("--end-of-life-soh-fraction", type=float, default=0.80)
    parser.add_argument("--replacement-cost-eur-per-mwh-capacity", type=float, default=60_000.0)
    parser.add_argument("--dod-breakpoints", default="0.05,0.08,0.10,0.125")
    parser.add_argument("--cycles-per-dod-point", type=int, default=12)
    parser.add_argument("--rest-minutes-between-cycles", type=float, default=3.0)
    parser.add_argument("--pybamm-model", default="SPMe", choices=["SPM", "SPMe", "DFN"])
    parser.add_argument("--pybamm-parameter-set", default="Chen2020")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics_output_dir = Path(args.diagnostics_output_dir or DEFAULT_DIAGNOSTICS_OUTPUT_DIR)
    optimizer_output_dir = Path(args.optimizer_output_dir or DEFAULT_OPTIMIZER_OUTPUT_DIR)

    config = PyBaMMLutConfig(
        nominal_energy_mwh=args.nominal_energy_mwh,
        max_power_mw=args.max_power_mw,
        timestep_hours=args.timestep_hours,
        max_c_rate=args.max_c_rate,
        cell_temperature_c=args.cell_temperature_c,
        end_of_life_soh_fraction=args.end_of_life_soh_fraction,
        replacement_cost_eur_per_mwh_capacity=args.replacement_cost_eur_per_mwh_capacity,
        dod_breakpoints=parse_dod_breakpoints(args.dod_breakpoints),
        cycles_per_dod_point=args.cycles_per_dod_point,
        rest_minutes_between_cycles=args.rest_minutes_between_cycles,
        pybamm_model_name=args.pybamm_model,
        pybamm_parameter_set=args.pybamm_parameter_set,
    )

    try:
        manifest = run_pipeline(diagnostics_output_dir, optimizer_output_dir, config)
    except (PyBaMMDegradationError, ValueError) as exc:
        raise SystemExit(f"Failed to build PyBaMM-only LUT: {exc}") from exc

    print("\nPyBaMM-only DoD degradation LUT run completed.")
    print("Validation: passed")
    print("Outputs:")
    for name, path in manifest["outputs"].items():
        print(f" - {name}: {path}")


if __name__ == "__main__":
    main()
