#!/usr/bin/env python3
"""
PyBaMM-only DoD -> degradation-cost LUT builder.

This script intentionally does not use an empirical fallback curve. It runs a
PyBaMM rest-only calendar baseline and PyBaMM representative 15-minute market-step cycling
points at fixed 25C, subtracts the calendar baseline from cycling degradation,
converts the incremental cycling SoH loss to EUR/MWh, and exports a small
optimizer-ready LUT.

The optimizer-facing output keeps the existing schema:
    energy,temperature_c,deg_cost_eur_per_MWh_throughput

Here `energy` is the one-timestep discharged energy in MWh:
    energy = DoD * E_nom_MWh

Run:
    python pybamm_only_dod_degradation_lut.py

If PyBaMM is missing:
    python pybamm_only_dod_degradation_lut.py --install-note
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BESS_SYSTEM_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_STATS_OUTPUT_DIR = SCRIPT_DIR / "training_statistics"
DEFAULT_OPTIMIZER_OUTPUT_DIR = BESS_SYSTEM_ROOT / "optimization" / "data" / "cleaned_data"


class PyBaMMDegradationError(RuntimeError):
    """Raised when PyBaMM does not provide a usable degradation signal."""


@dataclass
class PyBaMMOnlyConfig:
    # Base BESS module solved by the DAM optimizer.
    E_nom_MWh: float = 2.0
    P_max_MW: float = 1.0
    dt_h: float = 0.25
    c_rate_max: float = 0.50

    # Degradation-to-cost conversion.
    fixed_temperature_c: float = 25.0
    soh_eol: float = 0.80
    replacement_cost_eur_per_MWh_capacity: float = 120_000.0

    # DAM optimizer breakpoints. For 2MWh/1MW/15min, DoD <= 0.125 is feasible.
    dod_values: Tuple[float, ...] = (0.05, 0.08, 0.10, 0.125)

    # PyBaMM representative 15-minute market-step run settings.
    pybamm_cycles_per_point: int = 12
    pybamm_rest_minutes: float = 3.0
    pybamm_parameter_set: str = "Chen2020"
    pybamm_model: str = "SPMe"
    initial_soc: float = 0.50

    # Numeric guard for rejecting zero/noisy degradation signals.
    min_soh_drop_fraction: float = 1e-12

    @property
    def replacement_total_cost_eur(self) -> float:
        return self.replacement_cost_eur_per_MWh_capacity * self.E_nom_MWh

    @property
    def physical_c_rate_from_power(self) -> float:
        return self.P_max_MW / self.E_nom_MWh

    @property
    def effective_c_rate_max(self) -> float:
        return min(self.c_rate_max, self.physical_c_rate_from_power)

    @property
    def max_dod_per_step(self) -> float:
        return self.effective_c_rate_max * self.dt_h

    @property
    def experiment_elapsed_minutes(self) -> float:
        one_cycle_minutes = 2.0 * self.dt_h * 60.0 + max(self.pybamm_rest_minutes, 0.0)
        return self.pybamm_cycles_per_point * one_cycle_minutes


def parse_float_list(value: str) -> Tuple[float, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("DoD grid must contain at least one value.")
    values = tuple(float(p) for p in parts)
    if any(v <= 0 for v in values):
        raise ValueError("All DoD values must be positive.")
    return tuple(sorted(values))


def _build_pybamm_model(cfg: PyBaMMOnlyConfig):
    import pybamm

    option_sets = [
        {"thermal": "lumped", "SEI": "reaction limited"},
        {"thermal": "lumped", "SEI": "solvent-diffusion limited"},
        {"thermal": "lumped", "sei": "reaction limited"},
        {"thermal": "lumped", "sei": "solvent-diffusion limited"},
        {"SEI": "reaction limited"},
        {"SEI": "solvent-diffusion limited"},
        {"sei": "reaction limited"},
        {"sei": "solvent-diffusion limited"},
    ]

    last_error = None
    for options in option_sets:
        try:
            model_name = cfg.pybamm_model.upper()
            if model_name == "DFN":
                return pybamm.lithium_ion.DFN(options=options), options
            if model_name == "SPM":
                return pybamm.lithium_ion.SPM(options=options), options
            return pybamm.lithium_ion.SPMe(options=options), options
        except Exception as exc:
            last_error = exc

    raise PyBaMMDegradationError(f"Could not build PyBaMM degradation model: {last_error}")


def _solution_array(solution, variable_names: Iterable[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for name in variable_names:
        try:
            arr = np.asarray(solution[name].entries, dtype=float)
            if arr.size > 1 and np.all(np.isfinite(arr)):
                return name, arr
        except Exception:
            pass
    return None, None


def _parameter_float(params, names: Iterable[str]) -> Optional[float]:
    for name in names:
        try:
            value = float(params[name])
            if np.isfinite(value) and value > 0:
                return value
        except Exception:
            pass
    return None


def extract_soh_drop_fraction(solution, params, cfg: PyBaMMOnlyConfig) -> Tuple[float, str, Dict[str, float]]:
    """
    Extract an absolute SoH-drop fraction from PyBaMM variables.

    This function is intentionally strict. If PyBaMM does not expose a real
    degradation signal, the caller gets an error instead of a fallback curve.
    """
    candidates: List[Tuple[str, float]] = []
    diagnostics: Dict[str, float] = {}

    name, lli = _solution_array(
        solution,
        [
            "Loss of lithium inventory [%]",
            "Loss of lithium inventory",
        ],
    )
    if lli is not None:
        change = max(0.0, float(np.nanmax(lli) - np.nanmin(lli)))
        if "[%]" in name or np.nanmax(np.abs(lli)) > 1.0:
            change /= 100.0
        diagnostics[f"{name}__delta_fraction"] = change
        candidates.append((name, change))

    nominal_capacity_Ah = _parameter_float(
        params,
        [
            "Nominal cell capacity [A.h]",
            "Cell capacity [A.h]",
        ],
    )
    name, capacity_loss = _solution_array(
        solution,
        [
            "Loss of capacity to SEI [A.h]",
            "Loss of capacity to negative SEI [A.h]",
            "Loss of capacity to positive SEI [A.h]",
            "Total capacity lost to side reactions [A.h]",
            "Loss of capacity [A.h]",
        ],
    )
    if capacity_loss is not None and nominal_capacity_Ah is not None:
        change_Ah = max(0.0, float(np.nanmax(capacity_loss) - np.nanmin(capacity_loss)))
        change_fraction = change_Ah / nominal_capacity_Ah
        diagnostics[f"{name}__delta_Ah"] = change_Ah
        diagnostics[f"{name}__delta_fraction"] = change_fraction
        candidates.append((name, change_fraction))

    positive = [(source, value) for source, value in candidates if value > cfg.min_soh_drop_fraction]
    if not positive:
        available = sorted(getattr(solution, "all_variable_names", []) or [])
        sample = ", ".join(available[:25])
        raise PyBaMMDegradationError(
            "PyBaMM did not expose a positive degradation signal. "
            f"Checked LLI/capacity-loss variables. Available variable sample: {sample}"
        )

    source, soh_drop_fraction = max(positive, key=lambda item: item[1])
    return float(soh_drop_fraction), source, diagnostics


def _run_pybamm_steps(steps: List[str], cfg: PyBaMMOnlyConfig):
    import pybamm

    started = time.perf_counter()
    model, model_options = _build_pybamm_model(cfg)
    params = pybamm.ParameterValues(cfg.pybamm_parameter_set)
    try:
        params.update({"Ambient temperature [K]": cfg.fixed_temperature_c + 273.15}, check_already_exists=False)
    except Exception:
        pass

    experiment = pybamm.Experiment(steps)
    simulation = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
    try:
        solution = simulation.solve(initial_soc=cfg.initial_soc)
    except TypeError:
        solution = simulation.solve()

    return solution, params, model_options, time.perf_counter() - started


def run_calendar_baseline(cfg: PyBaMMOnlyConfig) -> Dict[str, object]:
    """
    Run a rest-only PyBaMM baseline for the same elapsed time as each cycling point.
    """
    steps = [f"Rest for {cfg.experiment_elapsed_minutes:.8g} minutes"]
    solution, params, model_options, runtime_seconds = _run_pybamm_steps(steps, cfg)
    soh_drop_fraction, signal_source, signal_diagnostics = extract_soh_drop_fraction(solution, params, cfg)

    return {
        "calendar_ok": 1,
        "calendar_elapsed_minutes": cfg.experiment_elapsed_minutes,
        "calendar_soh_drop_fraction": soh_drop_fraction,
        "calendar_signal_source": signal_source,
        "calendar_model": cfg.pybamm_model,
        "calendar_model_options": json.dumps(model_options, sort_keys=True),
        "calendar_parameter_set": cfg.pybamm_parameter_set,
        "calendar_error": "",
        "calendar_runtime_seconds": runtime_seconds,
        **{f"calendar__{k}": v for k, v in signal_diagnostics.items()},
    }


def calendar_baseline_error_row(cfg: PyBaMMOnlyConfig, exc: Exception) -> Dict[str, object]:
    return {
        "calendar_ok": 0,
        "calendar_elapsed_minutes": cfg.experiment_elapsed_minutes,
        "calendar_soh_drop_fraction": np.nan,
        "calendar_signal_source": "",
        "calendar_model": cfg.pybamm_model,
        "calendar_model_options": "",
        "calendar_parameter_set": cfg.pybamm_parameter_set,
        "calendar_error": str(exc),
        "calendar_runtime_seconds": np.nan,
    }


def dod_to_degradation_cost_per_mwh(
    dod: float,
    cfg: PyBaMMOnlyConfig,
    calendar_baseline: Dict[str, object],
) -> Dict[str, object]:
    """
    Run one PyBaMM DoD point and convert incremental cycling SoH loss to EUR/MWh.
    """
    dod = float(dod)
    if dod <= 0:
        raise ValueError("DoD must be positive.")

    c_rate = dod / cfg.dt_h
    if c_rate > cfg.effective_c_rate_max + 1e-12:
        raise ValueError(
            f"DoD={dod:g} implies {c_rate:g}C, above the effective "
            f"{cfg.effective_c_rate_max:g}C limit."
        )

    if int(calendar_baseline.get("calendar_ok", 0)) != 1:
        raise PyBaMMDegradationError(f"Calendar baseline failed: {calendar_baseline.get('calendar_error')}")

    minutes = cfg.dt_h * 60.0
    steps = []
    for _ in range(cfg.pybamm_cycles_per_point):
        steps.append(f"Discharge at {c_rate:.8g}C for {minutes:.8g} minutes")
        steps.append(f"Charge at {c_rate:.8g}C for {minutes:.8g} minutes")
        if cfg.pybamm_rest_minutes > 0:
            steps.append(f"Rest for {cfg.pybamm_rest_minutes:.8g} minutes")

    solution, params, model_options, runtime_seconds = _run_pybamm_steps(steps, cfg)
    gross_soh_drop_fraction, signal_source, signal_diagnostics = extract_soh_drop_fraction(solution, params, cfg)

    calendar_soh_drop_fraction = float(calendar_baseline["calendar_soh_drop_fraction"])
    incremental_soh_drop_fraction = gross_soh_drop_fraction - calendar_soh_drop_fraction
    total_discharged_MWh = dod * cfg.E_nom_MWh * cfg.pybamm_cycles_per_point
    eol_fraction_consumed = incremental_soh_drop_fraction / (1.0 - cfg.soh_eol)
    degradation_cost_eur = eol_fraction_consumed * cfg.replacement_total_cost_eur
    cost_per_mwh_discharged = degradation_cost_eur / total_discharged_MWh

    return {
        "dod": dod,
        "energy_MWh": dod * cfg.E_nom_MWh,
        "temperature_c": cfg.fixed_temperature_c,
        "dt_h": cfg.dt_h,
        "c_rate": c_rate,
        "cycles": cfg.pybamm_cycles_per_point,
        "elapsed_minutes": cfg.experiment_elapsed_minutes,
        "total_discharged_MWh": total_discharged_MWh,
        "gross_soh_drop_fraction": gross_soh_drop_fraction,
        "calendar_soh_drop_fraction": calendar_soh_drop_fraction,
        "soh_drop_fraction": incremental_soh_drop_fraction,
        "incremental_cycle_soh_drop_fraction": incremental_soh_drop_fraction,
        "eol_fraction_consumed": eol_fraction_consumed,
        "replacement_total_cost_eur": cfg.replacement_total_cost_eur,
        "degradation_cost_eur": degradation_cost_eur,
        "deg_cost_eur_per_MWh_discharged": cost_per_mwh_discharged,
        "deg_cost_eur_per_MWh_throughput": cost_per_mwh_discharged,
        "pybamm_ok": 1,
        "pybamm_signal_source": signal_source,
        "pybamm_model": cfg.pybamm_model,
        "pybamm_model_options": json.dumps(model_options, sort_keys=True),
        "pybamm_parameter_set": cfg.pybamm_parameter_set,
        "pybamm_error": "",
        "runtime_seconds": runtime_seconds,
        **signal_diagnostics,
    }


def build_pybamm_only_points(cfg: PyBaMMOnlyConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    try:
        calendar_baseline = run_calendar_baseline(cfg)
    except Exception as exc:
        calendar_baseline = calendar_baseline_error_row(cfg, exc)

    for dod in cfg.dod_values:
        try:
            rows.append(dod_to_degradation_cost_per_mwh(dod, cfg, calendar_baseline))
        except Exception as exc:
            rows.append(
                {
                    "dod": float(dod),
                    "energy_MWh": float(dod) * cfg.E_nom_MWh,
                    "temperature_c": cfg.fixed_temperature_c,
                    "dt_h": cfg.dt_h,
                    "c_rate": float(dod) / cfg.dt_h,
                    "cycles": cfg.pybamm_cycles_per_point,
                    "elapsed_minutes": cfg.experiment_elapsed_minutes,
                    "total_discharged_MWh": float(dod) * cfg.E_nom_MWh * cfg.pybamm_cycles_per_point,
                    "gross_soh_drop_fraction": np.nan,
                    "calendar_soh_drop_fraction": calendar_baseline.get("calendar_soh_drop_fraction", np.nan),
                    "soh_drop_fraction": np.nan,
                    "incremental_cycle_soh_drop_fraction": np.nan,
                    "eol_fraction_consumed": np.nan,
                    "replacement_total_cost_eur": cfg.replacement_total_cost_eur,
                    "degradation_cost_eur": np.nan,
                    "deg_cost_eur_per_MWh_discharged": np.nan,
                    "deg_cost_eur_per_MWh_throughput": np.nan,
                    "pybamm_ok": 0,
                    "pybamm_signal_source": "",
                    "pybamm_model": cfg.pybamm_model,
                    "pybamm_model_options": "",
                    "pybamm_parameter_set": cfg.pybamm_parameter_set,
                    "pybamm_error": str(exc),
                    "runtime_seconds": np.nan,
                }
            )

    return (
        pd.DataFrame(rows).sort_values("dod").reset_index(drop=True),
        pd.DataFrame([calendar_baseline]),
    )


def build_optimizer_lut(points: pd.DataFrame) -> pd.DataFrame:
    if points.empty or not (points["pybamm_ok"] == 1).all():
        raise PyBaMMDegradationError("Cannot build optimizer LUT unless every PyBaMM point succeeded.")

    required = [
        "energy_MWh",
        "temperature_c",
        "deg_cost_eur_per_MWh_throughput",
    ]
    missing = [c for c in required if c not in points.columns]
    if missing:
        raise PyBaMMDegradationError(f"PyBaMM points missing columns required for optimizer LUT: {missing}")

    out = points[required].rename(columns={"energy_MWh": "energy"}).copy()
    return out.sort_values("energy").reset_index(drop=True)


def evaluate_report_checks(
    points: pd.DataFrame,
    calendar_baseline: pd.DataFrame,
    cfg: PyBaMMOnlyConfig,
) -> Dict[str, Dict[str, object]]:
    ok = points["pybamm_ok"].astype(int) == 1
    costs = pd.to_numeric(points["deg_cost_eur_per_MWh_throughput"], errors="coerce")
    soh_drop = pd.to_numeric(points["soh_drop_fraction"], errors="coerce")
    gross_soh_drop = pd.to_numeric(points["gross_soh_drop_fraction"], errors="coerce")
    calendar_soh_drop = pd.to_numeric(points["calendar_soh_drop_fraction"], errors="coerce")
    dod = pd.to_numeric(points["dod"], errors="coerce")
    temps = pd.to_numeric(points["temperature_c"], errors="coerce")
    calendar_ok = (
        not calendar_baseline.empty
        and int(calendar_baseline.iloc[0].get("calendar_ok", 0)) == 1
    )

    checks = {
        "calendar_baseline_succeeded": {
            "passed": bool(calendar_ok),
            "value": (
                float(calendar_baseline.iloc[0]["calendar_soh_drop_fraction"])
                if calendar_ok
                else calendar_baseline.iloc[0].get("calendar_error", "missing baseline")
            ),
        },
        "all_pybamm_points_succeeded": {
            "passed": bool(ok.all()),
            "value": f"{int(ok.sum())}/{len(points)}",
        },
        "fixed_temperature_25C": {
            "passed": bool(np.allclose(temps, cfg.fixed_temperature_c, equal_nan=False)),
            "value": float(cfg.fixed_temperature_c),
        },
        "dod_within_physical_dispatch_limit": {
            "passed": bool((dod <= cfg.max_dod_per_step + 1e-12).all()),
            "value": f"max DoD={float(dod.max()):.6g}, limit={cfg.max_dod_per_step:.6g}",
        },
        "gross_cycling_soh_exceeds_calendar_baseline": {
            "passed": bool((gross_soh_drop[ok] > calendar_soh_drop[ok]).all()) if ok.any() else False,
            "value": {
                "gross_min": float(gross_soh_drop[ok].min()) if ok.any() else np.nan,
                "calendar": float(calendar_soh_drop[ok].iloc[0]) if ok.any() else np.nan,
            },
        },
        "positive_incremental_cycle_soh_drop": {
            "passed": bool((soh_drop[ok] > cfg.min_soh_drop_fraction).all()) if ok.any() else False,
            "value": float(soh_drop[ok].min()) if ok.any() else np.nan,
        },
        "positive_cost_per_mwh": {
            "passed": bool((costs[ok] > 0).all()) if ok.any() else False,
            "value": float(costs[ok].min()) if ok.any() else np.nan,
        },
        "cost_per_mwh_monotonic_in_dod": {
            "passed": bool((np.diff(costs[ok].to_numpy(dtype=float)) >= -1e-9).all()) if ok.sum() >= 2 else False,
            "value": costs[ok].round(8).tolist(),
        },
    }
    return checks


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"

    def fmt(value: object) -> str:
        if isinstance(value, float):
            if np.isnan(value):
                return ""
            return f"{value:.8g}"
        if pd.isna(value):
            return ""
        text = str(value).replace("\n", " ").replace("|", "\\|")
        return text

    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def write_report(
    report_path: Path,
    points: pd.DataFrame,
    calendar_baseline: pd.DataFrame,
    checks: Dict[str, Dict[str, object]],
    cfg: PyBaMMOnlyConfig,
    outputs: Dict[str, Optional[Path]],
) -> None:
    lines = [
        "# PyBaMM-Only DoD Degradation LUT Report",
        "",
        "## Method",
        "- No empirical fallback curve is used.",
        "- PyBaMM runs a rest-only calendar baseline for the same elapsed time as each cycling point.",
        "- PyBaMM runs representative 15-minute market-step discharge/charge cycles at fixed 25C.",
        "- Calendar baseline SoH loss is subtracted from gross cycling SoH loss.",
        "- Incremental cycling SoH loss is converted to replacement-cost consumption.",
        "- The optimizer LUT uses one-timestep discharged energy: `energy = DoD * E_nom_MWh`.",
        "",
        "Cost conversion:",
        "",
        "```text",
        "incremental_cycle_soh_drop = gross_cycling_soh_drop - calendar_baseline_soh_drop",
        "eol_fraction_consumed = incremental_cycle_soh_drop / (1 - soh_eol)",
        "degradation_cost_eur = eol_fraction_consumed * replacement_total_cost_eur",
        "EUR_per_MWh = degradation_cost_eur / total_discharged_MWh",
        "```",
        "",
        "## Configuration",
    ]
    for key, value in asdict(cfg).items():
        lines.append(f"- {key}: {value}")

    baseline_cols = [
        "calendar_ok",
        "calendar_elapsed_minutes",
        "calendar_soh_drop_fraction",
        "calendar_signal_source",
        "calendar_error",
        "calendar_runtime_seconds",
    ]
    existing_baseline_cols = [c for c in baseline_cols if c in calendar_baseline.columns]
    lines.extend(["", "## Calendar Baseline", "", _markdown_table(calendar_baseline[existing_baseline_cols])])

    lines.extend(["", "## Checks", "", "| Check | Status | Value |", "|---|---:|---|"])
    for name, result in checks.items():
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"| `{name}` | {status} | {result['value']} |")

    display_cols = [
        "dod",
        "energy_MWh",
        "c_rate",
        "gross_soh_drop_fraction",
        "calendar_soh_drop_fraction",
        "incremental_cycle_soh_drop_fraction",
        "soh_drop_fraction",
        "degradation_cost_eur",
        "deg_cost_eur_per_MWh_throughput",
        "pybamm_ok",
        "pybamm_signal_source",
        "pybamm_error",
    ]
    existing = [c for c in display_cols if c in points.columns]
    lines.extend(["", "## Points", "", _markdown_table(points[existing])])

    lines.extend(["", "## Outputs"])
    for name, path in outputs.items():
        lines.append(f"- {name}: `{path}`" if path else f"- {name}: not written")

    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    stats_output_dir: Path,
    optimizer_output_dir: Path,
    cfg: PyBaMMOnlyConfig,
) -> Dict[str, object]:
    stats_output_dir = stats_output_dir.resolve()
    optimizer_output_dir = optimizer_output_dir.resolve()
    stats_output_dir.mkdir(parents=True, exist_ok=True)
    optimizer_output_dir.mkdir(parents=True, exist_ok=True)

    points, calendar_baseline = build_pybamm_only_points(cfg)
    checks = evaluate_report_checks(points, calendar_baseline, cfg)
    all_checks_passed = all(bool(item["passed"]) for item in checks.values())

    calendar_path = stats_output_dir / "pybamm_only_calendar_baseline.csv"
    points_path = stats_output_dir / "pybamm_only_dod_points.csv"
    full_lut_path = stats_output_dir / "pybamm_only_dod_degradation_lut.csv"
    optimizer_lut_path = optimizer_output_dir / "Reduced_LUT_PyBaMM_Only.csv"
    report_path = stats_output_dir / "pybamm_only_dod_report.md"
    manifest_path = stats_output_dir / "pybamm_only_dod_manifest.json"

    calendar_baseline.to_csv(calendar_path, index=False)
    points.to_csv(points_path, index=False)

    outputs: Dict[str, Optional[Path]] = {
        "calendar_baseline": calendar_path,
        "points": points_path,
        "full_lut": None,
        "optimizer_lut": None,
        "report": report_path,
        "manifest": manifest_path,
    }

    if (points["pybamm_ok"].astype(int) == 1).all():
        full_cols = [
            "dod",
            "energy_MWh",
            "temperature_c",
            "dt_h",
            "c_rate",
            "cycles",
            "elapsed_minutes",
            "total_discharged_MWh",
            "gross_soh_drop_fraction",
            "calendar_soh_drop_fraction",
            "incremental_cycle_soh_drop_fraction",
            "soh_drop_fraction",
            "eol_fraction_consumed",
            "degradation_cost_eur",
            "deg_cost_eur_per_MWh_discharged",
            "deg_cost_eur_per_MWh_throughput",
            "pybamm_signal_source",
            "pybamm_model",
            "pybamm_model_options",
            "pybamm_parameter_set",
            "runtime_seconds",
        ]
        points[[c for c in full_cols if c in points.columns]].to_csv(full_lut_path, index=False)
        outputs["full_lut"] = full_lut_path

    if all_checks_passed:
        build_optimizer_lut(points).to_csv(optimizer_lut_path, index=False)
        outputs["optimizer_lut"] = optimizer_lut_path

    write_report(report_path, points, calendar_baseline, checks, cfg, outputs)

    manifest = {
        "pipeline": "pybamm_only_dod_degradation_lut",
        "strict_pybamm_only": True,
        "all_checks_passed": all_checks_passed,
        "stats_output_dir": str(stats_output_dir),
        "optimizer_output_dir": str(optimizer_output_dir),
        "config": asdict(cfg),
        "calendar_baseline": calendar_baseline.iloc[0].to_dict() if not calendar_baseline.empty else None,
        "checks": checks,
        "outputs": {k: str(v) if v is not None else None for k, v in outputs.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PyBaMM-only DoD degradation-cost LUT.")
    parser.add_argument(
        "--output-dir",
        help="Legacy alias for --stats-output-dir. Optimizer LUT still goes to --optimizer-output-dir.",
    )
    parser.add_argument(
        "--stats-output-dir",
        help=f"Folder for PyBaMM diagnostics. Default: {DEFAULT_STATS_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--optimizer-output-dir",
        help=f"Folder for Reduced_LUT_PyBaMM_Only.csv. Default: {DEFAULT_OPTIMIZER_OUTPUT_DIR}",
    )
    parser.add_argument("--E-nom-MWh", type=float, default=2.0)
    parser.add_argument("--P-max-MW", type=float, default=1.0)
    parser.add_argument("--dt-h", type=float, default=0.25)
    parser.add_argument("--c-rate-max", type=float, default=0.50)
    parser.add_argument("--dod-grid", default="0.05,0.08,0.10,0.125")
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--rest-minutes", type=float, default=3.0)
    parser.add_argument("--model", default="SPMe", choices=["SPM", "SPMe", "DFN"])
    parser.add_argument("--parameter-set", default="Chen2020")
    parser.add_argument("--install-note", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.install_note:
        print("Install PyBaMM in your active Python environment with:")
        print("  pip install 'pybamm[plot]'")
        print("This PyBaMM-only script does not use a fallback curve.")
        return

    stats_output_dir = Path(args.stats_output_dir or args.output_dir or DEFAULT_STATS_OUTPUT_DIR)
    optimizer_output_dir = Path(args.optimizer_output_dir or DEFAULT_OPTIMIZER_OUTPUT_DIR)

    cfg = PyBaMMOnlyConfig(
        E_nom_MWh=args.E_nom_MWh,
        P_max_MW=args.P_max_MW,
        dt_h=args.dt_h,
        c_rate_max=args.c_rate_max,
        dod_values=parse_float_list(args.dod_grid),
        pybamm_cycles_per_point=args.cycles,
        pybamm_rest_minutes=args.rest_minutes,
        pybamm_model=args.model,
        pybamm_parameter_set=args.parameter_set,
    )

    manifest = run_pipeline(stats_output_dir, optimizer_output_dir, cfg)
    print("\nPyBaMM-only DoD degradation LUT run completed.")
    print(f"All checks passed: {manifest['all_checks_passed']}")
    print("Outputs:")
    for name, path in manifest["outputs"].items():
        print(f" - {name}: {path}")

    if not manifest["all_checks_passed"]:
        raise SystemExit(
            "PyBaMM-only checks failed. Inspect pybamm_only_dod_report.md and "
            "pybamm_only_dod_points.csv before using any LUT."
        )


if __name__ == "__main__":
    main()
