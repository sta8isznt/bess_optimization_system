"""Shared degradation curve and LUT loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from bess_optimization.models import DegradationCurve
from bess_optimization.optimization.backtest_utils import build_dummy_cost_curve
from bess_optimization.paths import (
    CLEANED_DATA_DIR,
    DEFAULT_DEGRADATION_LUT_PATH,
    DEFAULT_PYBAMM_LUT_PATH,
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


def list_lut_files(cleaned_data_dir: Path = CLEANED_DATA_DIR) -> list[Path]:
    files: list[Path] = []
    for candidate in (DEFAULT_PYBAMM_LUT_PATH, DEFAULT_DEGRADATION_LUT_PATH):
        if candidate.exists() and candidate not in files:
            files.append(candidate)
    if cleaned_data_dir.exists():
        files.extend(
            p
            for p in sorted(cleaned_data_dir.glob("*.csv"))
            if p not in files and ("lut" in p.name.lower() or "degradation" in p.name.lower())
        )
    return files


def default_lut_for_source(
    source: str,
    lut_files: Iterable[Path] | None = None,
) -> Path | None:
    files = list(lut_files or list_lut_files())
    source = source.strip().lower()
    if source == "pybamm":
        for path in files:
            if "pybamm" in path.name.lower():
                return path
        return None
    if source == "lut":
        for path in files:
            if "pybamm" not in path.name.lower():
                return path
        return files[0] if files else None
    return None


def _find_column(df: pd.DataFrame, candidates: Iterable[str], purpose: str) -> str:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    raise ValueError(f"Could not infer {purpose} column. Available columns: {list(df.columns)}")


def load_degradation_lut_curve(
    csv_path: Path = DEFAULT_DEGRADATION_LUT_PATH,
    temperature_c: float = 25.0,
    allow_nearest_temperature: bool = False,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    curve = load_degradation_curve(
        source="lut",
        params={"p_max": 1.0, "dt": 0.25},
        lut_file=csv_path,
        temperature_c=temperature_c,
        allow_nearest_temperature=allow_nearest_temperature,
    )
    return curve.energy_points, curve.cost_points


def _read_lut_curve(
    csv_path: Path,
    temperature_c: float,
    max_interval_energy_mwh: float,
    multiplier: float,
    allow_nearest_temperature: bool,
) -> DegradationCurve:
    path = Path(csv_path)
    if not path.exists():
        raise ValueError(f"Degradation LUT file does not exist: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        raise ValueError(f"Degradation LUT is empty: {path.name}")

    energy_col = _find_column(df, LUT_ENERGY_COLUMNS, "LUT energy")
    cost_col = _find_column(df, LUT_COST_COLUMNS, "LUT degradation cost")
    warnings: list[str] = []

    filtered = df.copy()
    temp_label = "all temperatures"
    try:
        temp_col = _find_column(df, LUT_TEMP_COLUMNS, "LUT temperature")
        filtered[temp_col] = pd.to_numeric(filtered[temp_col], errors="coerce")
        available = filtered[temp_col].dropna().sort_values().unique()
        if len(available) == 0:
            raise ValueError("No valid temperature rows found in LUT.")
        if allow_nearest_temperature:
            nearest = float(available[np.argmin(np.abs(available - float(temperature_c)))])
            if abs(nearest - float(temperature_c)) > 1e-9:
                warnings.append(
                    f"LUT has no exact {temperature_c:g} C row; using nearest {nearest:g} C."
                )
            selected_temperature = nearest
        else:
            matches = available[np.isclose(available, float(temperature_c))]
            if len(matches) == 0:
                raise ValueError(f"LUT has no {temperature_c:g} C row.")
            selected_temperature = float(matches[0])
        filtered = filtered[np.isclose(filtered[temp_col], selected_temperature)].copy()
        temp_label = f"{selected_temperature:g} C"
    except ValueError as exc:
        if "temperature" not in str(exc).lower():
            raise
        if any(str(col).lower() in {item.lower() for item in LUT_TEMP_COLUMNS} for col in df.columns):
            raise
        warnings.append("LUT has no temperature column; using all rows.")

    filtered[energy_col] = pd.to_numeric(filtered[energy_col], errors="coerce")
    filtered[cost_col] = pd.to_numeric(filtered[cost_col], errors="coerce")
    filtered = filtered.dropna(subset=[energy_col, cost_col])
    filtered = filtered[filtered[energy_col] > 0].sort_values(energy_col)
    if filtered.empty:
        raise ValueError(f"No valid energy/cost rows found in {path.name}.")

    energy_points = filtered[energy_col].astype(float).tolist()
    cost_values = filtered[cost_col].astype(float)
    if "cost_eur_per" in cost_col.lower() or cost_col.lower().startswith("deg_cost"):
        cost_points = (filtered[energy_col].astype(float) * cost_values).tolist()
    else:
        cost_points = cost_values.tolist()

    multiplier = float(multiplier)
    if multiplier < 0:
        raise ValueError("Degradation cost multiplier cannot be negative.")
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
    return DegradationCurve(
        energy_points=tuple(energy_points),
        cost_points=tuple(cost_points),
        source_label=label,
        warnings=tuple(warnings),
    )


def load_degradation_curve(
    source: str,
    params: dict,
    lut_file: Path | None = None,
    temperature_c: float = 25.0,
    multiplier: float = 1.0,
    allow_nearest_temperature: bool = False,
) -> DegradationCurve:
    source = source.strip().lower()
    max_interval_energy = float(params["p_max"]) * float(params.get("dt", params.get("DT", 0.25)))

    if source == "dummy":
        energy, cost = build_dummy_cost_curve(params)
        cost = [float(value) * float(multiplier) for value in cost]
        return DegradationCurve(tuple(energy), tuple(cost), "synthetic dummy degradation curve")

    if source == "zero":
        energy = np.linspace(0.0, max_interval_energy, 5).tolist()
        return DegradationCurve(
            tuple(energy),
            tuple(0.0 for _ in energy),
            "zero degradation cost - comparison only",
        )

    if source not in {"lut", "pybamm"}:
        raise ValueError('Degradation source must be "lut", "pybamm", "dummy", or "zero".')

    selected_lut = Path(lut_file) if lut_file else default_lut_for_source(source)
    if selected_lut is None:
        raise ValueError(f"No LUT file is available for degradation source: {source}")
    return _read_lut_curve(
        selected_lut,
        temperature_c,
        max_interval_energy,
        multiplier,
        allow_nearest_temperature=allow_nearest_temperature,
    )
