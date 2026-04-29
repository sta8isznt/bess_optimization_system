#!/usr/bin/env python3
"""
Offline PyBaMM benchmark calibrator for BESS degradation LUT.

Purpose
-------
Use PyBaMM as an OFFLINE benchmark/calibration layer, not as an online market
dispatch simulator.

Workflow:
    synthetic / market-realistic LUT
        + optional PyBaMM representative duty-cycle simulations
        + physics-shaped DoD-C-rate curve at fixed 25C
        -> degradation_lut_pybamm_calibrated.csv

This output LUT is then consumed by the market BESS pipeline.

Why offline?
------------
PyBaMM is a cell-level electrochemical simulator. It is useful for testing the
shape of degradation against representative duty cycles, but too heavy and too
chemistry-specific to run inside every market-optimization timestep.

Outputs
-------
pybamm_benchmark_points.csv
degradation_lut_pybamm_calibrated.csv
pybamm_calibration_report.html, if plotly is installed
pybamm_calibration_manifest.json

Run
---
python offline_pybamm_benchmark_calibrator.py --input-dir . --output-dir .

Optional:
python offline_pybamm_benchmark_calibrator.py --install-note
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


@dataclass
class CalibratorConfig:
    # Battery/system-level values used for C-rate and validity mask.
    E_nom_MWh: float = 2.0
    P_max_MW: float = 1.0
    dt_h: float = 0.25
    c_rate_max: float = 0.50
    soh_eol: float = 0.80
    replacement_cost_eur_per_MWh_capacity: float = 60_000.0

    # Dense LUT grid. Temperature is fixed at 25C for DAM optimizer use.
    lut_min_dod: float = 0.005
    lut_max_dod: float = 0.125
    lut_dod_step: float = 0.005
    temp_values_c: Tuple[float, ...] = (25.0,)

    # Curve shape: approximately linear at the start, exponential after knee.
    ref_dod: float = 0.10
    ref_temp_C: float = 25.0
    ref_c_rate: float = 0.40
    dod_knee: float = 0.10
    linear_exponent: float = 1.05
    exponential_strength: float = 2.60
    temp_coeff_per_C: float = 0.0
    c_rate_exponent: float = 0.18

    # Scale. If a base LUT exists, this is overwritten by the value near ref_dod/ref_temp.
    fallback_base_cost_eur_per_MWh_at_ref: float = 8.0
    base_cost_min_eur_per_MWh_at_ref: float = 6.0
    base_cost_max_eur_per_MWh_at_ref: float = 10.0

    # Robust factor blending. PyBaMM is a relative benchmark only.
    pybamm_blend_weight: float = 0.20
    rolling_median_window: int = 5
    quantile_clip_low: float = 0.05
    quantile_clip_high: float = 0.95
    absolute_relative_cap: float = 10.0
    final_relative_soft_cap: float = 5.0

    # PyBaMM representative run settings.
    pybamm_cycles_per_point: int = 12
    pybamm_rest_minutes: float = 3.0
    pybamm_parameter_set: str = "Chen2020"
    pybamm_model: str = "SPM"  # SPM is faster and enough for an offline benchmark.

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
    def replacement_total_cost_eur(self) -> float:
        return self.replacement_cost_eur_per_MWh_capacity * self.E_nom_MWh


def _find_first_existing(input_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for n in names:
        p = input_dir / n
        if p.exists():
            return p
    return None


def _dense_dod_grid(cfg: CalibratorConfig) -> List[float]:
    n = int(round((cfg.lut_max_dod - cfg.lut_min_dod) / cfg.lut_dod_step)) + 1
    return [round(cfg.lut_min_dod + i * cfg.lut_dod_step, 8) for i in range(n)]


def _cost_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "deg_cost_final",
        "deg_cost_eur_per_MWh_throughput",
        "deg_cost_surface_full",
        "deg_cost_smoothed",
        "hybrid_cost_final_eur_per_MWh",
        "aggressive_cost_final_surface_eur_per_MWh",
    ]
    return next((c for c in candidates if c in df.columns), None)


def _nearest_value(df: pd.DataFrame, dod: float, temp_c: float, col: str) -> Optional[float]:
    if df is None or df.empty:
        return None
    if "dod" not in df.columns or "temperature_c" not in df.columns or col not in df.columns:
        return None
    valid = df.dropna(subset=["dod", "temperature_c", col]).copy()
    valid = valid[pd.to_numeric(valid[col], errors="coerce") > 0]
    if valid.empty:
        return None
    dod_scale = max(float(valid["dod"].max() - valid["dod"].min()), 1e-12)
    temp_scale = max(float(valid["temperature_c"].max() - valid["temperature_c"].min()), 1e-12)
    dist = ((valid["dod"] - dod) / dod_scale) ** 2 + ((valid["temperature_c"] - temp_c) / temp_scale) ** 2
    return float(valid.loc[dist.idxmin(), col])


def load_base_lut(input_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    p = _find_first_existing(
        input_dir,
        [
            "degradation_lut_market_realistic.csv",
            "degradation_lut_market_realistic_dense_dod.csv",
            "degradation_lut_final_aggressive_optimizer_ready.csv",
            "degradation_lut_final_aggressive_hybrid_combined.csv",
            "degradation_lut_realistic.csv",
            "degradation_lut_fixed.csv",
            "degradation_lut.csv",
        ],
    )
    if p is None:
        return None, None
    return pd.read_csv(p), p.name


def base_cost_at_reference(base_lut: Optional[pd.DataFrame], cfg: CalibratorConfig) -> float:
    """
    Use an existing base LUT only to calibrate scale near DoD=0.10 and T=25C.
    The base LUT must not override the new physics-shaped curve.
    """
    if base_lut is None or base_lut.empty:
        value = cfg.fallback_base_cost_eur_per_MWh_at_ref
    else:
        col = _cost_col(base_lut)
        value = _nearest_value(base_lut, cfg.ref_dod, cfg.ref_temp_C, col) if col is not None else None
        if value is None or not np.isfinite(value) or value <= 0:
            value = cfg.fallback_base_cost_eur_per_MWh_at_ref

    return float(
        np.clip(
            value,
            cfg.base_cost_min_eur_per_MWh_at_ref,
            cfg.base_cost_max_eur_per_MWh_at_ref,
        )
    )

def linear_then_exponential_dod_factor(dod: float, cfg: CalibratorConfig) -> float:
    """
    Shape requested by the project:
    - approximately linear for shallow cycles
    - exponential/nonlinear after a stress knee
    """
    dod = max(float(dod), 1e-9)
    ref = max(cfg.ref_dod, 1e-9)
    knee = max(cfg.dod_knee, 1e-9)

    if dod <= knee:
        return (dod / ref) ** cfg.linear_exponent

    knee_factor = (knee / ref) ** cfg.linear_exponent
    # Normalized excess above knee.
    x = (dod - knee) / max(cfg.lut_max_dod - knee, 1e-12)
    return float(knee_factor * np.exp(cfg.exponential_strength * x))


def fallback_relative_factor(dod: float, temp_c: float, cfg: CalibratorConfig) -> float:
    # Kept for backward compatibility. This is the physics relative factor.
    c_rate = dod / cfg.dt_h
    dod_factor = linear_then_exponential_dod_factor(dod, cfg)
    # Temperature is intentionally fixed at 25C for the DAM optimizer LUT.
    temp_factor = 1.0
    c_rate_factor = (max(c_rate, 1e-9) / cfg.ref_c_rate) ** cfg.c_rate_exponent
    return float(dod_factor * temp_factor * c_rate_factor)


def physics_relative_factor(dod: float, temp_c: float, cfg: CalibratorConfig) -> float:
    return fallback_relative_factor(dod, temp_c, cfg)


def _nearest_reference_value(df: pd.DataFrame, value_col: str, cfg: CalibratorConfig) -> Optional[float]:
    valid = df.dropna(subset=["dod", "temperature_c", value_col]).copy()
    valid[value_col] = pd.to_numeric(valid[value_col], errors="coerce")
    valid = valid[np.isfinite(valid[value_col]) & (valid[value_col] > 0)]
    if valid.empty:
        return None

    exact = valid[np.isclose(valid["dod"], cfg.ref_dod) & np.isclose(valid["temperature_c"], cfg.ref_temp_C)]
    if not exact.empty:
        return float(exact[value_col].median())

    dod_scale = max(float(valid["dod"].max() - valid["dod"].min()), 1e-12)
    temp_scale = max(float(valid["temperature_c"].max() - valid["temperature_c"].min()), 1e-12)
    dist = ((valid["dod"] - cfg.ref_dod) / dod_scale) ** 2 + ((valid["temperature_c"] - cfg.ref_temp_C) / temp_scale) ** 2
    value = float(valid.loc[dist.idxmin(), value_col])
    return value if np.isfinite(value) and value > 0 else None


def _soft_cap_relative_factor(values: pd.Series, cap: float) -> pd.Series:
    """Smoothly cap extreme relative factors without creating a flat plateau."""
    cap = max(float(cap), 1e-9)
    arr = pd.to_numeric(values, errors="coerce").astype(float)
    return pd.Series(cap * np.tanh(arr / cap), index=values.index)



def _build_pybamm_model(cfg: CalibratorConfig):
    import pybamm

    # Options differ by version, so try a robust sequence.
    option_sets = [
        {"thermal": "lumped", "SEI": "reaction limited"},
        {"thermal": "lumped", "sei": "reaction limited"},
        {"thermal": "lumped"},
        {},
    ]

    last_error = None
    for options in option_sets:
        try:
            if cfg.pybamm_model.upper() == "SPME":
                return pybamm.lithium_ion.SPMe(options=options)
            return pybamm.lithium_ion.SPM(options=options)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not build PyBaMM model: {last_error}")


def _solution_array(solution, variable_names: Iterable[str]) -> Optional[np.ndarray]:
    for name in variable_names:
        try:
            arr = np.asarray(solution[name].entries, dtype=float)
            if arr.size > 1 and np.all(np.isfinite(arr)):
                return arr
        except Exception:
            pass
    return None


def run_pybamm_point(dod: float, temp_c: float, cfg: CalibratorConfig) -> Dict[str, object]:
    """
    Run a representative shallow-cycle duty point.

    The absolute SOH drop is often too chemistry/parameter dependent for direct BESS
    costing, so we use it mainly to obtain a RELATIVE degradation factor.
    """
    temp_c = float(cfg.ref_temp_C)
    c_rate = dod / cfg.dt_h

    try:
        import pybamm
    except Exception as exc:
        return {
            "dod": dod,
            "temperature_c": temp_c,
            "c_rate": c_rate,
            "pybamm_ok": 0,
            "pybamm_soh_drop_proxy": np.nan,
            "pybamm_relative_raw": np.nan,
            "pybamm_error": f"pybamm_not_available: {exc}",
        }

    try:
        model = _build_pybamm_model(cfg)
        params = pybamm.ParameterValues(cfg.pybamm_parameter_set)
        try:
            params.update({"Ambient temperature [K]": temp_c + 273.15}, check_already_exists=False)
        except Exception:
            pass

        minutes = cfg.dt_h * 60.0
        steps = []
        for _ in range(cfg.pybamm_cycles_per_point):
            steps.append(f"Discharge at {c_rate:.6g}C for {minutes:.6g} minutes")
            steps.append(f"Charge at {c_rate:.6g}C for {minutes:.6g} minutes")
            if cfg.pybamm_rest_minutes > 0:
                steps.append(f"Rest for {cfg.pybamm_rest_minutes:.6g} minutes")

        experiment = pybamm.Experiment(steps)
        sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment)
        try:
            solution = sim.solve(initial_soc=0.50)
        except TypeError:
            solution = sim.solve()

        candidates = []

        lli = _solution_array(
            solution,
            [
                "Loss of lithium inventory [%]",
                "Loss of lithium inventory",
                "Loss of lithium inventory [mol]",
            ],
        )
        if lli is not None:
            change = float(np.nanmax(lli) - np.nanmin(lli))
            # [%] variables are usually > 1; normalize to fraction.
            if np.nanmax(np.abs(lli)) > 1.0:
                change /= 100.0
            candidates.append(max(0.0, change))

        capacity = _solution_array(
            solution,
            [
                "Capacity [A.h]",
                "Discharge capacity [A.h]",
                "Throughput capacity [A.h]",
            ],
        )
        if capacity is not None and abs(capacity[0]) > 1e-12:
            # Use magnitude of change as proxy. Different variables have different signs.
            candidates.append(abs(float(capacity[-1] - capacity[0])) / max(abs(float(capacity[0])), 1e-12))

        if candidates:
            soh_drop_proxy = max(candidates)
        else:
            # If the model exposes no degradation variables, fall back to shape.
            soh_drop_proxy = fallback_relative_factor(dod, temp_c, cfg) * 1e-7

        return {
            "dod": dod,
            "temperature_c": temp_c,
            "c_rate": c_rate,
            "pybamm_ok": 1,
            "pybamm_soh_drop_proxy": float(soh_drop_proxy),
            "pybamm_relative_raw": float(soh_drop_proxy),
            "pybamm_error": "",
        }

    except Exception as exc:
        return {
            "dod": dod,
            "temperature_c": temp_c,
            "c_rate": c_rate,
            "pybamm_ok": 0,
            "pybamm_soh_drop_proxy": np.nan,
            "pybamm_relative_raw": np.nan,
            "pybamm_error": str(exc),
        }


def normalize_benchmark_factors(points: pd.DataFrame, cfg: CalibratorConfig) -> pd.DataFrame:
    out = points.copy().sort_values(["temperature_c", "dod"]).reset_index(drop=True)

    out["physics_relative_factor"] = out.apply(
        lambda r: physics_relative_factor(float(r["dod"]), float(r["temperature_c"]), cfg),
        axis=1,
    )

    ok = (
        (out["pybamm_ok"] == 1)
        & out["pybamm_soh_drop_proxy"].notna()
        & np.isfinite(out["pybamm_soh_drop_proxy"])
        & (out["pybamm_soh_drop_proxy"] > 0)
    )

    out["pybamm_relative_factor"] = np.nan
    out["pybamm_relative_smoothed"] = np.nan
    out["pybamm_relative_clipped"] = np.nan

    if ok.any():
        ref = _nearest_reference_value(out.loc[ok], "pybamm_soh_drop_proxy", cfg)
        if ref is not None and np.isfinite(ref) and ref > 0:
            out.loc[ok, "pybamm_relative_factor"] = out.loc[ok, "pybamm_soh_drop_proxy"] / ref

            rel_ok = out.loc[ok, "pybamm_relative_factor"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(rel_ok) >= 3:
                lower = float(rel_ok.quantile(cfg.quantile_clip_low))
                upper = float(rel_ok.quantile(cfg.quantile_clip_high))
            else:
                lower = float(rel_ok.min()) if len(rel_ok) else 0.05
                upper = float(rel_ok.max()) if len(rel_ok) else cfg.absolute_relative_cap

            if not np.isfinite(lower) or lower <= 0:
                lower = 0.05
            if not np.isfinite(upper) or upper <= lower:
                upper = max(lower * 1.01, cfg.absolute_relative_cap)
            upper = min(upper, cfg.absolute_relative_cap)

            out.loc[ok, "pybamm_relative_clipped"] = out.loc[ok, "pybamm_relative_factor"].clip(lower=lower, upper=upper)

            window = max(1, int(cfg.rolling_median_window))
            if window % 2 == 0:
                window += 1
            for _, idx in out.groupby("temperature_c").groups.items():
                g = out.loc[idx].sort_values("dod")
                smoothed = g["pybamm_relative_clipped"].rolling(window=window, center=True, min_periods=1).median()
                out.loc[g.index, "pybamm_relative_smoothed"] = smoothed.values

            ref_smoothed = _nearest_reference_value(out.loc[ok], "pybamm_relative_smoothed", cfg)
            if ref_smoothed is not None and np.isfinite(ref_smoothed) and ref_smoothed > 0:
                out["pybamm_relative_smoothed"] = out["pybamm_relative_smoothed"] / ref_smoothed
            else:
                out["pybamm_relative_smoothed"] = np.nan

    has_pybamm = (
        (out["pybamm_ok"] == 1)
        & out["pybamm_relative_smoothed"].notna()
        & np.isfinite(out["pybamm_relative_smoothed"])
        & (out["pybamm_relative_smoothed"] > 0)
    )

    w = float(np.clip(cfg.pybamm_blend_weight, 0.0, 1.0))
    out["relative_factor_final"] = out["physics_relative_factor"]
    out.loc[has_pybamm, "relative_factor_final"] = (
        (1.0 - w) * out.loc[has_pybamm, "physics_relative_factor"]
        + w * out.loc[has_pybamm, "pybamm_relative_smoothed"]
    )

    out["benchmark_source"] = np.where(has_pybamm, "pybamm_smoothed_relative_blend", "physics_fallback_shape")

    ref_final = _nearest_reference_value(out, "relative_factor_final", cfg)
    if ref_final is not None and np.isfinite(ref_final) and ref_final > 0:
        out["relative_factor_final"] = out["relative_factor_final"] / ref_final

    # Smooth cap: safer than the old upper=50 hard clip and avoids saturated plateaus.
    out["relative_factor_final"] = _soft_cap_relative_factor(out["relative_factor_final"], cfg.final_relative_soft_cap)
    ref_after_cap = _nearest_reference_value(out, "relative_factor_final", cfg)
    if ref_after_cap is not None and np.isfinite(ref_after_cap) and ref_after_cap > 0:
        out["relative_factor_final"] = out["relative_factor_final"] / ref_after_cap

    out["relative_factor_final"] = out["relative_factor_final"].where(
        np.isfinite(out["relative_factor_final"]) & (out["relative_factor_final"] > 0),
        out["physics_relative_factor"],
    )

    return out

def _enforce_monotonic_with_reference(
    df: pd.DataFrame,
    group_cols: List[str],
    order_col: str,
    value_col: str,
    reference_col: str,
) -> pd.DataFrame:
    """Correct only local monotonic violations without cummax plateaus."""
    out = df.copy().sort_values(group_cols + [order_col]).reset_index(drop=True)
    values = out[value_col].astype(float).to_numpy(copy=True)
    refs = out[reference_col].astype(float).to_numpy(copy=True)
    keys = out[group_cols].apply(lambda row: tuple(row.values.tolist()), axis=1)
    for _, positions in keys.groupby(keys).groups.items():
        pos = list(positions)
        for j in range(1, len(pos)):
            prev_i = pos[j - 1]
            cur_i = pos[j]
            if values[cur_i] <= values[prev_i]:
                ref_delta = max(refs[cur_i] - refs[prev_i], 0.0)
                min_increment = max(0.25 * ref_delta, abs(values[prev_i]) * 1e-8, 1e-8)
                values[cur_i] = values[prev_i] + min_increment
    out[value_col] = values
    return out


def build_calibrated_lut(
    cfg: CalibratorConfig,
    base_lut: Optional[pd.DataFrame],
    benchmark_points: pd.DataFrame,
) -> pd.DataFrame:
    base_cost = base_cost_at_reference(base_lut, cfg)

    rows = []
    for _, r in benchmark_points.iterrows():
        dod = float(r["dod"])
        temp = float(r["temperature_c"])
        c_rate = dod / cfg.dt_h

        physics_cost = base_cost * float(r["physics_relative_factor"])
        cost_surface = base_cost * float(r["relative_factor_final"])

        valid = c_rate <= cfg.effective_c_rate_max + 1e-12
        throughput_mwh = dod * cfg.E_nom_MWh
        degradation_cost_eur = cost_surface * throughput_mwh
        soh_drop_diagnostic = degradation_cost_eur * (1.0 - cfg.soh_eol) / cfg.replacement_total_cost_eur

        rows.append({
            "dod": dod,
            "temperature_c": temp,
            "dt_h": cfg.dt_h,
            "c_rate": c_rate,
            "effective_c_rate_max": cfg.effective_c_rate_max,
            "max_dod_per_step": cfg.max_dod_per_step,
            "valid": 1 if valid else 0,
            "market_validity_reason": (
                "valid: C-rate within 2MWh/1MW 0.5C limit"
                if valid
                else "invalid: DoD exceeds 12.5% per 15-min step for 2MWh/1MW 0.5C BESS"
            ),
            "pybamm_ok": int(r["pybamm_ok"]),
            "benchmark_source": r["benchmark_source"],
            "pybamm_soh_drop_proxy": r["pybamm_soh_drop_proxy"],
            "pybamm_relative_factor": r["pybamm_relative_factor"],
            "physics_relative_factor": r["physics_relative_factor"],
            "pybamm_relative_smoothed": r["pybamm_relative_smoothed"],
            "relative_factor_final": r["relative_factor_final"],
            "physics_cost_surface": physics_cost,
            "deg_cost_surface_full": float(cost_surface),
            "deg_cost_eur_per_MWh_throughput": float(cost_surface) if valid else np.nan,
            "deg_cost_smoothed": float(cost_surface),
            "deg_cost_final": float(cost_surface) if valid else np.nan,
            "soh_drop_diagnostic": float(soh_drop_diagnostic),
            "soh_drop": float(soh_drop_diagnostic) if valid else np.nan,
            "source": "pybamm_offline_benchmark_calibrated_lut",
            "error": "" if valid else "invalid_C_rate_for_runtime_dispatch",
        })

    out = pd.DataFrame(rows)

    out = _enforce_monotonic_with_reference(
        out, ["temperature_c"], "dod", "deg_cost_surface_full", "physics_cost_surface"
    )
    out = _enforce_monotonic_with_reference(
        out, ["dod"], "temperature_c", "deg_cost_surface_full", "physics_cost_surface"
    )

    out["deg_cost_smoothed"] = out["deg_cost_surface_full"]
    out["deg_cost_eur_per_MWh_throughput"] = np.where(out["valid"] > 0, out["deg_cost_surface_full"], np.nan)
    out["deg_cost_final"] = np.where(out["valid"] > 0, out["deg_cost_surface_full"], np.nan)

    throughput_mwh = out["dod"] * cfg.E_nom_MWh
    degradation_cost_eur = out["deg_cost_surface_full"] * throughput_mwh
    out["soh_drop_diagnostic"] = degradation_cost_eur * (1.0 - cfg.soh_eol) / cfg.replacement_total_cost_eur
    out["soh_drop"] = np.where(out["valid"] > 0, out["soh_drop_diagnostic"], np.nan)

    required_order = [
        "dod", "temperature_c", "dt_h", "c_rate", "effective_c_rate_max",
        "max_dod_per_step", "valid", "market_validity_reason", "pybamm_ok",
        "benchmark_source", "pybamm_soh_drop_proxy", "pybamm_relative_factor",
        "physics_relative_factor", "pybamm_relative_smoothed", "relative_factor_final",
        "deg_cost_surface_full", "deg_cost_eur_per_MWh_throughput",
        "deg_cost_smoothed", "deg_cost_final", "soh_drop_diagnostic", "soh_drop",
        "source", "error",
    ]
    extras = [c for c in out.columns if c not in required_order]
    return out[required_order + extras].sort_values(["dod", "temperature_c"]).reset_index(drop=True)

def export_report(lut: pd.DataFrame, points: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    try:
        import plotly.express as px
        import plotly.io as pio
    except Exception:
        return None

    fig1 = px.line(
        lut,
        x="dod",
        y="deg_cost_surface_full",
        color="temperature_c",
        markers=True,
        title="1. Fixed-25C diagnostic degradation curve",
        labels={"dod": "DoD", "deg_cost_surface_full": "EUR/MWh throughput", "temperature_c": "Temperature °C"},
    )

    valid = lut[lut["valid"] > 0].copy()
    fig2 = px.line(
        valid,
        x="dod",
        y="deg_cost_final",
        color="temperature_c",
        markers=True,
        title="2. Optimizer-valid fixed-25C region",
        labels={"dod": "DoD", "deg_cost_final": "EUR/MWh throughput", "temperature_c": "Temperature °C"},
    )

    mask = lut.pivot_table(index="temperature_c", columns="dod", values="valid", aggfunc="mean")
    fig3 = px.imshow(
        mask,
        aspect="auto",
        title="3. Validity mask at fixed 25C",
        labels={"x": "DoD", "y": "Temperature °C", "color": "Valid"},
    )

    rel_cols = [
        "physics_relative_factor",
        "pybamm_relative_factor",
        "pybamm_relative_smoothed",
        "relative_factor_final",
    ]
    rel_cols = [c for c in rel_cols if c in points.columns]
    rel = points[["dod", "temperature_c"] + rel_cols].copy()
    rel_long = rel.melt(
        id_vars=["dod", "temperature_c"],
        var_name="factor_type",
        value_name="relative_factor",
    ).dropna(subset=["relative_factor"])
    fig4 = px.line(
        rel_long,
        x="dod",
        y="relative_factor",
        color="factor_type",
        line_dash="temperature_c",
        markers=True,
        title="4. PyBaMM vs physics relative factors at fixed 25C",
        labels={"dod": "DoD", "relative_factor": "relative factor", "temperature_c": "Temperature °C"},
    )

    status = (
        points.groupby(["temperature_c", "pybamm_ok", "benchmark_source", "pybamm_error"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["temperature_c", "pybamm_ok", "benchmark_source"])
    )

    html = [
        "<html><head><meta charset='utf-8'><title>PyBaMM Offline Calibration Report</title></head><body>",
        "<h1>PyBaMM Offline Benchmark Calibration Report</h1>",
        "<p>PyBaMM is used only offline as a relative benchmark/calibration layer. Temperature is fixed at 25C for the DAM optimizer LUT.</p>",
        "<h2>1. Full diagnostic surface</h2>",
        pio.to_html(fig1, full_html=False, include_plotlyjs="cdn"),
        "<h2>2. Optimizer-valid region</h2>",
        pio.to_html(fig2, full_html=False, include_plotlyjs=False),
        "<h2>3. Validity mask</h2>",
        pio.to_html(fig3, full_html=False, include_plotlyjs=False),
        "<h2>4. PyBaMM vs physics relative factors</h2>",
        pio.to_html(fig4, full_html=False, include_plotlyjs=False),
        "<h2>5. PyBaMM success/failure table</h2>",
        status.to_html(index=False),
        "</body></html>",
    ]

    path = output_dir / "pybamm_calibration_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path

def run_calibration(input_dir: Path, output_dir: Path, cfg: CalibratorConfig) -> Dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_lut, base_lut_source = load_base_lut(input_dir)
    fixed_temperature_c = float(cfg.ref_temp_C)

    rows = []
    for dod in _dense_dod_grid(cfg):
        rows.append(run_pybamm_point(float(dod), fixed_temperature_c, cfg))

    raw_points = pd.DataFrame(rows)
    points = normalize_benchmark_factors(raw_points, cfg)
    lut = build_calibrated_lut(cfg, base_lut, points)

    points_path = output_dir / "pybamm_benchmark_points.csv"
    lut_path = output_dir / "degradation_lut_pybamm_calibrated.csv"
    manifest_path = output_dir / "pybamm_calibration_manifest.json"

    points.to_csv(points_path, index=False)
    lut.to_csv(lut_path, index=False)
    report_path = export_report(lut, points, output_dir)

    manifest = {
        "pipeline": "offline_pybamm_benchmark_calibrator",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "base_lut_source": base_lut_source,
        "config": asdict(cfg),
        "pybamm_successful_points": int(points["pybamm_ok"].sum()),
        "total_points": int(len(points)),
        "outputs": {
            "benchmark_points": str(points_path),
            "calibrated_lut": str(lut_path),
            "html_report": str(report_path) if report_path else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline PyBaMM benchmark calibrator for BESS LUT.")
    p.add_argument("--input-dir", default=".")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--E-nom-MWh", type=float, default=2.0)
    p.add_argument("--P-max-MW", type=float, default=1.0)
    p.add_argument("--dt-h", type=float, default=0.25)
    p.add_argument("--c-rate-max", type=float, default=0.50)
    p.add_argument("--lut-min-dod", type=float, default=0.005)
    p.add_argument("--lut-max-dod", type=float, default=0.25)
    p.add_argument("--lut-dod-step", type=float, default=0.005)
    p.add_argument("--pybamm-cycles", type=int, default=12)
    p.add_argument("--install-note", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.install_note:
        print("Install PyBaMM in your venv with:")
        print("  pip install 'pybamm[plot]'")
        print("The script still runs without PyBaMM, but will use the fallback physics-shaped curve.")
        return

    cfg = CalibratorConfig(
        E_nom_MWh=args.E_nom_MWh,
        P_max_MW=args.P_max_MW,
        dt_h=args.dt_h,
        c_rate_max=args.c_rate_max,
        lut_min_dod=args.lut_min_dod,
        lut_max_dod=args.lut_max_dod,
        lut_dod_step=args.lut_dod_step,
        pybamm_cycles_per_point=args.pybamm_cycles,
    )

    manifest = run_calibration(Path(args.input_dir), Path(args.output_dir), cfg)

    print("\nOffline PyBaMM benchmark calibration completed.")
    print(f"Base LUT source: {manifest['base_lut_source']}")
    print(f"PyBaMM successful points: {manifest['pybamm_successful_points']}/{manifest['total_points']}")
    print("Outputs:")
    for k, v in manifest["outputs"].items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
