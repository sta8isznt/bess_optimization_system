#!/usr/bin/env python3
"""
BESS full market-realistic simulation pipeline.

Final version for the 2 MWh / 1 MW BESS case.

Market-realistic technical logic
--------------------------------
Battery specs:
    E_nom_MWh = 2.0
    P_max_MW = 1.0
    dt_h = 0.25

Therefore:
    C-rate limit = P_max / E_nom = 0.5C
    max DoD per 15-min step = 0.5 * 0.25 = 0.125 = 12.5%

Important design choice
-----------------------
The LUT keeps the full diagnostic degradation surface:
    DoD = 0.05 ... 0.25

But the optimizer-ready cost is valid only where:
    C_rate = DoD / dt_h <= effective_c_rate_max

For this battery:
    DoD > 0.125 is invalid for dispatch.

Pipeline
--------
1. Load price/temperature market input.
2. Load the best available degradation LUT if present.
3. Rebuild a market-realistic full LUT with a validity mask.
4. Generate constraint-aware dispatch.
5. Evaluate SoC, SoH, degradation cost, penalties and net profit.
6. Export CSVs + HTML report.

Run
---
python bess_full_market_realistic_pipeline.py

Optional:
python bess_full_market_realistic_pipeline.py --input-dir . --output-dir .
python bess_full_market_realistic_pipeline.py --E-nom-MWh 2 --P-max-MW 1 --c-rate-max 0.5
python bess_full_market_realistic_pipeline.py --dod-min 0.05 --dod-max 0.25 --dod-step 0.01
python bess_full_market_realistic_pipeline.py --temp-min 10 --temp-max 40 --temp-step 2.5
python bess_full_market_realistic_pipeline.py --dod-grid 0.05,0.08,0.10 --temp-grid 10,25,40
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BESSConfig:
    # Final market-realistic default: 2 MWh / 1 MW = 0.5C = 2-hour battery.
    E_nom_MWh: float = 2.0
    P_max_MW: float = 1.0
    dt_h: float = 0.25

    # State limits.
    soc_min: float = 0.10
    soc_max: float = 0.90
    soh_eol: float = 0.80

    # The effective value is min(c_rate_max, P_max/E_nom).
    # For 2 MWh / 1 MW this is 0.5C.
    c_rate_max: float = 0.50

    # AC-side efficiency.
    eta_charge: float = 0.92
    eta_discharge: float = 0.92

    # Simple liquid-cooled BESS thermal proxy.
    T_cell_min_C: float = 5.0
    T_cell_max_C: float = 40.0
    thermal_power_coeff_C_per_Crate: float = 6.0

    # Economics.
    replacement_cost_eur_per_MWh_capacity: float = 60_000.0

    # Dispatch heuristic.
    soc_guard_band: float = 0.002
    low_price_quantile: float = 0.25
    high_price_quantile: float = 0.75
    dispatch_power_fraction: float = 1.00

    # Fallback degradation surface calibration.
    # Used only if no valid external LUT point exists.
    base_deg_cost_eur_per_MWh_at_ref: float = 3.0
    ref_dod: float = 0.10
    ref_temp_C: float = 25.0
    ref_c_rate: float = 0.40
    dod_exponent: float = 1.35
    temp_coeff_per_C: float = 0.022
    c_rate_exponent: float = 0.20

    # Penalties. Feasible generated dispatch should have zero penalties.
    penalty_soc: float = 1.0e6
    penalty_power: float = 1.0e4
    penalty_c_rate: float = 1.0e7
    penalty_temperature: float = 1.0e5
    penalty_simultaneous: float = 1.0e6
    penalty_soh: float = 1.0e8

    @property
    def replacement_total_cost_eur(self) -> float:
        return self.replacement_cost_eur_per_MWh_capacity * self.E_nom_MWh

    @property
    def roundtrip_efficiency(self) -> float:
        return self.eta_charge * self.eta_discharge

    @property
    def physical_c_rate_from_power(self) -> float:
        return self.P_max_MW / self.E_nom_MWh

    @property
    def effective_c_rate_max(self) -> float:
        return min(self.c_rate_max, self.physical_c_rate_from_power)

    @property
    def p_c_rate_limit_MW(self) -> float:
        return self.effective_c_rate_max * self.E_nom_MWh

    @property
    def max_dod_per_step(self) -> float:
        return self.effective_c_rate_max * self.dt_h


# ---------------------------------------------------------------------------
# Loading / input preparation
# ---------------------------------------------------------------------------

def _find_first_existing(input_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for name in candidates:
        p = input_dir / name
        if p.exists():
            return p
    return None


def load_market_input(input_dir: Path) -> Tuple[pd.DataFrame, str]:
    """
    Load timestamp, price and temperature.

    We intentionally ignore old P_ch_MW/P_dis_MW columns because this script
    rebuilds a clean physically feasible dispatch.
    """
    source = _find_first_existing(
        input_dir,
        [
            "market_input.csv",
            "bess_market_input.csv",
            "bess_market_input_realistic.csv",
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
        elif "T_cell_C" in df.columns:
            out["T_amb_C"] = pd.to_numeric(df["T_cell_C"], errors="coerce").fillna(25.0)
        elif "T_cell_C_used" in df.columns:
            out["T_amb_C"] = pd.to_numeric(df["T_cell_C_used"], errors="coerce").fillna(25.0)
        else:
            out["T_amb_C"] = 25.0

        out = out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return out, source.name

    # Fallback synthetic one-day market profile.
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2030-01-01", periods=96, freq="15min")
    price = (
        80.0
        + 35.0 * np.sin(np.linspace(-np.pi, np.pi, len(timestamps)))
        + 10.0 * rng.normal(size=len(timestamps))
    )
    out = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price_eur_per_MWh": price,
            "T_amb_C": 25.0,
        }
    )
    return out, "synthetic_generated_market"


# ---------------------------------------------------------------------------
# Physical formulas
# ---------------------------------------------------------------------------

def estimate_cell_temperature_C(T_amb_C: float, P_ch_MW: float, P_dis_MW: float, cfg: BESSConfig) -> float:
    p_abs = max(P_ch_MW, P_dis_MW)
    c_rate = p_abs / cfg.E_nom_MWh
    return float(T_amb_C + cfg.thermal_power_coeff_C_per_Crate * c_rate)


def update_soc(soc: float, soh: float, P_ch_MW: float, P_dis_MW: float, cfg: BESSConfig) -> float:
    usable_capacity_MWh = cfg.E_nom_MWh * max(soh, cfg.soh_eol)
    return float(
        soc
        + cfg.eta_charge * P_ch_MW * cfg.dt_h / usable_capacity_MWh
        - P_dis_MW * cfg.dt_h / (cfg.eta_discharge * usable_capacity_MWh)
    )


def timestep_dod(P_ch_MW: float, P_dis_MW: float, cfg: BESSConfig) -> float:
    return float(max(P_ch_MW, P_dis_MW) * cfg.dt_h / cfg.E_nom_MWh)


def soh_drop_from_degradation_cost(cost_eur: float, cfg: BESSConfig) -> float:
    return float(cost_eur * (1.0 - cfg.soh_eol) / cfg.replacement_total_cost_eur)


# ---------------------------------------------------------------------------
# Market-realistic degradation LUT
# ---------------------------------------------------------------------------

def _default_dod_grid() -> List[float]:
    # Full diagnostic surface; validity is applied separately.
    return [0.05, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25]


def _default_temp_grid() -> List[float]:
    return [10, 20, 25, 30, 35, 40]


def _parse_float_list(value: Optional[str]) -> Optional[List[float]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    return [float(p) for p in parts]


def _build_range_grid(min_value: float, max_value: float, step: float) -> List[float]:
    if step <= 0:
        raise ValueError("step must be positive")
    if max_value < min_value:
        raise ValueError("max_value must be >= min_value")
    values = []
    current = min_value
    while current <= max_value + 1e-12:
        values.append(float(current))
        current += step
    return values


def _cost_column_candidates(raw_lut: pd.DataFrame) -> List[str]:
    return [
        c for c in [
            "aggressive_cost_final_surface_eur_per_MWh",
            "hybrid_cost_final_eur_per_MWh",
            "deg_cost_smoothed",
            "deg_cost_final",
            "deg_cost_eur_per_MWh_throughput",
            "deg_cost_eur_per_MWh",
            "deg_cost_final_hybrid",
        ]
        if c in raw_lut.columns
    ]


def fallback_deg_cost_surface(dod: float, temp_c: float, cfg: BESSConfig, base_cost: Optional[float] = None) -> float:
    if dod <= 0:
        return 0.0
    c_rate = dod / cfg.dt_h
    base = cfg.base_deg_cost_eur_per_MWh_at_ref if base_cost is None else float(base_cost)

    dod_factor = (max(dod, 1e-9) / cfg.ref_dod) ** cfg.dod_exponent
    temp_factor = float(np.exp(cfg.temp_coeff_per_C * (temp_c - cfg.ref_temp_C)))
    c_rate_factor = (max(c_rate, 1e-9) / cfg.ref_c_rate) ** cfg.c_rate_exponent

    # Additional smooth stress above the market-realistic boundary, kept for diagnostics.
    high_dod_stress = 1.0 + 0.45 * max(dod / cfg.max_dod_per_step - 1.0, 0.0) ** 1.25
    high_temp_stress = 1.0 + 0.18 * max((temp_c - 35.0) / 10.0, 0.0) ** 1.20

    return float(base * dod_factor * temp_factor * c_rate_factor * high_dod_stress * high_temp_stress)


def _estimate_base_from_raw(raw_lut: Optional[pd.DataFrame], cfg: BESSConfig) -> float:
    if raw_lut is None or raw_lut.empty:
        return cfg.base_deg_cost_eur_per_MWh_at_ref

    cols = _cost_column_candidates(raw_lut)
    if not cols:
        return cfg.base_deg_cost_eur_per_MWh_at_ref

    lut = raw_lut.copy()
    if "dod" not in lut.columns or "temperature_c" not in lut.columns:
        return cfg.base_deg_cost_eur_per_MWh_at_ref

    for col in cols:
        valid = lut.dropna(subset=["dod", "temperature_c", col]).copy()
        valid = valid[pd.to_numeric(valid[col], errors="coerce") > 0]
        if valid.empty:
            continue

        dist = (
            ((valid["dod"] - cfg.ref_dod) / max(valid["dod"].max() - valid["dod"].min(), 1e-9)) ** 2
            + ((valid["temperature_c"] - cfg.ref_temp_C) / max(valid["temperature_c"].max() - valid["temperature_c"].min(), 1e-9)) ** 2
        )
        value = float(valid.loc[dist.idxmin(), col])
        if np.isfinite(value) and value > 0:
            return float(np.clip(value, 1.0, 80.0))

    return cfg.base_deg_cost_eur_per_MWh_at_ref


def _nearest_raw_surface_value(raw_lut: Optional[pd.DataFrame], dod: float, temp_c: float) -> Optional[float]:
    if raw_lut is None or raw_lut.empty:
        return None
    if "dod" not in raw_lut.columns or "temperature_c" not in raw_lut.columns:
        return None

    cols = _cost_column_candidates(raw_lut)
    if not cols:
        return None

    # Prefer exact row; fall back to nearest row.
    lut = raw_lut.copy()
    dod_scale = max(float(lut["dod"].max() - lut["dod"].min()), 1e-9)
    temp_scale = max(float(lut["temperature_c"].max() - lut["temperature_c"].min()), 1e-9)
    lut["_dist"] = ((lut["dod"] - dod) / dod_scale) ** 2 + ((lut["temperature_c"] - temp_c) / temp_scale) ** 2
    lut = lut.sort_values("_dist")

    for _, row in lut.iterrows():
        for col in cols:
            value = row.get(col)
            if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0:
                return float(value)
        if row["_dist"] <= 1e-18:
            break

    return None


def build_market_realistic_lut(
    cfg: BESSConfig,
    raw_lut: Optional[pd.DataFrame] = None,
    dod_values: Optional[Iterable[float]] = None,
    temp_values_c: Optional[Iterable[float]] = None,
    include_default_grid: bool = True,
) -> pd.DataFrame:
    """
    Build final LUT:
    - Full surface exists for DoD 0.05-0.25.
    - Optimizer-ready `deg_cost_final` is NaN for invalid C-rate points.
    """
    # Always keep full grid. Merge raw grid if present, but do not lose defaults.
    dod_grid = set(_default_dod_grid()) if include_default_grid else set()
    temp_grid = set(_default_temp_grid()) if include_default_grid else set()

    if dod_values is not None:
        dod_grid |= set(float(x) for x in dod_values)
    if temp_values_c is not None:
        temp_grid |= set(float(x) for x in temp_values_c)

    if not dod_grid or not temp_grid:
        raise ValueError("DoD and temperature grids must be non-empty")

    dod_grid = sorted(dod_grid)
    temp_grid = sorted(temp_grid)
    base = _estimate_base_from_raw(raw_lut, cfg)

    rows: List[Dict[str, object]] = []

    for dod in dod_grid:
        for temp in temp_grid:
            c_rate = dod / cfg.dt_h
            raw_surface = _nearest_raw_surface_value(raw_lut, dod, temp)
            if raw_surface is None:
                raw_surface = fallback_deg_cost_surface(dod, temp, cfg, base_cost=base)

            # Mild final market-realistic stress correction above the feasible boundary.
            # It is diagnostic only if the point is invalid.
            dod_over_limit = max(dod / cfg.max_dod_per_step - 1.0, 0.0)
            stress_multiplier = 1.0 + 0.15 * dod_over_limit ** 1.25
            surface_cost = float(raw_surface * stress_multiplier)

            market_valid = c_rate <= cfg.effective_c_rate_max + 1e-12

            rows.append(
                {
                    "dod": float(dod),
                    "temperature_c": float(temp),
                    "dt_h": cfg.dt_h,
                    "c_rate": float(c_rate),
                    "effective_c_rate_max": cfg.effective_c_rate_max,
                    "max_dod_per_step": cfg.max_dod_per_step,
                    "valid": 1.0 if market_valid else 0.0,
                    "market_validity_reason": (
                        "valid: C-rate within 2MWh/1MW 0.5C limit"
                        if market_valid
                        else "invalid: DoD exceeds 12.5% per 15-min step for 2MWh/1MW 0.5C BESS"
                    ),
                    "raw_or_fallback_surface_cost": float(raw_surface),
                    "stress_multiplier": float(stress_multiplier),
                    "deg_cost_surface_full": float(surface_cost),
                    "deg_cost_eur_per_MWh_throughput": float(surface_cost) if market_valid else np.nan,
                    "deg_cost_smoothed": float(surface_cost),
                    "deg_cost_final": float(surface_cost) if market_valid else np.nan,
                    "source": "market_realistic_full_surface_with_c_rate_validity_mask",
                    "error": "" if market_valid else "invalid_C_rate_for_2MWh_1MW_15min",
                }
            )

    out = pd.DataFrame(rows)

    # Enforce monotonicity across DoD for each temperature on the full surface.
    out = out.sort_values(["temperature_c", "dod"]).reset_index(drop=True)
    out["deg_cost_surface_full"] = out.groupby("temperature_c")["deg_cost_surface_full"].cummax()

    # Enforce monotonicity across temperature for each DoD.
    out = out.sort_values(["dod", "temperature_c"]).reset_index(drop=True)
    out["deg_cost_surface_full"] = out.groupby("dod")["deg_cost_surface_full"].cummax()

    out["deg_cost_smoothed"] = out["deg_cost_surface_full"]
    out["deg_cost_eur_per_MWh_throughput"] = np.where(out["valid"] > 0, out["deg_cost_surface_full"], np.nan)
    out["deg_cost_final"] = np.where(out["valid"] > 0, out["deg_cost_surface_full"], np.nan)

    # Convert cost/MWh to SoH drop for one timestep at this DoD for valid points.
    throughput_mwh = out["dod"] * cfg.E_nom_MWh
    degradation_cost_eur = out["deg_cost_surface_full"] * throughput_mwh
    out["soh_drop_diagnostic"] = degradation_cost_eur.apply(lambda x: soh_drop_from_degradation_cost(float(x), cfg))
    out["soh_drop"] = np.where(out["valid"] > 0, out["soh_drop_diagnostic"], np.nan)

    return out.sort_values(["dod", "temperature_c"]).reset_index(drop=True)


def lookup_degradation_cost_per_mwh(
    dod: float,
    temp_c: float,
    lut: Optional[pd.DataFrame],
    cfg: BESSConfig,
) -> Tuple[float, str]:
    if dod <= 0:
        return 0.0, "zero_dispatch"

    # If external dispatch exceeds market-realistic DoD, return a valid cost estimate,
    # but the physical penalty layer will flag the dispatch as infeasible.
    if dod > cfg.max_dod_per_step + 1e-12:
        surface = fallback_deg_cost_surface(dod, temp_c, cfg)
        return surface, "fallback_surface_invalid_dod_penalty_applies"

    if lut is None or lut.empty:
        return fallback_deg_cost_surface(dod, temp_c, cfg), "fallback_no_lut"

    if "deg_cost_final" in lut.columns:
        cost_col = "deg_cost_final"
    elif "deg_cost_eur_per_MWh_throughput" in lut.columns:
        cost_col = "deg_cost_eur_per_MWh_throughput"
    else:
        return fallback_deg_cost_surface(dod, temp_c, cfg), "fallback_missing_cost_column"

    valid = lut.dropna(subset=["dod", "temperature_c", cost_col]).copy()
    if "valid" in valid.columns:
        valid = valid[pd.to_numeric(valid["valid"], errors="coerce").fillna(0) > 0]

    if valid.empty:
        return fallback_deg_cost_surface(dod, temp_c, cfg), "fallback_invalid_lut"

    dod_scale = max(valid["dod"].max() - valid["dod"].min(), 1e-9)
    temp_scale = max(valid["temperature_c"].max() - valid["temperature_c"].min(), 1e-9)
    dist = ((valid["dod"] - dod) / dod_scale) ** 2 + ((valid["temperature_c"] - temp_c) / temp_scale) ** 2
    i = dist.idxmin()
    return float(valid.loc[i, cost_col]), f"nearest_{cost_col}_market_valid"


# ---------------------------------------------------------------------------
# Constraint-aware dispatch simulator
# ---------------------------------------------------------------------------

def _allowed_charge_power_MW(soc: float, soh: float, cfg: BESSConfig) -> float:
    usable_capacity_MWh = cfg.E_nom_MWh * max(soh, cfg.soh_eol)
    headroom_MWh = max(0.0, ((cfg.soc_max - cfg.soc_guard_band) - soc) * usable_capacity_MWh)
    by_soc = headroom_MWh / max(cfg.eta_charge * cfg.dt_h, 1e-12)
    return float(max(0.0, min(cfg.P_max_MW, cfg.p_c_rate_limit_MW, by_soc)))


def _allowed_discharge_power_MW(soc: float, soh: float, cfg: BESSConfig) -> float:
    usable_capacity_MWh = cfg.E_nom_MWh * max(soh, cfg.soh_eol)
    available_MWh = max(0.0, (soc - (cfg.soc_min + cfg.soc_guard_band)) * usable_capacity_MWh)
    by_soc = available_MWh * cfg.eta_discharge / max(cfg.dt_h, 1e-12)
    return float(max(0.0, min(cfg.P_max_MW, cfg.p_c_rate_limit_MW, by_soc)))


def make_constraint_aware_dispatch(
    market: pd.DataFrame,
    cfg: BESSConfig,
    initial_soc: float = 0.50,
    initial_soh: float = 1.00,
) -> pd.DataFrame:
    required = {"timestamp", "price_eur_per_MWh", "T_amb_C"}
    missing = sorted(required - set(market.columns))
    if missing:
        raise ValueError(f"Market input is missing columns: {missing}")

    out = market.copy().sort_values("timestamp").reset_index(drop=True)

    prices = out["price_eur_per_MWh"].astype(float)
    low_thr = float(prices.quantile(cfg.low_price_quantile))
    high_thr = float(prices.quantile(cfg.high_price_quantile))
    base_power = min(cfg.P_max_MW, cfg.p_c_rate_limit_MW) * cfg.dispatch_power_fraction

    soc = float(initial_soc)
    soh = float(initial_soh)
    rows: List[Dict[str, object]] = []

    for _, row in out.iterrows():
        price = float(row["price_eur_per_MWh"])
        T_amb = float(row["T_amb_C"])

        P_ch = 0.0
        P_dis = 0.0
        action = "idle"

        if price <= low_thr:
            P_ch = min(base_power, _allowed_charge_power_MW(soc, soh, cfg))
            action = "charge" if P_ch > 1e-9 else "idle_soc_full"
        elif price >= high_thr:
            P_dis = min(base_power, _allowed_discharge_power_MW(soc, soh, cfg))
            action = "discharge" if P_dis > 1e-9 else "idle_soc_empty"

        if P_ch < 1e-9:
            P_ch = 0.0
        if P_dis < 1e-9:
            P_dis = 0.0

        # Safety: never allow both.
        if P_ch > 0 and P_dis > 0:
            if price <= (low_thr + high_thr) / 2:
                P_dis = 0.0
            else:
                P_ch = 0.0

        soc_next = update_soc(soc, soh, P_ch, P_dis, cfg)
        soc_next = float(np.clip(soc_next, cfg.soc_min, cfg.soc_max))
        T_cell = estimate_cell_temperature_C(T_amb, P_ch, P_dis, cfg)

        rows.append(
            {
                "P_ch_MW": P_ch,
                "P_dis_MW": P_dis,
                "action": action,
                "dispatch_low_price_threshold": low_thr,
                "dispatch_high_price_threshold": high_thr,
                "simulated_SoC_before": soc,
                "simulated_SoC_after": soc_next,
                "simulated_T_cell_C": T_cell,
            }
        )
        soc = soc_next

    return pd.concat([out, pd.DataFrame(rows)], axis=1)


# ---------------------------------------------------------------------------
# Financial layer
# ---------------------------------------------------------------------------

def physical_violations(
    soc: float,
    soc_next: float,
    soh: float,
    soh_next: float,
    P_ch_MW: float,
    P_dis_MW: float,
    T_cell_C: float,
    cfg: BESSConfig,
) -> List[str]:
    violations: List[str] = []
    tol = 1e-9

    if P_ch_MW < -tol or P_dis_MW < -tol:
        violations.append("negative_power")
    if P_ch_MW > tol and P_dis_MW > tol:
        violations.append("simultaneous_charge_discharge")
    if P_ch_MW > cfg.P_max_MW + tol:
        violations.append("charge_power_exceeded")
    if P_dis_MW > cfg.P_max_MW + tol:
        violations.append("discharge_power_exceeded")

    c_rate = max(P_ch_MW, P_dis_MW) / cfg.E_nom_MWh
    if c_rate > cfg.effective_c_rate_max + tol:
        violations.append("C_rate_exceeded")

    dod_step = max(P_ch_MW, P_dis_MW) * cfg.dt_h / cfg.E_nom_MWh
    if dod_step > cfg.max_dod_per_step + tol:
        violations.append("DoD_step_exceeded")

    if soc < cfg.soc_min - tol or soc > cfg.soc_max + tol:
        violations.append("SoC_current_out_of_bounds")
    if soc_next < cfg.soc_min - tol or soc_next > cfg.soc_max + tol:
        violations.append("SoC_next_out_of_bounds")

    if T_cell_C < cfg.T_cell_min_C - tol:
        violations.append("cell_temperature_too_low")
    if T_cell_C > cfg.T_cell_max_C + tol:
        violations.append("cell_temperature_too_high")

    if soh_next > soh + 1e-12:
        violations.append("SoH_increased")
    if soh_next < cfg.soh_eol - tol:
        violations.append("EOL_reached")

    return violations


def infeasibility_penalty_eur(
    soc: float,
    soc_next: float,
    soh: float,
    soh_next: float,
    P_ch_MW: float,
    P_dis_MW: float,
    T_cell_C: float,
    cfg: BESSConfig,
) -> float:
    penalty = 0.0

    for x in [soc, soc_next]:
        penalty += cfg.penalty_soc * max(0.0, cfg.soc_min - x) ** 2
        penalty += cfg.penalty_soc * max(0.0, x - cfg.soc_max) ** 2

    penalty += cfg.penalty_power * max(0.0, P_ch_MW - cfg.P_max_MW) ** 2
    penalty += cfg.penalty_power * max(0.0, P_dis_MW - cfg.P_max_MW) ** 2
    penalty += cfg.penalty_power * max(0.0, -P_ch_MW) ** 2
    penalty += cfg.penalty_power * max(0.0, -P_dis_MW) ** 2

    if P_ch_MW > 0 and P_dis_MW > 0:
        simultaneous_ratio = min(P_ch_MW, P_dis_MW) / max(cfg.P_max_MW, 1e-12)
        penalty += cfg.penalty_simultaneous * simultaneous_ratio ** 2

    c_rate = max(P_ch_MW, P_dis_MW) / cfg.E_nom_MWh
    penalty += cfg.penalty_c_rate * max(0.0, c_rate - cfg.effective_c_rate_max) ** 2

    dod_step = max(P_ch_MW, P_dis_MW) * cfg.dt_h / cfg.E_nom_MWh
    penalty += cfg.penalty_c_rate * max(0.0, dod_step - cfg.max_dod_per_step) ** 2

    penalty += cfg.penalty_temperature * max(0.0, cfg.T_cell_min_C - T_cell_C) ** 2
    penalty += cfg.penalty_temperature * max(0.0, T_cell_C - cfg.T_cell_max_C) ** 2

    penalty += cfg.penalty_soh * max(0.0, soh_next - soh - 1e-12) ** 2
    penalty += cfg.penalty_soh * max(0.0, cfg.soh_eol - soh_next) ** 2

    return float(penalty)


def run_bess_financial_layer(
    dispatch: pd.DataFrame,
    cfg: BESSConfig,
    degradation_lut: pd.DataFrame,
    initial_soc: float = 0.50,
    initial_soh: float = 1.00,
) -> pd.DataFrame:
    required = {"timestamp", "price_eur_per_MWh", "T_amb_C", "P_ch_MW", "P_dis_MW"}
    missing = sorted(required - set(dispatch.columns))
    if missing:
        raise ValueError(f"Dispatch input is missing columns: {missing}")

    out = dispatch.copy().sort_values("timestamp").reset_index(drop=True)

    soc = float(initial_soc)
    soh = float(initial_soh)
    rows: List[Dict[str, object]] = []

    for _, row in out.iterrows():
        P_ch = float(row["P_ch_MW"])
        P_dis = float(row["P_dis_MW"])
        price = float(row["price_eur_per_MWh"])
        T_amb = float(row["T_amb_C"])
        T_cell = estimate_cell_temperature_C(T_amb, P_ch, P_dis, cfg)

        soc_next = update_soc(soc, soh, P_ch, P_dis, cfg)
        dod_step = timestep_dod(P_ch, P_dis, cfg)
        throughput_MWh = max(P_ch, P_dis) * cfg.dt_h

        deg_cost_per_mwh, deg_source = lookup_degradation_cost_per_mwh(dod_step, T_cell, degradation_lut, cfg)
        degradation_cost = deg_cost_per_mwh * throughput_MWh

        delta_soh = soh_drop_from_degradation_cost(degradation_cost, cfg)
        soh_next = max(0.0, soh - delta_soh)

        charge_cost = P_ch * cfg.dt_h * price
        discharge_revenue = P_dis * cfg.dt_h * price
        gross_margin = discharge_revenue - charge_cost

        violations = physical_violations(
            soc=soc,
            soc_next=soc_next,
            soh=soh,
            soh_next=soh_next,
            P_ch_MW=P_ch,
            P_dis_MW=P_dis,
            T_cell_C=T_cell,
            cfg=cfg,
        )
        penalty = infeasibility_penalty_eur(
            soc=soc,
            soc_next=soc_next,
            soh=soh,
            soh_next=soh_next,
            P_ch_MW=P_ch,
            P_dis_MW=P_dis,
            T_cell_C=T_cell,
            cfg=cfg,
        )

        net_profit = gross_margin - degradation_cost - penalty

        rows.append(
            {
                "SoC": soc,
                "SoC_next": soc_next,
                "SoH": soh,
                "SoH_next": soh_next,
                "delta_SoH": delta_soh,
                "T_cell_C_used": T_cell,
                "C_rate": max(P_ch, P_dis) / cfg.E_nom_MWh,
                "DoD_step": dod_step,
                "throughput_MWh": throughput_MWh,
                "charge_cost_eur": charge_cost,
                "discharge_revenue_eur": discharge_revenue,
                "gross_margin_eur": gross_margin,
                "deg_cost_eur_per_MWh": deg_cost_per_mwh,
                "degradation_cost_eur": degradation_cost,
                "degradation_source": deg_source,
                "infeasibility_penalty_eur": penalty,
                "net_profit_eur": net_profit,
                "violations": violations,
                "is_feasible": len(violations) == 0,
            }
        )

        soc = float(np.clip(soc_next, cfg.soc_min, cfg.soc_max))
        soh = soh_next

    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def summarize_financial_result(result: pd.DataFrame, cfg: BESSConfig) -> pd.DataFrame:
    all_violations: List[str] = []
    for values in result["violations"]:
        if isinstance(values, list):
            all_violations.extend(values)
        elif isinstance(values, str) and values not in ("", "[]"):
            all_violations.append(values)

    counts = pd.Series(all_violations).value_counts() if all_violations else pd.Series(dtype=int)

    summary = {
        "rows": len(result),
        "feasible_rows": int(result["is_feasible"].sum()),
        "feasible_share": float(result["is_feasible"].mean()),
        "E_nom_MWh": cfg.E_nom_MWh,
        "P_max_MW": cfg.P_max_MW,
        "physical_c_rate_from_power": cfg.physical_c_rate_from_power,
        "effective_c_rate_max": cfg.effective_c_rate_max,
        "dt_h": cfg.dt_h,
        "max_DoD_step_allowed": cfg.max_dod_per_step,
        "total_charge_MWh": float((result["P_ch_MW"] * cfg.dt_h).sum()),
        "total_discharge_MWh": float((result["P_dis_MW"] * cfg.dt_h).sum()),
        "total_throughput_MWh": float(result["throughput_MWh"].sum()),
        "total_gross_margin_eur": float(result["gross_margin_eur"].sum()),
        "total_degradation_cost_eur": float(result["degradation_cost_eur"].sum()),
        "total_penalty_eur": float(result["infeasibility_penalty_eur"].sum()),
        "total_net_profit_eur": float(result["net_profit_eur"].sum()),
        "total_delta_SoH": float(result["delta_SoH"].sum()),
        "initial_SoC": float(result["SoC"].iloc[0]),
        "final_SoC": float(result["SoC_next"].iloc[-1]),
        "initial_SoH": float(result["SoH"].iloc[0]),
        "final_SoH": float(result["SoH_next"].iloc[-1]),
        "max_C_rate_observed": float(result["C_rate"].max()),
        "max_DoD_step_observed": float(result["DoD_step"].max()),
        "max_T_cell_C": float(result["T_cell_C_used"].max()),
    }

    out = pd.DataFrame([summary])
    for violation, count in counts.items():
        out[f"violation__{violation}"] = int(count)
    return out


# ---------------------------------------------------------------------------
# Outputs / visualization
# ---------------------------------------------------------------------------

def make_telemetry_table(result: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp",
        "price_eur_per_MWh",
        "T_amb_C",
        "T_cell_C_used",
        "P_ch_MW",
        "P_dis_MW",
        "action",
        "SoC",
        "SoC_next",
        "SoH",
        "SoH_next",
        "delta_SoH",
        "C_rate",
        "DoD_step",
        "throughput_MWh",
        "gross_margin_eur",
        "degradation_cost_eur",
        "net_profit_eur",
        "is_feasible",
        "violations",
    ]
    return result[[c for c in cols if c in result.columns]].copy()


def try_export_html_report(result: pd.DataFrame, lut: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except Exception:
        return None

    df = result.copy()
    df["P_net_MW"] = df["P_dis_MW"] - df["P_ch_MW"]
    for c in ["gross_margin_eur", "degradation_cost_eur", "infeasibility_penalty_eur", "net_profit_eur"]:
        df[f"cum_{c}"] = df[c].fillna(0).cumsum()

    figs = []

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["P_net_MW"], name="Net power MW"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price_eur_per_MWh"], name="Price €/MWh", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title="Market-realistic constraint-aware dispatch vs price", hovermode="x unified")
    fig.update_yaxes(title_text="MW", secondary_y=False)
    fig.update_yaxes(title_text="€/MWh", secondary_y=True)
    figs.append(("Dispatch vs price", fig))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SoC"], name="SoC", mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SoH"], name="SoH", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title="SoC / SoH trend", hovermode="x unified")
    fig.update_yaxes(title_text="SoC", secondary_y=False)
    fig.update_yaxes(title_text="SoH", secondary_y=True)
    figs.append(("SoC / SoH", fig))

    fig = go.Figure()
    for col, name in [
        ("cum_gross_margin_eur", "gross margin"),
        ("cum_degradation_cost_eur", "degradation cost"),
        ("cum_infeasibility_penalty_eur", "penalty"),
        ("cum_net_profit_eur", "net profit"),
    ]:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=name, mode="lines+markers"))
    fig.update_layout(title="Cumulative economics", yaxis_title="EUR", hovermode="x unified")
    figs.append(("Cumulative economics", fig))

    fig = px.line(
        lut,
        x="dod",
        y="deg_cost_surface_full",
        color="temperature_c",
        markers=True,
        title="Full degradation surface: DoD 0.05–0.25",
        labels={"dod": "DoD", "deg_cost_surface_full": "EUR/MWh throughput", "temperature_c": "Temperature °C"},
    )
    figs.append(("Full degradation surface", fig))

    valid_lut = lut[lut["valid"] > 0].copy()
    fig = px.line(
        valid_lut,
        x="dod",
        y="deg_cost_final",
        color="temperature_c",
        markers=True,
        title="Optimizer-valid degradation LUT: DoD ≤ 0.125 for 2MWh/1MW 0.5C BESS",
        labels={"dod": "DoD", "deg_cost_final": "EUR/MWh throughput", "temperature_c": "Temperature °C"},
    )
    figs.append(("Optimizer-valid degradation LUT", fig))

    mask = lut.pivot_table(index="temperature_c", columns="dod", values="valid", aggfunc="mean")
    fig = px.imshow(
        mask,
        aspect="auto",
        title="Validity mask: DoD > 0.125 invalid for this 2MWh/1MW battery",
        labels={"x": "DoD", "y": "Temperature °C", "color": "Valid"},
    )
    figs.append(("Validity mask", fig))

    fig = px.scatter(
        df,
        x="price_eur_per_MWh",
        y="net_profit_eur",
        color="action" if "action" in df.columns else None,
        size="throughput_MWh",
        title="Price vs net profit",
        hover_data=["timestamp", "SoC", "SoC_next", "degradation_cost_eur", "is_feasible"],
    )
    figs.append(("Price vs profit", fig))

    html = [
        "<html><head><meta charset='utf-8'><title>BESS Market-Realistic Simulation Report</title></head><body>",
        "<h1>BESS Market-Realistic Simulation Report</h1>",
        "<p>Final version: 2 MWh / 1 MW BESS, 0.5C, 15-min max DoD = 12.5%.</p>",
        "<h2>Summary</h2>",
        summary.T.reset_index().rename(columns={"index": "metric", 0: "value"}).to_html(index=False),
    ]
    for i, (title, fig) in enumerate(figs):
        html.append(f"<h2>{title}</h2>")
        html.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html.append("</body></html>")

    path = output_dir / "bess_market_realistic_visual_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    initial_soc: float = 0.50,
    initial_soh: float = 1.00,
    cfg: Optional[BESSConfig] = None,
    dod_values: Optional[Iterable[float]] = None,
    temp_values_c: Optional[Iterable[float]] = None,
    include_default_grid: bool = True,
    include_raw_grid: bool = True,
) -> Dict[str, object]:
    cfg = cfg or BESSConfig()
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts, metadata = build_market_realistic_artifacts(
        input_dir=input_dir,
        cfg=cfg,
        initial_soc=initial_soc,
        initial_soh=initial_soh,
        dod_values=dod_values,
        temp_values_c=temp_values_c,
        include_default_grid=include_default_grid,
        include_raw_grid=include_raw_grid,
    )

    market = artifacts["market"]
    lut = artifacts["lut"]
    dispatch = artifacts["dispatch"]
    result = artifacts["result"]
    summary = artifacts["summary"]
    telemetry = artifacts["telemetry"]

    outputs = {
        "market": output_dir / "bess_market_input_market_realistic.csv",
        "lut": output_dir / "degradation_lut_market_realistic.csv",
        "dispatch": output_dir / "bess_dispatch_market_realistic.csv",
        "result": output_dir / "bess_financial_result_market_realistic.csv",
        "summary": output_dir / "bess_financial_summary_market_realistic.csv",
        "telemetry": output_dir / "bess_telemetry_market_realistic.csv",
        "manifest": output_dir / "pipeline_market_realistic_manifest.json",
    }

    market.to_csv(outputs["market"], index=False)
    lut.to_csv(outputs["lut"], index=False)
    dispatch.to_csv(outputs["dispatch"], index=False)
    result.to_csv(outputs["result"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    telemetry.to_csv(outputs["telemetry"], index=False)

    report_path = try_export_html_report(result, lut, summary, output_dir)
    if report_path is not None:
        outputs["html_report"] = report_path

    manifest = {
        "pipeline": "bess_full_market_realistic_pipeline",
        "market_logic": "2MWh/1MW BESS => 0.5C => max DoD per 15-min step = 0.125",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "market_source": metadata["market_source"],
        "raw_lut_source": metadata["raw_lut_source"],
        "grid_config": {
            "include_default_grid": include_default_grid,
            "include_raw_grid": include_raw_grid,
            "dod_values": list(dod_values) if dod_values is not None else None,
            "temp_values_c": list(temp_values_c) if temp_values_c is not None else None,
        },
        "config": asdict(cfg),
        "derived_constraints": {
            "physical_c_rate_from_power": cfg.physical_c_rate_from_power,
            "effective_c_rate_max": cfg.effective_c_rate_max,
            "max_dod_per_step": cfg.max_dod_per_step,
        },
        "outputs": {k: str(v) for k, v in outputs.items()},
        "summary": summary.iloc[0].to_dict(),
    }

    outputs["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def build_market_realistic_artifacts(
    input_dir: Path,
    cfg: BESSConfig,
    initial_soc: float = 0.50,
    initial_soh: float = 1.00,
    dod_values: Optional[Iterable[float]] = None,
    temp_values_c: Optional[Iterable[float]] = None,
    include_default_grid: bool = True,
    include_raw_grid: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    """Run the full pipeline and return dataframes for modular integration."""
    input_dir = input_dir.resolve()
    market, market_source = load_market_input(input_dir)

    raw_lut_path = _find_first_existing(
        input_dir,
        [
            "degradation_lut_final_aggressive_hybrid_combined.csv",
            "degradation_lut_final_aggressive_optimizer_ready.csv",
            "degradation_lut_hybrid_combined.csv",
            "degradation_lut_hybrid_optimizer_ready.csv",
            "degradation_lut_fixed.csv",
            "degradation_lut.csv",
        ],
    )
    raw_lut = pd.read_csv(raw_lut_path) if raw_lut_path is not None else None

    raw_dod_values = (
        sorted(raw_lut["dod"].dropna().unique().tolist())
        if raw_lut is not None and "dod" in raw_lut.columns
        else None
    )
    raw_temp_values = (
        sorted(raw_lut["temperature_c"].dropna().unique().tolist())
        if raw_lut is not None and "temperature_c" in raw_lut.columns
        else None
    )

    merged_dod_values: Optional[List[float]] = None
    merged_temp_values: Optional[List[float]] = None
    if include_raw_grid and raw_dod_values is not None:
        merged_dod_values = list(raw_dod_values)
    if include_raw_grid and raw_temp_values is not None:
        merged_temp_values = list(raw_temp_values)

    if dod_values is not None:
        merged_dod_values = (merged_dod_values or []) + list(dod_values)
    if temp_values_c is not None:
        merged_temp_values = (merged_temp_values or []) + list(temp_values_c)

    lut = build_market_realistic_lut(
        cfg,
        raw_lut=raw_lut,
        dod_values=merged_dod_values,
        temp_values_c=merged_temp_values,
        include_default_grid=include_default_grid,
    )
    dispatch = make_constraint_aware_dispatch(market, cfg, initial_soc=initial_soc, initial_soh=initial_soh)
    result = run_bess_financial_layer(dispatch, cfg, lut, initial_soc=initial_soc, initial_soh=initial_soh)
    summary = summarize_financial_result(result, cfg)
    telemetry = make_telemetry_table(result)

    artifacts = {
        "market": market,
        "lut": lut,
        "dispatch": dispatch,
        "result": result,
        "summary": summary,
        "telemetry": telemetry,
    }
    metadata = {
        "market_source": market_source,
        "raw_lut_source": raw_lut_path.name if raw_lut_path is not None else None,
    }
    return artifacts, metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run final market-realistic BESS simulation pipeline.")
    p.add_argument("--input-dir", default=".", help="Folder containing the input CSVs.")
    p.add_argument("--output-dir", default=".", help="Folder where output CSVs are written.")
    p.add_argument("--initial-soc", type=float, default=0.50)
    p.add_argument("--initial-soh", type=float, default=1.00)

    p.add_argument("--E-nom-MWh", type=float, default=2.0)
    p.add_argument("--P-max-MW", type=float, default=1.0)
    p.add_argument("--dt-h", type=float, default=0.25)
    p.add_argument("--soc-min", type=float, default=0.10)
    p.add_argument("--soc-max", type=float, default=0.90)
    p.add_argument("--c-rate-max", type=float, default=0.50)

    p.add_argument("--low-q", type=float, default=0.25)
    p.add_argument("--high-q", type=float, default=0.75)
    p.add_argument("--power-fraction", type=float, default=1.00)

    p.add_argument("--dod-grid", help="Comma-separated DoD values (e.g. 0.05,0.08,0.10)")
    p.add_argument("--dod-min", type=float, help="Minimum DoD when using range grid")
    p.add_argument("--dod-max", type=float, help="Maximum DoD when using range grid")
    p.add_argument("--dod-step", type=float, help="DoD step when using range grid")

    p.add_argument("--temp-grid", help="Comma-separated temperature values (e.g. 10,25,40)")
    p.add_argument("--temp-min", type=float, help="Minimum temperature when using range grid")
    p.add_argument("--temp-max", type=float, help="Maximum temperature when using range grid")
    p.add_argument("--temp-step", type=float, help="Temperature step when using range grid")

    p.add_argument("--no-default-grid", action="store_true", help="Do not include the default DoD/temperature grid")
    p.add_argument("--no-raw-grid", action="store_true", help="Do not merge grid points from the raw LUT")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    dod_values = _parse_float_list(args.dod_grid)
    temp_values = _parse_float_list(args.temp_grid)

    if dod_values is None and (args.dod_min is not None or args.dod_max is not None or args.dod_step is not None):
        if args.dod_min is None or args.dod_max is None or args.dod_step is None:
            raise ValueError("dod-min, dod-max, and dod-step are required when using a DoD range grid")
        dod_values = _build_range_grid(args.dod_min, args.dod_max, args.dod_step)

    if temp_values is None and (args.temp_min is not None or args.temp_max is not None or args.temp_step is not None):
        if args.temp_min is None or args.temp_max is None or args.temp_step is None:
            raise ValueError("temp-min, temp-max, and temp-step are required when using a temperature range grid")
        temp_values = _build_range_grid(args.temp_min, args.temp_max, args.temp_step)

    cfg = BESSConfig(
        E_nom_MWh=args.E_nom_MWh,
        P_max_MW=args.P_max_MW,
        dt_h=args.dt_h,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        c_rate_max=args.c_rate_max,
        low_price_quantile=args.low_q,
        high_price_quantile=args.high_q,
        dispatch_power_fraction=args.power_fraction,
    )

    manifest = run_pipeline(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        initial_soc=args.initial_soc,
        initial_soh=args.initial_soh,
        cfg=cfg,
        dod_values=dod_values,
        temp_values_c=temp_values,
        include_default_grid=not args.no_default_grid,
        include_raw_grid=not args.no_raw_grid,
    )

    s = manifest["summary"]
    d = manifest["derived_constraints"]

    print("\nBESS market-realistic simulation completed.")
    print(f"Market source: {manifest['market_source']}")
    print(f"Raw LUT source: {manifest['raw_lut_source']}")
    print(f"E_nom_MWh: {cfg.E_nom_MWh:.4g}")
    print(f"P_max_MW: {cfg.P_max_MW:.4g}")
    print(f"Effective C-rate max: {d['effective_c_rate_max']:.4g}C")
    print(f"Max DoD per {cfg.dt_h:g}h step: {d['max_dod_per_step']:.4g} ({100*d['max_dod_per_step']:.2f}%)")
    print(f"Feasible rows: {int(s['feasible_rows'])}/{int(s['rows'])} ({100.0 * float(s['feasible_share']):.2f}%)")
    print(f"Max observed C-rate: {float(s['max_C_rate_observed']):.4g}C")
    print(f"Max observed DoD step: {float(s['max_DoD_step_observed']):.4g} ({100*float(s['max_DoD_step_observed']):.2f}%)")
    print(f"Total gross margin EUR: {float(s['total_gross_margin_eur']):,.2f}")
    print(f"Total degradation cost EUR: {float(s['total_degradation_cost_eur']):,.2f}")
    print(f"Total penalty EUR: {float(s['total_penalty_eur']):,.2f}")
    print(f"Total net profit EUR: {float(s['total_net_profit_eur']):,.2f}")
    print(f"Final SoC: {float(s['final_SoC']):.4f}")
    print(f"Final SoH: {float(s['final_SoH']):.6f}")
    print("\nFiles written:")
    for key, path in manifest["outputs"].items():
        print(f" - {key}: {path}")


if __name__ == "__main__":
    main()
