#!/usr/bin/env python3
"""
BESS market pipeline using PyBaMM-calibrated degradation LUT.

Purpose
-------
This is the runtime market/synthetic-data pipeline. It DOES NOT run PyBaMM.
It consumes the offline-calibrated LUT:

    degradation_lut_pybamm_calibrated.csv

and then creates physically feasible BESS data:

    market input -> constraint-aware dispatch -> SoC/SoH -> degradation cost -> PnL CSVs

Use this after running:
    python offline_pybamm_benchmark_calibrator.py --input-dir . --output-dir .

Outputs
-------
bess_market_input_pybamm_calibrated.csv
bess_dispatch_pybamm_calibrated.csv
bess_financial_result_pybamm_calibrated.csv
bess_financial_summary_pybamm_calibrated.csv
bess_telemetry_pybamm_calibrated.csv
bess_pybamm_calibrated_visual_report.html

Run
---
python bess_pipeline_pybamm_calibrated.py --input-dir . --output-dir .
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
class BESSConfig:
    # Final project default: 2 MWh / 1 MW = 0.5C = 2-hour battery.
    E_nom_MWh: float = 2.0
    P_max_MW: float = 1.0
    dt_h: float = 0.25

    soc_min: float = 0.10
    soc_max: float = 0.90
    soh_eol: float = 0.80
    c_rate_max: float = 0.50

    eta_charge: float = 0.92
    eta_discharge: float = 0.92

    T_cell_min_C: float = 5.0
    T_cell_max_C: float = 40.0
    thermal_power_coeff_C_per_Crate: float = 6.0

    replacement_cost_eur_per_MWh_capacity: float = 60_000.0

    soc_guard_band: float = 0.002
    low_price_quantile: float = 0.25
    high_price_quantile: float = 0.75
    dispatch_power_fraction: float = 1.00

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
            "bess_dispatch_fixed.csv",
            "bess_financial_result_fixed.csv",
            "bess_dispatch.csv",
            "bess_financial_result.csv",
        ],
    )

    if source is not None:
        df = pd.read_csv(source)
        if "timestamp" not in df.columns:
            raise ValueError(f"{source.name} must contain timestamp.")
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

        return out.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True), source.name

    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2030-01-01", periods=96, freq="15min")
    price = 80 + 35 * np.sin(np.linspace(-np.pi, np.pi, len(timestamps))) + 10 * rng.normal(size=len(timestamps))
    return pd.DataFrame({"timestamp": timestamps, "price_eur_per_MWh": price, "T_amb_C": 25.0}), "synthetic_generated_market"


def load_calibrated_lut(input_dir: Path) -> Tuple[pd.DataFrame, str]:
    p = _find_first_existing(
        input_dir,
        [
            "degradation_lut_pybamm_calibrated.csv",
            "degradation_lut_market_realistic.csv",
            "degradation_lut_market_realistic_dense_dod.csv",
        ],
    )
    if p is None:
        raise FileNotFoundError(
            "No calibrated LUT found. Run offline_pybamm_benchmark_calibrator.py first, "
            "or place degradation_lut_market_realistic.csv in the folder."
        )
    return pd.read_csv(p), p.name


def estimate_cell_temperature_C(T_amb_C: float, P_ch_MW: float, P_dis_MW: float, cfg: BESSConfig) -> float:
    c_rate = max(P_ch_MW, P_dis_MW) / cfg.E_nom_MWh
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


def make_constraint_aware_dispatch(market: pd.DataFrame, cfg: BESSConfig, initial_soc: float, initial_soh: float) -> pd.DataFrame:
    out = market.copy().sort_values("timestamp").reset_index(drop=True)
    prices = out["price_eur_per_MWh"].astype(float)
    low_thr = float(prices.quantile(cfg.low_price_quantile))
    high_thr = float(prices.quantile(cfg.high_price_quantile))
    base_power = min(cfg.P_max_MW, cfg.p_c_rate_limit_MW) * cfg.dispatch_power_fraction

    soc = float(initial_soc)
    soh = float(initial_soh)
    rows = []

    for _, row in out.iterrows():
        price = float(row["price_eur_per_MWh"])
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

        soc_next = float(np.clip(update_soc(soc, soh, P_ch, P_dis, cfg), cfg.soc_min, cfg.soc_max))

        rows.append({
            "P_ch_MW": P_ch,
            "P_dis_MW": P_dis,
            "action": action,
            "dispatch_low_price_threshold": low_thr,
            "dispatch_high_price_threshold": high_thr,
            "simulated_SoC_before": soc,
            "simulated_SoC_after": soc_next,
        })
        soc = soc_next

    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def _nearest_lut_cost(dod: float, temp_c: float, lut: pd.DataFrame, cfg: BESSConfig) -> Tuple[float, str]:
    if dod <= 0:
        return 0.0, "zero_dispatch"

    if "deg_cost_final" in lut.columns:
        col = "deg_cost_final"
    elif "deg_cost_eur_per_MWh_throughput" in lut.columns:
        col = "deg_cost_eur_per_MWh_throughput"
    else:
        raise ValueError("LUT needs deg_cost_final or deg_cost_eur_per_MWh_throughput.")

    valid = lut.dropna(subset=["dod", "temperature_c", col]).copy()
    if "valid" in valid.columns:
        valid = valid[pd.to_numeric(valid["valid"], errors="coerce").fillna(0) > 0]

    if valid.empty:
        raise ValueError("No valid degradation LUT rows found.")

    dod_scale = max(valid["dod"].max() - valid["dod"].min(), 1e-12)
    temp_scale = max(valid["temperature_c"].max() - valid["temperature_c"].min(), 1e-12)
    dist = ((valid["dod"] - dod) / dod_scale) ** 2 + ((valid["temperature_c"] - temp_c) / temp_scale) ** 2
    idx = dist.idxmin()
    return float(valid.loc[idx, col]), f"nearest_{col}"


def physical_violations(soc, soc_next, soh, soh_next, P_ch_MW, P_dis_MW, T_cell_C, cfg: BESSConfig) -> List[str]:
    v = []
    tol = 1e-9
    if P_ch_MW < -tol or P_dis_MW < -tol:
        v.append("negative_power")
    if P_ch_MW > tol and P_dis_MW > tol:
        v.append("simultaneous_charge_discharge")
    if P_ch_MW > cfg.P_max_MW + tol:
        v.append("charge_power_exceeded")
    if P_dis_MW > cfg.P_max_MW + tol:
        v.append("discharge_power_exceeded")

    c_rate = max(P_ch_MW, P_dis_MW) / cfg.E_nom_MWh
    if c_rate > cfg.effective_c_rate_max + tol:
        v.append("C_rate_exceeded")

    dod = max(P_ch_MW, P_dis_MW) * cfg.dt_h / cfg.E_nom_MWh
    if dod > cfg.max_dod_per_step + tol:
        v.append("DoD_step_exceeded")

    if soc < cfg.soc_min - tol or soc > cfg.soc_max + tol:
        v.append("SoC_current_out_of_bounds")
    if soc_next < cfg.soc_min - tol or soc_next > cfg.soc_max + tol:
        v.append("SoC_next_out_of_bounds")

    if T_cell_C < cfg.T_cell_min_C - tol:
        v.append("cell_temperature_too_low")
    if T_cell_C > cfg.T_cell_max_C + tol:
        v.append("cell_temperature_too_high")
    if soh_next > soh + 1e-12:
        v.append("SoH_increased")
    if soh_next < cfg.soh_eol - tol:
        v.append("EOL_reached")
    return v


def infeasibility_penalty_eur(soc, soc_next, soh, soh_next, P_ch_MW, P_dis_MW, T_cell_C, cfg: BESSConfig) -> float:
    penalty = 0.0
    for x in (soc, soc_next):
        penalty += cfg.penalty_soc * max(0.0, cfg.soc_min - x) ** 2
        penalty += cfg.penalty_soc * max(0.0, x - cfg.soc_max) ** 2

    penalty += cfg.penalty_power * max(0.0, P_ch_MW - cfg.P_max_MW) ** 2
    penalty += cfg.penalty_power * max(0.0, P_dis_MW - cfg.P_max_MW) ** 2
    penalty += cfg.penalty_power * max(0.0, -P_ch_MW) ** 2
    penalty += cfg.penalty_power * max(0.0, -P_dis_MW) ** 2

    if P_ch_MW > 0 and P_dis_MW > 0:
        penalty += cfg.penalty_simultaneous * (min(P_ch_MW, P_dis_MW) / max(cfg.P_max_MW, 1e-12)) ** 2

    c_rate = max(P_ch_MW, P_dis_MW) / cfg.E_nom_MWh
    penalty += cfg.penalty_c_rate * max(0.0, c_rate - cfg.effective_c_rate_max) ** 2

    dod = max(P_ch_MW, P_dis_MW) * cfg.dt_h / cfg.E_nom_MWh
    penalty += cfg.penalty_c_rate * max(0.0, dod - cfg.max_dod_per_step) ** 2

    penalty += cfg.penalty_temperature * max(0.0, cfg.T_cell_min_C - T_cell_C) ** 2
    penalty += cfg.penalty_temperature * max(0.0, T_cell_C - cfg.T_cell_max_C) ** 2

    penalty += cfg.penalty_soh * max(0.0, soh_next - soh - 1e-12) ** 2
    penalty += cfg.penalty_soh * max(0.0, cfg.soh_eol - soh_next) ** 2
    return float(penalty)


def run_financial_layer(dispatch: pd.DataFrame, lut: pd.DataFrame, cfg: BESSConfig, initial_soc: float, initial_soh: float) -> pd.DataFrame:
    out = dispatch.copy().sort_values("timestamp").reset_index(drop=True)

    soc = float(initial_soc)
    soh = float(initial_soh)
    rows = []

    for _, row in out.iterrows():
        P_ch = float(row["P_ch_MW"])
        P_dis = float(row["P_dis_MW"])
        price = float(row["price_eur_per_MWh"])
        T_amb = float(row["T_amb_C"])
        T_cell = estimate_cell_temperature_C(T_amb, P_ch, P_dis, cfg)

        soc_next = update_soc(soc, soh, P_ch, P_dis, cfg)
        dod_step = timestep_dod(P_ch, P_dis, cfg)
        throughput_MWh = max(P_ch, P_dis) * cfg.dt_h

        deg_cost_per_mwh, deg_source = _nearest_lut_cost(dod_step, T_cell, lut, cfg)
        degradation_cost = deg_cost_per_mwh * throughput_MWh
        delta_soh = soh_drop_from_degradation_cost(degradation_cost, cfg)
        soh_next = max(0.0, soh - delta_soh)

        charge_cost = P_ch * cfg.dt_h * price
        discharge_revenue = P_dis * cfg.dt_h * price
        gross_margin = discharge_revenue - charge_cost

        violations = physical_violations(soc, soc_next, soh, soh_next, P_ch, P_dis, T_cell, cfg)
        penalty = infeasibility_penalty_eur(soc, soc_next, soh, soh_next, P_ch, P_dis, T_cell, cfg)
        net_profit = gross_margin - degradation_cost - penalty

        rows.append({
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
        })

        soc = float(np.clip(soc_next, cfg.soc_min, cfg.soc_max))
        soh = soh_next

    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def summarize(result: pd.DataFrame, cfg: BESSConfig) -> pd.DataFrame:
    all_v = []
    for x in result["violations"]:
        if isinstance(x, list):
            all_v.extend(x)
        elif isinstance(x, str) and x not in ("", "[]"):
            all_v.append(x)
    counts = pd.Series(all_v).value_counts() if all_v else pd.Series(dtype=int)

    s = {
        "rows": len(result),
        "feasible_rows": int(result["is_feasible"].sum()),
        "feasible_share": float(result["is_feasible"].mean()),
        "E_nom_MWh": cfg.E_nom_MWh,
        "P_max_MW": cfg.P_max_MW,
        "effective_c_rate_max": cfg.effective_c_rate_max,
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
    out = pd.DataFrame([s])
    for name, count in counts.items():
        out[f"violation__{name}"] = int(count)
    return out


def make_telemetry(result: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp", "price_eur_per_MWh", "T_amb_C", "T_cell_C_used",
        "P_ch_MW", "P_dis_MW", "action",
        "SoC", "SoC_next", "SoH", "SoH_next", "delta_SoH",
        "C_rate", "DoD_step", "throughput_MWh",
        "gross_margin_eur", "degradation_cost_eur", "net_profit_eur",
        "is_feasible", "violations",
    ]
    return result[[c for c in cols if c in result.columns]].copy()


def export_html(result: pd.DataFrame, lut: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
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
    fig.update_layout(title="Dispatch vs price", hovermode="x unified")
    figs.append(fig)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SoC"], name="SoC", mode="lines+markers"), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["SoH"], name="SoH", mode="lines+markers"), secondary_y=True)
    fig.update_layout(title="SoC / SoH trend", hovermode="x unified")
    figs.append(fig)

    fig = go.Figure()
    for col in ["cum_gross_margin_eur", "cum_degradation_cost_eur", "cum_infeasibility_penalty_eur", "cum_net_profit_eur"]:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df[col], name=col, mode="lines+markers"))
    fig.update_layout(title="Cumulative economics", yaxis_title="EUR", hovermode="x unified")
    figs.append(fig)

    fig = px.line(
        lut,
        x="dod",
        y="deg_cost_surface_full" if "deg_cost_surface_full" in lut.columns else "deg_cost_smoothed",
        color="temperature_c",
        markers=True,
        title="PyBaMM-calibrated full degradation surface",
    )
    figs.append(fig)

    valid = lut[lut["valid"] > 0].copy() if "valid" in lut.columns else lut.copy()
    fig = px.line(valid, x="dod", y="deg_cost_final", color="temperature_c", markers=True, title="Optimizer-valid LUT")
    figs.append(fig)

    html = [
        "<html><head><meta charset='utf-8'><title>BESS PyBaMM-Calibrated Runtime Report</title></head><body>",
        "<h1>BESS PyBaMM-Calibrated Runtime Report</h1>",
        "<p>Runtime pipeline consumes the offline PyBaMM-calibrated LUT. It does not run PyBaMM online.</p>",
        "<h2>Summary</h2>",
        summary.T.reset_index().rename(columns={"index": "metric", 0: "value"}).to_html(index=False),
    ]
    for i, fig in enumerate(figs):
        html.append(pio.to_html(fig, full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    html.append("</body></html>")

    path = output_dir / "bess_pybamm_calibrated_visual_report.html"
    path.write_text("\n".join(html), encoding="utf-8")
    return path


def run_pipeline(input_dir: Path, output_dir: Path, cfg: BESSConfig, initial_soc: float, initial_soh: float) -> Dict[str, object]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    market, market_source = load_market_input(input_dir)
    lut, lut_source = load_calibrated_lut(input_dir)

    dispatch = make_constraint_aware_dispatch(market, cfg, initial_soc, initial_soh)
    result = run_financial_layer(dispatch, lut, cfg, initial_soc, initial_soh)
    summary = summarize(result, cfg)
    telemetry = make_telemetry(result)

    outputs = {
        "market": output_dir / "bess_market_input_pybamm_calibrated.csv",
        "dispatch": output_dir / "bess_dispatch_pybamm_calibrated.csv",
        "result": output_dir / "bess_financial_result_pybamm_calibrated.csv",
        "summary": output_dir / "bess_financial_summary_pybamm_calibrated.csv",
        "telemetry": output_dir / "bess_telemetry_pybamm_calibrated.csv",
        "manifest": output_dir / "bess_pybamm_calibrated_manifest.json",
    }

    market.to_csv(outputs["market"], index=False)
    dispatch.to_csv(outputs["dispatch"], index=False)
    result.to_csv(outputs["result"], index=False)
    summary.to_csv(outputs["summary"], index=False)
    telemetry.to_csv(outputs["telemetry"], index=False)

    report = export_html(result, lut, summary, output_dir)
    if report:
        outputs["html_report"] = report

    manifest = {
        "pipeline": "bess_pipeline_pybamm_calibrated",
        "market_source": market_source,
        "lut_source": lut_source,
        "config": asdict(cfg),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "summary": summary.iloc[0].to_dict(),
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BESS runtime pipeline with PyBaMM-calibrated LUT.")
    p.add_argument("--input-dir", default=".")
    p.add_argument("--output-dir", default=".")
    p.add_argument("--initial-soc", type=float, default=0.50)
    p.add_argument("--initial-soh", type=float, default=1.00)

    p.add_argument("--E-nom-MWh", type=float, default=2.0)
    p.add_argument("--P-max-MW", type=float, default=1.0)
    p.add_argument("--dt-h", type=float, default=0.25)
    p.add_argument("--soc-min", type=float, default=0.10)
    p.add_argument("--soc-max", type=float, default=0.90)
    p.add_argument("--c-rate-max", type=float, default=0.50)
    p.add_argument("--power-fraction", type=float, default=1.00)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BESSConfig(
        E_nom_MWh=args.E_nom_MWh,
        P_max_MW=args.P_max_MW,
        dt_h=args.dt_h,
        soc_min=args.soc_min,
        soc_max=args.soc_max,
        c_rate_max=args.c_rate_max,
        dispatch_power_fraction=args.power_fraction,
    )

    manifest = run_pipeline(Path(args.input_dir), Path(args.output_dir), cfg, args.initial_soc, args.initial_soh)

    s = manifest["summary"]
    print("\nBESS PyBaMM-calibrated runtime pipeline completed.")
    print(f"Market source: {manifest['market_source']}")
    print(f"LUT source: {manifest['lut_source']}")
    print(f"Feasible rows: {int(s['feasible_rows'])}/{int(s['rows'])} ({100*float(s['feasible_share']):.2f}%)")
    print(f"Max DoD observed: {100*float(s['max_DoD_step_observed']):.2f}%")
    print(f"Total gross margin EUR: {float(s['total_gross_margin_eur']):,.2f}")
    print(f"Total degradation cost EUR: {float(s['total_degradation_cost_eur']):,.2f}")
    print(f"Total penalty EUR: {float(s['total_penalty_eur']):,.2f}")
    print(f"Total net profit EUR: {float(s['total_net_profit_eur']):,.2f}")
    print("\nFiles written:")
    for k, v in manifest["outputs"].items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
