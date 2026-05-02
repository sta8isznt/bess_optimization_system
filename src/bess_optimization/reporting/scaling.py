"""Shared base-module to park-scale normalization helpers."""

from __future__ import annotations

import pandas as pd


SCALED_COLUMNS = {
    "p_buy_mw": "park_p_buy_mw",
    "p_sell_mw": "park_p_sell_mw",
    "net_export_mw": "park_net_export_mw",
    "buy_energy_mwh": "park_buy_energy_mwh",
    "sell_energy_mwh": "park_sell_energy_mwh",
    "soc_mwh": "park_soc_mwh",
    "gross_revenue_eur": "park_gross_revenue_eur",
    "gross_purchase_eur": "park_gross_purchase_eur",
    "degradation_cost_eur": "park_degradation_cost_eur",
    "interval_profit_eur": "park_interval_profit_eur",
}


def apply_scale(
    schedule: pd.DataFrame,
    summary: dict,
    params: dict,
    scale_capacity_mw: float | None,
) -> tuple[pd.DataFrame, dict]:
    p_max = float(params["p_max"])
    installed_mw = float(scale_capacity_mw) if scale_capacity_mw else p_max
    if installed_mw <= 0:
        raise ValueError("Installed capacity must be positive.")
    scale_factor = installed_mw / p_max

    scaled = schedule.copy()
    for source_col, target_col in SCALED_COLUMNS.items():
        if source_col in scaled.columns:
            scaled[target_col] = scaled[source_col] * scale_factor

    out = dict(summary)
    out["installed_capacity_mw"] = installed_mw
    out["installed_energy_capacity_mwh"] = float(params["e_max"]) * scale_factor
    out["scale_factor"] = scale_factor
    out["park_net_profit_eur"] = float(out["net_profit_eur"] * scale_factor)
    out["park_gross_revenue_eur"] = float(out["gross_revenue_eur"] * scale_factor)
    out["park_gross_purchase_eur"] = float(out["gross_purchase_eur"] * scale_factor)
    out["park_degradation_cost_eur"] = float(out["degradation_cost_eur"] * scale_factor)
    out["park_buy_energy_mwh"] = float(out["buy_energy_mwh"] * scale_factor)
    out["park_sell_energy_mwh"] = float(out["sell_energy_mwh"] * scale_factor)
    if "total_throughput_mwh" in out:
        out["park_total_throughput_mwh"] = float(out["total_throughput_mwh"] * scale_factor)
    return scaled, out


def summary_value(summary: dict, base_key: str, park_key: str) -> float:
    if float(summary.get("scale_factor", 1.0)) != 1.0 and park_key in summary:
        return float(summary[park_key])
    return float(summary.get(base_key, 0.0))
