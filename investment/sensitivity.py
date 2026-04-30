from __future__ import annotations

import numpy as np
import pandas as pd

from .investment_model import compute_cashflows, compute_irr, compute_npv


def npv_vs_discount_rate(cashflows, rates=None) -> pd.DataFrame:
    rates = np.linspace(0.0, 0.20, 41) if rates is None else np.asarray(rates, dtype=float)
    return pd.DataFrame(
        {
            "discount_rate": rates,
            "npv": [compute_npv(cashflows, rate) for rate in rates],
        }
    )


def irr_vs_capex(capex_range, base_inputs) -> pd.DataFrame:
    capex_values = np.asarray(capex_range, dtype=float)
    annual_profit = float(base_inputs.get("annual_profit", 0.0))
    capacity = float(base_inputs.get("battery_capacity_MWh", 1.0))
    rows = []
    for capex_per_mwh in capex_values:
        inputs = dict(base_inputs)
        inputs["CAPEX_per_MWh"] = float(capex_per_mwh)
        inputs["battery_capacity_MWh"] = capacity
        cashflows = compute_cashflows(annual_profit, inputs)
        rows.append({"CAPEX_per_MWh": float(capex_per_mwh), "irr": compute_irr(cashflows)})
    return pd.DataFrame(rows)


def profit_vs_degradation_multiplier(base_profit=0.0, degradation_cost=0.0, multipliers=None) -> pd.DataFrame:
    multipliers = np.linspace(0.5, 2.0, 31) if multipliers is None else np.asarray(multipliers, dtype=float)
    base_profit = float(base_profit)
    degradation_cost = float(degradation_cost)
    return pd.DataFrame(
        {
            "degradation_multiplier": multipliers,
            "profit": base_profit - degradation_cost * (multipliers - 1.0),
        }
    )
