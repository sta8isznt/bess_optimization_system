from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def load_financial_inputs() -> dict[str, float | int]:
    return {
        "CAPEX_per_MWh": 400000.0,
        "OPEX_percent": 0.02,
        "lifetime_years": 10,
        "discount_rate": 0.08,
        "battery_capacity_MWh": 2.0,
    }


def _find_metric_value(df: pd.DataFrame, names: Sequence[str]) -> float | None:
    if df.empty:
        return None

    normalized_names = {name.lower().replace(" ", "_") for name in names}
    lower_cols = {str(col).lower().strip(): col for col in df.columns}

    if "metric" in lower_cols and "value" in lower_cols:
        metric_col = lower_cols["metric"]
        value_col = lower_cols["value"]
        work = df[[metric_col, value_col]].copy()
        work["_metric"] = work[metric_col].astype(str).str.lower().str.replace(" ", "_")
        match = work[work["_metric"].isin(normalized_names)]
        if not match.empty:
            return float(pd.to_numeric(match[value_col], errors="coerce").dropna().iloc[0])

    for col in df.columns:
        key = str(col).lower().strip().replace(" ", "_")
        if key in normalized_names:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            if not values.empty:
                return float(values.iloc[0] if len(df) == 1 else values.sum())
    return None


def compute_annual_profit(financial_summary_df: pd.DataFrame) -> float:
    total_profit = _find_metric_value(
        financial_summary_df,
        [
            "daily_profit",
            "net_profit_eur",
            "profit_eur",
            "net_profit",
            "park_net_profit_eur",
        ],
    )
    if total_profit is None:
        numeric = financial_summary_df.select_dtypes(include="number")
        total_profit = float(numeric.sum().sum()) if not numeric.empty else 0.0

    days = _find_metric_value(financial_summary_df, ["days", "num_days", "n_days"])
    if days and days > 1:
        return float(total_profit / days * 365.0)
    return float(total_profit * 365.0)


def compute_cashflows(annual_profit: float, inputs: dict) -> list[float]:
    capex = float(inputs["CAPEX_per_MWh"]) * float(inputs["battery_capacity_MWh"])
    opex = capex * float(inputs["OPEX_percent"])
    lifetime = int(inputs["lifetime_years"])
    return [-capex] + [float(annual_profit) - opex for _ in range(lifetime)]


def compute_payback(capex: float, annual_profit: float) -> float:
    if annual_profit <= 0:
        return float("inf")
    return float(capex / annual_profit)


def compute_npv(cashflows: Sequence[float], discount_rate: float) -> float:
    rate = float(discount_rate)
    return float(sum(float(cf) / ((1.0 + rate) ** year) for year, cf in enumerate(cashflows)))


def compute_irr(cashflows: Sequence[float]) -> float:
    values = [float(cf) for cf in cashflows]
    if not values or all(cf >= 0 for cf in values) or all(cf <= 0 for cf in values):
        return float("nan")

    def npv_at(rate: float) -> float:
        return sum(cf / ((1.0 + rate) ** year) for year, cf in enumerate(values))

    low, high = -0.95, 10.0
    npv_low, npv_high = npv_at(low), npv_at(high)
    if npv_low * npv_high > 0:
        return float("nan")

    for _ in range(100):
        mid = (low + high) / 2.0
        npv_mid = npv_at(mid)
        if abs(npv_mid) < 1e-7:
            return float(mid)
        if npv_low * npv_mid <= 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
    return float((low + high) / 2.0)


def compute_investment_metrics(df: pd.DataFrame, inputs: dict) -> dict[str, float]:
    annual_profit = compute_annual_profit(df)
    capex = float(inputs["CAPEX_per_MWh"]) * float(inputs["battery_capacity_MWh"])
    opex = capex * float(inputs["OPEX_percent"])
    cashflows = compute_cashflows(annual_profit, inputs)
    return {
        "annual_profit": float(annual_profit),
        "capex": float(capex),
        "opex": float(opex),
        "payback": compute_payback(capex, annual_profit),
        "npv": compute_npv(cashflows, float(inputs["discount_rate"])),
        "irr": compute_irr(cashflows),
    }


def metric_from_summary(df: pd.DataFrame, names: Sequence[str], default: float = 0.0) -> float:
    value = _find_metric_value(df, names)
    return default if value is None else float(value)
