"""Shared forecast-settlement cashflow helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aligned_price_values(
    prices: pd.Series,
    timestamps: pd.Series,
    label: str,
) -> np.ndarray:
    series = prices.copy()
    series.index = pd.to_datetime(series.index)
    series = series.groupby(series.index).mean().sort_index()
    aligned = series.reindex(pd.to_datetime(timestamps))
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(f"{missing} {label} prices could not be aligned to schedule.")
    return aligned.to_numpy(dtype=float)


def settle_schedule_on_actual_prices(
    forecast_schedule: pd.DataFrame,
    actual_prices: pd.Series,
    forecast_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    settled = forecast_schedule.copy()
    settled["timestamp"] = pd.to_datetime(settled["timestamp"])
    settled["forecast_price_eur_mwh"] = settled["price_eur_mwh"].astype(float)
    settled["actual_price_eur_mwh"] = aligned_price_values(
        actual_prices,
        settled["timestamp"],
        "actual",
    )
    settled["forecast_gross_revenue_eur"] = settled["gross_revenue_eur"]
    settled["forecast_gross_purchase_eur"] = settled["gross_purchase_eur"]
    settled["forecast_interval_profit_eur"] = settled["interval_profit_eur"]

    if forecast_frame is not None and "forecast_reason" in forecast_frame.columns:
        reasons = forecast_frame[["timestamp", "forecast_reason"]].copy()
        reasons["timestamp"] = pd.to_datetime(reasons["timestamp"])
        settled = settled.merge(reasons, on="timestamp", how="left")

    settled["price_error_eur_mwh"] = (
        settled["forecast_price_eur_mwh"] - settled["actual_price_eur_mwh"]
    )
    settled["actual_gross_revenue_eur"] = (
        settled["actual_price_eur_mwh"] * settled["sell_energy_mwh"]
    )
    settled["actual_gross_purchase_eur"] = (
        settled["actual_price_eur_mwh"] * settled["buy_energy_mwh"]
    )
    settled["actual_interval_profit_eur"] = (
        settled["actual_gross_revenue_eur"]
        - settled["actual_gross_purchase_eur"]
        - settled["degradation_cost_eur"]
    )
    settled["actual_minus_forecast_profit_eur"] = (
        settled["actual_interval_profit_eur"]
        - settled["forecast_interval_profit_eur"]
    )
    return settled
