"""Shared price loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bess_optimization.forecasting.dam_15min_forecast import load_price_history
from bess_optimization.paths import CLEANED_DATA_DIR, DEFAULT_PRICE_SIGNALS_PATH


def display_path(path: Path, root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def list_price_files(cleaned_data_dir: Path = CLEANED_DATA_DIR) -> list[Path]:
    files: list[Path] = []
    if DEFAULT_PRICE_SIGNALS_PATH.exists():
        files.append(DEFAULT_PRICE_SIGNALS_PATH)
    if cleaned_data_dir.exists():
        files.extend(
            p
            for p in sorted(cleaned_data_dir.glob("*.csv"))
            if p not in files and ("price" in p.name.lower() or "signal" in p.name.lower())
        )
    return files


def load_price_series(
    csv_path: Path,
    timestamp_col: str | None = None,
    price_col: str | None = None,
    allow_hourly_upsampling: bool = False,
) -> pd.Series:
    history = load_price_history(
        csv_path,
        timestamp_col=timestamp_col,
        price_col=price_col,
        allow_hourly_upsampling=allow_hourly_upsampling,
    )
    prices = (
        history.frame.assign(timestamp=pd.to_datetime(history.frame["timestamp"]))
        .groupby("timestamp")["price_eur_mwh"]
        .mean()
        .sort_index()
    )
    prices.name = "price_eur_mwh"
    return prices


def available_dates(csv_path: Path) -> list[pd.Timestamp]:
    prices = load_price_series(csv_path)
    return [pd.Timestamp(d) for d in sorted(pd.Series(prices.index.date).unique())]


def available_years(csv_path: Path) -> list[int]:
    prices = load_price_series(csv_path)
    return sorted(int(year) for year in pd.Series(prices.index.year).unique())


def load_price_signal_day(
    csv_path: Path = DEFAULT_PRICE_SIGNALS_PATH,
    target_date: str = "2025-01-10",
    dt: float = 0.25,
    fill_missing: bool = True,
) -> pd.Series:
    prices = load_price_series(csv_path)
    target_day = pd.Timestamp(target_date).floor("D")
    same_day = prices[prices.index.normalize() == target_day]
    if same_day.empty:
        raise ValueError(f"No price data exists for {target_day.date().isoformat()}.")

    step_minutes = int(round(float(dt) * 60))
    periods = int(round(24 * 60 / step_minutes))
    target_index = pd.date_range(target_day, periods=periods, freq=f"{step_minutes}min")
    if fill_missing:
        day_prices = (
            same_day.reindex(same_day.index.union(target_index))
            .sort_index()
            .ffill()
            .bfill()
            .reindex(target_index)
        )
    else:
        day_prices = same_day.reindex(target_index)
    if day_prices.isna().any():
        missing = int(day_prices.isna().sum())
        raise ValueError(f"{missing} 15-minute prices are missing for {target_day.date()}.")
    day_prices.name = "price_eur_mwh"
    return day_prices


def load_price_signal_year(
    csv_path: Path = DEFAULT_PRICE_SIGNALS_PATH,
    year: int = 2025,
    dt: float = 0.25,
) -> pd.Series:
    prices = load_price_series(csv_path)
    step_minutes = int(round(float(dt) * 60))
    year_start = pd.Timestamp(year=int(year), month=1, day=1)
    year_end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
    target_index = pd.date_range(
        start=year_start,
        end=year_end - pd.Timedelta(minutes=step_minutes),
        freq=f"{step_minutes}min",
    )
    year_prices = prices.reindex(target_index)
    if year_prices.isna().any():
        missing = int(year_prices.isna().sum())
        raise ValueError(f"{missing} prices are missing for {year}.")
    year_prices.name = "price_eur_mwh"
    return year_prices
