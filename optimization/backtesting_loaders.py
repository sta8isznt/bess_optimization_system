"""Load cleaned optimizer input data."""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CLEANED_DATA_DIR = BASE_DIR / "data" / "cleaned_data"
DEFAULT_PRICE_SIGNALS_PATH = CLEANED_DATA_DIR / "price_signals_15m.csv"
DEFAULT_DEGRADATION_LUT_PATH = CLEANED_DATA_DIR / "Reduced_LUT_Final.csv"


def load_price_signal_day(
    csv_path: Path = DEFAULT_PRICE_SIGNALS_PATH,
    target_date: str = "2025-01-10",
    dt: float = 0.25,
) -> pd.Series:
    """Load one day of 15-minute DAM prices from price_signals_15m.csv."""
    df = pd.read_csv(csv_path)
    df["DELIVERY_MTU"] = pd.to_datetime(df["DELIVERY_MTU"])
    prices = df.groupby("DELIVERY_MTU")["DAM_Price_EUR_MWh"].mean().sort_index()

    day_start = pd.Timestamp(target_date).floor("D")
    step_minutes = int(round(dt * 60))
    periods = int(round(24 * 60 / step_minutes))
    target_index = pd.date_range(
        start=day_start,
        periods=periods,
        freq=f"{step_minutes}min",
    )

    prices = (
        prices.reindex(prices.index.union(target_index))
        .sort_index()
        .ffill()
        .reindex(target_index)
    )
    prices.name = "price_eur_mwh"
    return prices


def load_price_signal_year(
    csv_path: Path = DEFAULT_PRICE_SIGNALS_PATH,
    year: int = 2025,
    dt: float = 0.25,
) -> pd.Series:
    """Load one calendar year of 15-minute DAM prices from price_signals_15m.csv."""
    df = pd.read_csv(csv_path)
    df["DELIVERY_MTU"] = pd.to_datetime(df["DELIVERY_MTU"])
    prices = df.groupby("DELIVERY_MTU")["DAM_Price_EUR_MWh"].mean().sort_index()

    step_minutes = int(round(dt * 60))
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year + 1, month=1, day=1)
    target_index = pd.date_range(
        start=year_start,
        end=year_end - pd.Timedelta(minutes=step_minutes),
        freq=f"{step_minutes}min",
    )

    prices = prices.reindex(target_index)
    if prices.isna().any():
        missing_count = int(prices.isna().sum())
        raise ValueError(f"{missing_count} prices are missing for {year}.")

    prices.name = "price_eur_mwh"
    return prices


def load_degradation_lut_curve(
    csv_path: Path = DEFAULT_DEGRADATION_LUT_PATH,
    temperature_c: float = 25.0,
) -> tuple[list[float], list[float]]:
    """Load degradation LUT as energy points and absolute EUR costs."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    filtered = df[df["temperature_c"].sub(float(temperature_c)).abs() < 1e-9].copy()
    filtered = filtered.sort_values("energy")

    energy_points = filtered["energy"].astype(float).tolist()
    cost_points = (
        filtered["energy"] * filtered["deg_cost_eur_per_MWh_throughput"]
    ).astype(float).tolist()

    energy_points.insert(0, 0.0)
    cost_points.insert(0, 0.0)
    return energy_points, cost_points
