from pathlib import Path
import warnings

import pandas as pd

try:
    from .preproc import henexing
except ImportError:  # Allows running this file from inside optimization/.
    from preproc import henexing


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_2025_DAM_DIR = BASE_DIR / "Results_2025" / "DAM"
DEFAULT_PRICE_SIGNALS_PATH = BASE_DIR / "price_signals_15m.csv"
DEFAULT_DEGRADATION_LUT_PATH = BASE_DIR / "Reduced_LUT_Final.csv"


def _read_dam_excel(file_path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Workbook contains no default style.*",
            category=UserWarning,
        )
        return pd.read_excel(file_path, usecols=["DELIVERY_MTU", "MCP"])


def default_2025_dam_file(folder: Path = DEFAULT_RESULTS_2025_DAM_DIR) -> Path:
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No DAM Excel files found in {folder}")
    return files[0]


def load_dam_price_curve(file_path: Path, dt: float = 0.25) -> pd.Series:
    """Load one DAM daily Excel file and align MCP prices to the optimizer timestep."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"DAM file not found: {file_path}")

    daily_df = _read_dam_excel(file_path).dropna(subset=["DELIVERY_MTU", "MCP"])
    if daily_df.empty:
        raise ValueError(f"DAM file has no DELIVERY_MTU/MCP rows: {file_path}")

    daily_df["DELIVERY_MTU"] = pd.to_datetime(daily_df["DELIVERY_MTU"])
    hourly_prices = daily_df.groupby("DELIVERY_MTU")["MCP"].mean().sort_index()
    if hourly_prices.empty:
        raise ValueError(f"No hourly DAM prices were loaded from {file_path}")

    step_minutes = int(round(dt * 60))
    if step_minutes <= 0:
        raise ValueError("dt must be positive.")
    periods = int(round(24 * 60 / step_minutes))
    start = hourly_prices.index.min().floor("D")
    target_index = pd.date_range(start=start, periods=periods, freq=f"{step_minutes}min")

    aligned = (
        hourly_prices.reindex(hourly_prices.index.union(target_index))
        .sort_index()
        .ffill()
        .reindex(target_index)
    )
    if aligned.isna().any():
        raise ValueError(f"Could not align DAM prices to {step_minutes}-minute steps.")

    aligned.name = "price_eur_mwh"
    return aligned


def load_price_signal_csv_day(
    csv_path: Path = DEFAULT_PRICE_SIGNALS_PATH,
    target_date: str = "2025-01-10",
    dt: float = 0.25,
) -> pd.Series:
    """Load one day of 15-minute DAM prices from price_signals_15m.csv."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Price signal CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"DELIVERY_MTU", "DAM_Price_EUR_MWh"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV column(s): {', '.join(sorted(missing))}")

    df["DELIVERY_MTU"] = pd.to_datetime(df["DELIVERY_MTU"])
    price_series = (
        df.groupby("DELIVERY_MTU")["DAM_Price_EUR_MWh"]
        .mean()
        .sort_index()
    )

    day_start = pd.Timestamp(target_date).floor("D")
    day_end = day_start + pd.Timedelta(days=1)
    day_mask = (price_series.index >= day_start) & (price_series.index < day_end)
    if not day_mask.any():
        raise ValueError(f"No price rows found for {day_start.date()} in {csv_path}")

    step_minutes = int(round(dt * 60))
    if step_minutes <= 0:
        raise ValueError("dt must be positive.")
    periods = int(round(24 * 60 / step_minutes))
    target_index = pd.date_range(
        start=day_start,
        periods=periods,
        freq=f"{step_minutes}min",
    )

    aligned = (
        price_series.reindex(price_series.index.union(target_index))
        .sort_index()
        .ffill()
        .reindex(target_index)
    )
    if aligned.isna().any():
        raise ValueError(
            f"Could not align prices for {day_start.date()} to {step_minutes}-minute steps."
        )

    aligned.name = "price_eur_mwh"
    return aligned


def load_price_signal_csv_year(
    csv_path: Path = DEFAULT_PRICE_SIGNALS_PATH,
    year: int = 2025,
    dt: float = 0.25,
) -> pd.Series:
    """Load a complete year of 15-minute DAM prices from price_signals_15m.csv."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Price signal CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"DELIVERY_MTU", "DAM_Price_EUR_MWh"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing CSV column(s): {', '.join(sorted(missing))}")

    df["DELIVERY_MTU"] = pd.to_datetime(df["DELIVERY_MTU"])
    price_series = (
        df.groupby("DELIVERY_MTU")["DAM_Price_EUR_MWh"]
        .mean()
        .sort_index()
    )

    step_minutes = int(round(dt * 60))
    if step_minutes <= 0:
        raise ValueError("dt must be positive.")

    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year + 1, month=1, day=1)
    target_index = pd.date_range(
        start=year_start,
        end=year_end - pd.Timedelta(minutes=step_minutes),
        freq=f"{step_minutes}min",
    )

    aligned = price_series.reindex(target_index)
    if aligned.isna().any():
        missing_count = int(aligned.isna().sum())
        raise ValueError(
            f"{missing_count} expected {step_minutes}-minute prices are missing for {year}."
        )

    aligned.name = "price_eur_mwh"
    return aligned


def load_degradation_lut_curve(
    csv_path: Path = DEFAULT_DEGRADATION_LUT_PATH,
    temperature_c: float = 25.0,
) -> tuple[list[float], list[float]]:
    """Load degradation lookup table and return energy points with absolute EUR costs."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Degradation LUT not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required_columns = {
        "energy",
        "temperature_c",
        "deg_cost_eur_per_MWh_throughput",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing LUT column(s): {', '.join(sorted(missing))}")

    filtered = df[df["temperature_c"].sub(float(temperature_c)).abs() < 1e-9].copy()
    if filtered.empty:
        available = ", ".join(str(value) for value in sorted(df["temperature_c"].unique()))
        raise ValueError(
            f"No LUT rows found for temperature {temperature_c}. "
            f"Available temperatures: {available}"
        )

    filtered = filtered.sort_values("energy")
    if filtered["energy"].duplicated().any():
        raise ValueError("Degradation LUT contains duplicate energy points.")
    if (filtered["energy"] <= 0).any():
        raise ValueError("Degradation LUT energy points must be positive.")
    if (filtered["deg_cost_eur_per_MWh_throughput"] < 0).any():
        raise ValueError("Degradation LUT costs cannot be negative.")

    energy_points = filtered["energy"].astype(float).tolist()
    cost_points = (
        filtered["energy"] * filtered["deg_cost_eur_per_MWh_throughput"]
    ).astype(float).tolist()

    energy_points.insert(0, 0.0)
    cost_points.insert(0, 0.0)
    return energy_points, cost_points


def build_price_signals(
    folder_2024: Path = BASE_DIR / "Results_2024" / "DAM",
    folder_2025: Path = DEFAULT_RESULTS_2025_DAM_DIR,
    output_path: Path = BASE_DIR / "price_signals_15m.csv",
) -> pd.DataFrame:
    df_24 = henexing(folder_2024)
    df_25 = henexing(folder_2025)

    df_raw_total = pd.concat([df_24, df_25], ignore_index=True)
    if df_raw_total.empty:
        raise ValueError("No DAM price rows were loaded from the configured folders.")

    df_raw_total["DELIVERY_MTU"] = pd.to_datetime(df_raw_total["DELIVERY_MTU"])

    df_clean = df_raw_total.groupby("DELIVERY_MTU").mean()
    df_clean = df_clean.sort_index()

    df_15min = df_clean.resample("15min").ffill()
    df_15min.columns = ["DAM_Price_EUR_MWh"]
    df_15min.to_csv(output_path)
    return df_15min


if __name__ == "__main__":
    build_price_signals()
