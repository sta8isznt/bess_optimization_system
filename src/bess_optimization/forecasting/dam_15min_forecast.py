"""Greek DAM 15-minute next-day price forecasting baseline.

The module intentionally keeps the model simple and explainable: it forecasts
each 15-minute slot from recent slot-of-day medians blended with the previous
day's same slot when available.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


from bess_optimization.paths import FORECAST_OUTPUT_DIR, PROJECT_ROOT


REPO_ROOT = PROJECT_ROOT
DEFAULT_FORECAST_DIR = FORECAST_OUTPUT_DIR
DEFAULT_FORECAST_OUTPUT = DEFAULT_FORECAST_DIR / "dam_15min_forecast_next_day.csv"
DEFAULT_BACKTEST_OUTPUT = (
    DEFAULT_FORECAST_DIR / "dam_15min_forecast_backtest_metrics.csv"
)

TIMESTAMP_CANDIDATES = (
    "timestamp",
    "delivery_mtu",
    "DELIVERY_MTU",
    "datetime",
    "date_time",
    "time",
    "delivery_start",
    "market_time_unit",
)
PRICE_CANDIDATES = (
    "price_eur_mwh",
    "DAM_Price_EUR_MWh",
    "dam_price_eur_mwh",
    "market_price",
    "market_clearing_price",
    "price",
    "value",
)

EXPECTED_PERIODS_PER_DAY = 96
EXPECTED_FREQ = pd.Timedelta(minutes=15)
HOURLY_FREQ = pd.Timedelta(hours=1)
MIN_PRICE_EUR_MWH = -500.0
MAX_PRICE_EUR_MWH = 1000.0
GREEK_DAM_15MIN_START = pd.Timestamp("2025-10-01")


class ForecastingError(RuntimeError):
    """Raised when a forecast cannot be produced from the supplied data."""


@dataclass(frozen=True)
class PriceHistory:
    """Normalized price history plus input metadata."""

    frame: pd.DataFrame
    input_file: Path
    timestamp_col: str
    price_col: str
    warnings: tuple[str, ...]


def utc_created_at() -> str:
    """Return an ISO-8601 UTC timestamp suitable for CSV metadata."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _normalized_name(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _resolve_column(
    df: pd.DataFrame,
    explicit_col: str | None,
    candidates: Sequence[str],
    purpose: str,
) -> str:
    if explicit_col:
        if explicit_col not in df.columns:
            raise ForecastingError(
                f"{purpose} column '{explicit_col}' was not found. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return explicit_col

    normalized_to_actual = {_normalized_name(col): col for col in df.columns}
    for candidate in candidates:
        match = normalized_to_actual.get(_normalized_name(candidate))
        if match is not None:
            return match

    if purpose == "timestamp":
        best_col = None
        best_score = 0.0
        for col in df.columns:
            parsed = pd.to_datetime(df[col].head(200), errors="coerce")
            score = float(parsed.notna().mean())
            if score > best_score:
                best_col = col
                best_score = score
        if best_col is not None and best_score >= 0.8:
            return str(best_col)

    if purpose == "price":
        numeric_candidates = []
        for col in df.columns:
            numeric = pd.to_numeric(df[col].head(500), errors="coerce")
            if numeric.notna().mean() >= 0.8:
                name = _normalized_name(col)
                score = 0
                if "price" in name:
                    score += 4
                if "dam" in name:
                    score += 2
                if "eur" in name or "mwh" in name:
                    score += 1
                numeric_candidates.append((score, str(col)))
        if numeric_candidates:
            numeric_candidates.sort(reverse=True)
            return numeric_candidates[0][1]

    raise ForecastingError(
        f"Could not infer the {purpose} column. Pass --{purpose}-col explicitly. "
        f"Available columns: {', '.join(map(str, df.columns))}"
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ForecastingError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ForecastingError(
        f"Unsupported input file type '{path.suffix}'. Use CSV, XLSX, or XLS."
    )


def _strip_timezone(timestamps: pd.Series) -> pd.Series:
    if getattr(timestamps.dt, "tz", None) is not None:
        return timestamps.dt.tz_localize(None)
    return timestamps


def _validate_or_upsample_frequency(
    frame: pd.DataFrame,
    allow_hourly_upsampling: bool,
) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    if len(frame) < 2:
        warnings.append(
            "Input has fewer than two valid timestamps; 15-minute spacing could "
            "not be inferred. Falling back to global median logic."
        )
        return frame, warnings

    diffs = frame["timestamp"].diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    if positive_diffs.empty:
        raise ForecastingError("Input timestamps do not advance after de-duplication.")

    most_common_diff = positive_diffs.value_counts().idxmax()
    min_diff = positive_diffs.min()

    if min_diff == HOURLY_FREQ and most_common_diff == HOURLY_FREQ:
        if not allow_hourly_upsampling:
            raise ForecastingError(
                "Input appears to be hourly, not 15-minute DAM MTU data. "
                "Pass --allow-hourly-upsampling only for testing or explicit "
                "legacy-data experiments."
            )
        upsampled = (
            frame.set_index("timestamp")
            .sort_index()
            .resample("15min")
            .ffill()
            .reset_index()
        )
        warnings.append(
            "Hourly input was upsampled to 15-minute slots by forward filling "
            "because --allow-hourly-upsampling was set."
        )
        return upsampled, warnings

    remainder_ns = positive_diffs.map(lambda value: value.value % EXPECTED_FREQ.value)
    if not (min_diff == EXPECTED_FREQ or most_common_diff == EXPECTED_FREQ):
        raise ForecastingError(
            "Input does not look like 15-minute DAM MTU data. "
            f"Most common timestamp step is {most_common_diff}."
        )
    if (remainder_ns != 0).any():
        raise ForecastingError(
            "Input timestamp spacing includes intervals that are not multiples "
            "of 15 minutes."
        )
    if (positive_diffs != EXPECTED_FREQ).any():
        warnings.append(
            "Input is 15-minute based but has gaps. Forecasting will use the "
            "available historical slots."
        )
    return frame, warnings


def load_price_history(
    input_file: str | Path,
    timestamp_col: str | None = None,
    price_col: str | None = None,
    allow_hourly_upsampling: bool = False,
) -> PriceHistory:
    """Load and normalize a DAM price history file.

    The returned frame always contains ``timestamp`` and ``price_eur_mwh``.
    Duplicate timestamps are averaged, invalid rows are dropped, and timestamp
    spacing is validated as 15-minute based by default.
    """

    path = Path(input_file).expanduser().resolve()
    raw = _read_table(path)
    if raw.empty:
        raise ForecastingError(f"Input file is empty: {path}")

    resolved_timestamp_col = _resolve_column(
        raw,
        timestamp_col,
        TIMESTAMP_CANDIDATES,
        "timestamp",
    )
    resolved_price_col = _resolve_column(raw, price_col, PRICE_CANDIDATES, "price")

    timestamps = pd.to_datetime(raw[resolved_timestamp_col], errors="coerce")
    timestamps = _strip_timezone(timestamps)
    prices = pd.to_numeric(raw[resolved_price_col], errors="coerce")
    frame = pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})
    invalid_rows = int(frame.isna().any(axis=1).sum())
    frame = frame.dropna(subset=["timestamp", "price_eur_mwh"])
    if frame.empty:
        raise ForecastingError(
            "No valid rows remain after parsing timestamps and prices."
        )

    frame = (
        frame.groupby("timestamp", as_index=False)["price_eur_mwh"]
        .mean()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    warnings: list[str] = []
    if invalid_rows:
        warnings.append(f"Dropped {invalid_rows} rows with invalid timestamps/prices.")
    if frame["timestamp"].min() < GREEK_DAM_15MIN_START:
        warnings.append(
            "Input includes timestamps before 2025-10-01, when Greek DAM moved "
            "to 15-minute MTUs. Ensure legacy hourly data was explicitly "
            "resampled before using it for 15-minute forecasts."
        )

    frame, frequency_warnings = _validate_or_upsample_frequency(
        frame,
        allow_hourly_upsampling=allow_hourly_upsampling,
    )
    warnings.extend(frequency_warnings)
    return PriceHistory(
        frame=frame,
        input_file=path,
        timestamp_col=str(resolved_timestamp_col),
        price_col=str(resolved_price_col),
        warnings=tuple(warnings),
    )


def _with_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["slot_id"] = out["hour"] * 4 + out["minute"] // 15
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def _target_index(target_date: str | pd.Timestamp) -> pd.DatetimeIndex:
    target_day = pd.Timestamp(target_date).floor("D")
    return pd.date_range(target_day, periods=EXPECTED_PERIODS_PER_DAY, freq="15min")


def _target_date_from_history(history: pd.DataFrame) -> pd.Timestamp:
    latest = pd.to_datetime(history["timestamp"]).max()
    return (latest.floor("D") + pd.Timedelta(days=1)).normalize()


def _safe_median(values: pd.Series, fallback: float) -> float:
    value = float(values.median()) if not values.empty else np.nan
    if np.isfinite(value):
        return value
    return fallback


def _clip_forecasts(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    prices = frame["forecast_price_eur_mwh"].to_numpy(dtype=float)
    clipped = np.clip(prices, MIN_PRICE_EUR_MWH, MAX_PRICE_EUR_MWH)
    if np.array_equal(prices, clipped):
        return frame, []

    out = frame.copy()
    out["forecast_price_eur_mwh"] = clipped
    count = int(np.sum(prices != clipped))
    return out, [
        f"Clipped {count} forecast prices to the safety range "
        f"[{MIN_PRICE_EUR_MWH:g}, {MAX_PRICE_EUR_MWH:g}] EUR/MWh."
    ]


def validate_forecast_output(frame: pd.DataFrame) -> None:
    """Validate the required 96-row 15-minute forecast output contract."""

    if len(frame) != EXPECTED_PERIODS_PER_DAY:
        raise ForecastingError(
            f"Forecast must contain exactly {EXPECTED_PERIODS_PER_DAY} rows; "
            f"got {len(frame)}."
        )

    timestamps = pd.to_datetime(frame["timestamp"])
    diffs = timestamps.diff().dropna()
    if not (diffs == EXPECTED_FREQ).all():
        raise ForecastingError("Forecast timestamps are not exactly 15 minutes apart.")

    prices = pd.to_numeric(frame["forecast_price_eur_mwh"], errors="coerce")
    if prices.isna().any() or not np.isfinite(prices.to_numpy(dtype=float)).all():
        raise ForecastingError("Forecast prices must be numeric and finite.")


def forecast_next_day(
    history: pd.DataFrame,
    target_date: str | pd.Timestamp | None = None,
    window_days: int = 30,
    model: str = "seasonal",
    created_at_utc: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Forecast one 96-slot Greek DAM delivery day.

    Parameters
    ----------
    history:
        DataFrame with ``timestamp`` and ``price_eur_mwh`` columns.
    target_date:
        Delivery date to forecast. If omitted, uses the day after the latest
        timestamp in ``history``.
    window_days:
        Number of recent historical days used for slot medians.
    model:
        Currently only ``seasonal`` is supported.
    created_at_utc:
        Optional timestamp used in the output metadata.
    """

    model = model.strip().lower()
    if model != "seasonal":
        raise ForecastingError(
            f"Unsupported model '{model}'. The available lightweight baseline is "
            "'seasonal'."
        )
    if window_days <= 0:
        raise ForecastingError("--window-days must be a positive integer.")

    required = {"timestamp", "price_eur_mwh"}
    missing = required.difference(history.columns)
    if missing:
        raise ForecastingError(
            "History is missing required columns: " + ", ".join(sorted(missing))
        )

    warnings: list[str] = []
    featured = _with_time_features(history)
    target_day = (
        _target_date_from_history(featured)
        if target_date is None
        else pd.Timestamp(target_date).floor("D")
    )
    index = _target_index(target_day)
    training = featured[featured["timestamp"] < target_day].copy()
    if training.empty:
        raise ForecastingError(
            "No historical rows are available before the target date. "
            "Choose a later --target-date or provide more history."
        )

    window_start = target_day - pd.Timedelta(days=int(window_days))
    recent = training[training["timestamp"] >= window_start].copy()
    if recent.empty:
        recent = training.copy()
        warnings.append(
            f"No rows found in the last {window_days} days; using all prior "
            "history for the seasonal baseline."
        )

    global_recent_median = _safe_median(
        recent["price_eur_mwh"],
        fallback=float(training["price_eur_mwh"].median()),
    )
    slot_medians = recent.groupby("slot_id")["price_eur_mwh"].median()
    yesterday = training[
        training["date"] == (target_day - pd.Timedelta(days=1)).date()
    ].copy()
    yesterday_by_slot = yesterday.groupby("slot_id")["price_eur_mwh"].last()

    rows = []
    for timestamp in index:
        slot_id = int(timestamp.hour * 4 + timestamp.minute // 15)
        slot_value = float(slot_medians.get(slot_id, np.nan))
        yesterday_value = float(yesterday_by_slot.get(slot_id, np.nan))

        has_slot = np.isfinite(slot_value)
        has_yesterday = np.isfinite(yesterday_value)
        if has_slot and has_yesterday:
            forecast_price = 0.60 * slot_value + 0.40 * yesterday_value
            reason = "recent_slot_median_yesterday_blend"
        elif has_slot:
            forecast_price = slot_value
            reason = "recent_slot_median"
        elif has_yesterday:
            forecast_price = yesterday_value
            reason = "yesterday_same_slot"
        else:
            forecast_price = global_recent_median
            reason = "global_recent_median"

        rows.append(
            {
                "timestamp": timestamp,
                "forecast_price_eur_mwh": float(forecast_price),
                "slot_id": slot_id,
                "date": timestamp.date().isoformat(),
                "hour": int(timestamp.hour),
                "minute": int(timestamp.minute),
                "day_of_week": int(timestamp.dayofweek),
                "is_weekend": int(timestamp.dayofweek in {5, 6}),
                "model_name": "seasonal_slot_blend",
                "created_at_utc": created_at_utc or utc_created_at(),
                "forecast_reason": reason,
            }
        )

    forecast = pd.DataFrame(rows)
    forecast, clipping_warnings = _clip_forecasts(forecast)
    warnings.extend(clipping_warnings)
    validate_forecast_output(forecast)
    return forecast, warnings


def _optimizer_input_path(output_file: Path) -> Path:
    return output_file.with_name(f"{output_file.stem}_optimizer_input.csv")


def write_forecast_outputs(
    forecast: pd.DataFrame,
    output_file: str | Path,
    write_optimizer_input: bool = True,
) -> tuple[Path, Path | None]:
    """Write the main forecast CSV and optional optimizer-compatible CSV."""

    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)

    optimizer_path = None
    if write_optimizer_input:
        optimizer_path = _optimizer_input_path(output_path)
        optimizer_input = pd.DataFrame(
            {
                "timestamp": forecast["timestamp"],
                "price_eur_mwh": forecast["forecast_price_eur_mwh"],
                "DELIVERY_MTU": forecast["timestamp"],
                "DAM_Price_EUR_MWh": forecast["forecast_price_eur_mwh"],
            }
        )
        optimizer_input.to_csv(optimizer_path, index=False)

    return output_path, optimizer_path


def _score_candidate(path: Path) -> int:
    name = _normalized_name(path.name)
    score = 0
    if "price" in name:
        score += 5
    if "dam" in name:
        score += 3
    if "15m" in name or "15min" in name:
        score += 2
    if "forecast" in name:
        score -= 5
    if "summary" in name:
        score -= 4
    if "cleaneddata" in _normalized_name(str(path.parent)):
        score += 1
    return score


def discover_default_input_file(repo_root: Path = REPO_ROOT) -> Path:
    """Find a sensible local DAM price input from existing project folders."""

    known_default = repo_root / "data" / "cleaned_data" / "price_signals_15m.csv"
    if known_default.exists():
        return known_default

    search_dirs = [
        repo_root / "data" / "cleaned_data",
        repo_root / "data",
    ]
    candidates: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for suffix in ("*.csv", "*.xlsx", "*.xls"):
            candidates.extend(directory.glob(suffix))

    ranked = sorted(
        ((candidate, _score_candidate(candidate)) for candidate in candidates),
        key=lambda item: (item[1], item[0].stat().st_mtime),
        reverse=True,
    )
    ranked = [item for item in ranked if item[1] > 0]
    if ranked:
        return ranked[0][0]

    raise ForecastingError(
        "No default DAM price file was found. Pass --input-file explicitly. "
        "Tried data/cleaned_data and data."
    )


def _complete_15min_days(history: pd.DataFrame) -> list[pd.Timestamp]:
    featured = _with_time_features(history)
    counts = featured.groupby("date")["slot_id"].nunique()
    complete_dates = [date for date, count in counts.items() if count == EXPECTED_PERIODS_PER_DAY]
    return [pd.Timestamp(date) for date in sorted(complete_dates)]


def _metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    error = predicted_values - actual_values
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    actual_abs = np.abs(actual_values)
    mape_mask = actual_abs > 1e-9
    mape = (
        float(np.mean(np.abs(error[mape_mask]) / actual_abs[mape_mask]) * 100.0)
        if mape_mask.any()
        else np.nan
    )
    smape_denominator = np.abs(actual_values) + np.abs(predicted_values)
    smape_mask = smape_denominator > 1e-9
    smape = (
        float(
            np.mean(
                2.0
                * np.abs(error[smape_mask])
                / smape_denominator[smape_mask]
            )
            * 100.0
        )
        if smape_mask.any()
        else np.nan
    )
    return {
        "mae_eur_mwh": mae,
        "rmse_eur_mwh": rmse,
        "mape_percent": mape,
        "smape_percent": smape,
    }


def run_backtest(
    history: pd.DataFrame,
    backtest_days: int = 7,
    window_days: int = 30,
    model: str = "seasonal",
) -> pd.DataFrame:
    """Backtest the seasonal baseline on the last complete historical days."""

    if backtest_days <= 0:
        raise ForecastingError("--backtest-days must be positive when used.")

    featured = _with_time_features(history)
    complete_days = _complete_15min_days(featured)
    if not complete_days:
        raise ForecastingError("No complete 96-slot days are available for backtesting.")

    rows = []
    for day in complete_days[-int(backtest_days) :]:
        train = featured[featured["timestamp"] < day].copy()
        actual = featured[featured["date"] == day.date()].sort_values("slot_id")
        if train.empty or len(actual) != EXPECTED_PERIODS_PER_DAY:
            continue

        forecast, warnings = forecast_next_day(
            train[["timestamp", "price_eur_mwh"]],
            target_date=day,
            window_days=window_days,
            model=model,
            created_at_utc=utc_created_at(),
        )
        comparison = actual[["timestamp", "price_eur_mwh"]].merge(
            forecast[["timestamp", "forecast_price_eur_mwh"]],
            on="timestamp",
            how="inner",
        )
        if len(comparison) != EXPECTED_PERIODS_PER_DAY:
            continue

        metric_values = _metrics(
            comparison["price_eur_mwh"],
            comparison["forecast_price_eur_mwh"],
        )
        rows.append(
            {
                "target_date": day.date().isoformat(),
                "model_name": "seasonal_slot_blend",
                "train_rows": int(len(train)),
                "actual_rows": int(len(comparison)),
                "warnings": " | ".join(warnings),
                **metric_values,
            }
        )

    if not rows:
        raise ForecastingError(
            "Backtest could not run because no day had enough prior training data."
        )
    return pd.DataFrame(rows)


def build_synthetic_history(days: int = 40, start_date: str = "2026-03-01") -> pd.DataFrame:
    """Create synthetic 15-minute DAM-like prices for self-checks."""

    periods = int(days) * EXPECTED_PERIODS_PER_DAY
    timestamps = pd.date_range(start=start_date, periods=periods, freq="15min")
    slot = timestamps.hour * 4 + timestamps.minute // 15
    weekday = timestamps.dayofweek
    daily_shape = 15.0 * np.sin((slot / EXPECTED_PERIODS_PER_DAY) * 2.0 * np.pi - 1.4)
    evening_peak = 12.0 * np.exp(-((slot - 76) ** 2) / 120.0)
    weekend_discount = np.where(np.isin(weekday, [5, 6]), -5.0, 0.0)
    trend = np.linspace(0.0, 3.0, periods)
    prices = 75.0 + daily_shape + evening_peak + weekend_discount + trend
    return pd.DataFrame({"timestamp": timestamps, "price_eur_mwh": prices})


def run_self_check() -> None:
    """Run a lightweight synthetic data self-check."""

    history = build_synthetic_history()
    forecast, warnings = forecast_next_day(
        history,
        target_date="2026-04-10",
        window_days=30,
    )
    validate_forecast_output(forecast)
    if warnings:
        raise ForecastingError("Self-check produced warnings: " + " | ".join(warnings))
    if not np.isfinite(forecast["forecast_price_eur_mwh"].to_numpy()).all():
        raise ForecastingError("Self-check forecast contains non-finite values.")


def _write_backtest_metrics(metrics: pd.DataFrame, output_file: str | Path) -> Path:
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    return output_path


def _print_warnings(warnings: Iterable[str]) -> None:
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Forecast next-day Greek DAM prices at 15-minute MTU resolution.",
    )
    parser.add_argument("--input-file", help="CSV/XLSX file with historical DAM prices.")
    parser.add_argument(
        "--output-file",
        default=str(DEFAULT_FORECAST_OUTPUT),
        help="Forecast CSV path.",
    )
    parser.add_argument(
        "--target-date",
        help=(
            "Delivery date to forecast as YYYY-MM-DD. Defaults to day after "
            "latest input timestamp."
        ),
    )
    parser.add_argument("--timestamp-col", help="Timestamp column name.")
    parser.add_argument("--price-col", help="Price column name.")
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Recent days used for slot-of-day medians.",
    )
    parser.add_argument(
        "--model",
        default="seasonal",
        choices=["seasonal"],
        help="Forecast model to run.",
    )
    parser.add_argument(
        "--allow-hourly-upsampling",
        action="store_true",
        help="Allow hourly input to be forward-filled into 15-minute slots for testing.",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=0,
        help="Optionally backtest on the last N complete historical days.",
    )
    parser.add_argument(
        "--backtest-output-file",
        default=str(DEFAULT_BACKTEST_OUTPUT),
        help="Backtest metrics CSV path.",
    )
    parser.add_argument(
        "--no-optimizer-input",
        action="store_true",
        help="Skip writing the optimizer-compatible price CSV.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run a synthetic 40-day self-check and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.self_check:
            run_self_check()
            print("Self-check passed: generated a valid 96-row 15-minute forecast.")
            return 0

        input_file = Path(args.input_file) if args.input_file else discover_default_input_file()
        history = load_price_history(
            input_file,
            timestamp_col=args.timestamp_col,
            price_col=args.price_col,
            allow_hourly_upsampling=args.allow_hourly_upsampling,
        )
        _print_warnings(history.warnings)

        created_at = utc_created_at()
        forecast, forecast_warnings = forecast_next_day(
            history.frame,
            target_date=args.target_date,
            window_days=args.window_days,
            model=args.model,
            created_at_utc=created_at,
        )
        _print_warnings(forecast_warnings)
        output_path, optimizer_path = write_forecast_outputs(
            forecast,
            args.output_file,
            write_optimizer_input=not args.no_optimizer_input,
        )

        print(f"Input file: {history.input_file}")
        print(f"Timestamp column: {history.timestamp_col}")
        print(f"Price column: {history.price_col}")
        print(f"Forecast rows: {len(forecast)}")
        print(f"Forecast CSV: {output_path}")
        if optimizer_path is not None:
            print(f"Optimizer input CSV: {optimizer_path}")

        if args.backtest_days:
            metrics = run_backtest(
                history.frame,
                backtest_days=args.backtest_days,
                window_days=args.window_days,
                model=args.model,
            )
            metrics_path = _write_backtest_metrics(metrics, args.backtest_output_file)
            print(f"Backtest metrics CSV: {metrics_path}")
            print(
                "Backtest MAE mean: "
                f"{metrics['mae_eur_mwh'].mean():.3f} EUR/MWh"
            )

        return 0
    except ForecastingError as exc:
        print(f"Forecasting error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
