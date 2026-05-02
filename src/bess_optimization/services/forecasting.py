"""Forecast workflow service."""

from __future__ import annotations

from pathlib import Path

from bess_optimization.forecasting.dam_15min_forecast import (
    DEFAULT_FORECAST_OUTPUT,
    forecast_next_day,
    load_price_history,
    utc_created_at,
    write_forecast_outputs,
)
from bess_optimization.models import ForecastRequest, ForecastResult


def run_forecast(request: ForecastRequest) -> ForecastResult:
    history = load_price_history(
        request.input_file,
        timestamp_col=request.timestamp_col,
        price_col=request.price_col,
        allow_hourly_upsampling=request.allow_hourly_upsampling,
    )
    forecast, forecast_warnings = forecast_next_day(
        history.frame,
        target_date=request.target_date,
        window_days=request.window_days,
        model=request.model,
        created_at_utc=utc_created_at(),
    )
    output_path = None
    optimizer_path = None
    if request.output_file is not None:
        output_path, optimizer_path = write_forecast_outputs(
            forecast,
            request.output_file or DEFAULT_FORECAST_OUTPUT,
            write_optimizer_input=request.write_optimizer_input,
        )
    return ForecastResult(
        forecast=forecast,
        output_path=Path(output_path) if output_path is not None else None,
        optimizer_input_path=Path(optimizer_path) if optimizer_path is not None else None,
        warnings=tuple(list(history.warnings) + list(forecast_warnings)),
        metadata={
            "input_file": str(history.input_file),
            "timestamp_col": history.timestamp_col,
            "price_col": history.price_col,
        },
    )
