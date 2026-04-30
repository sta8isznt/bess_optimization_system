"""Simple Streamlit page for forecast-driven BESS optimization."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DASHBOARD_DIR.parent
for candidate in (DASHBOARD_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from optimizer_adapter import (  # noqa: E402
    DashboardOptimizerError,
    default_lut_for_source,
    display_path,
    list_lut_files,
    list_price_files,
    run_daily_optimization,
    validate_parameters,
)
from optimization.forecasting.dam_15min_forecast import (  # noqa: E402
    DEFAULT_FORECAST_OUTPUT,
    ForecastingError,
    forecast_next_day,
    load_price_history,
    utc_created_at,
    write_forecast_outputs,
)
from styles import inject_css, primary_kpi_grid, section_title  # noqa: E402


st.set_page_config(
    page_title="Forecast Optimizer",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_history(path: str) -> tuple[pd.DataFrame, dict, list[str]]:
    history = load_price_history(path)
    metadata = {
        "input_file": str(history.input_file),
        "timestamp_col": history.timestamp_col,
        "price_col": history.price_col,
    }
    return history.frame, metadata, list(history.warnings)


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def next_target_date(history: pd.DataFrame) -> pd.Timestamp:
    latest = pd.to_datetime(history["timestamp"]).max()
    return (latest.floor("D") + pd.Timedelta(days=1)).normalize()


def latest_complete_day(history: pd.DataFrame) -> pd.Timestamp | None:
    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["date"] = frame["timestamp"].dt.normalize()
    complete = frame.groupby("date")["timestamp"].nunique()
    complete = complete[complete == 96]
    if complete.empty:
        return None
    return pd.Timestamp(complete.index.max())


def history_date_bounds(history: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamps = pd.to_datetime(history["timestamp"])
    return timestamps.min().floor("D"), timestamps.max().floor("D")


def history_day_series(history: pd.DataFrame, target_date) -> pd.Series | None:
    target_day = pd.Timestamp(target_date).floor("D")
    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    day = frame[frame["timestamp"].dt.normalize() == target_day]
    if day.empty:
        return None
    prices = day.groupby("timestamp")["price_eur_mwh"].mean().sort_index()
    prices.name = "Real EUR/MWh"
    return prices


def price_chart_frame(forecast: pd.DataFrame, actual: pd.Series | None) -> pd.DataFrame:
    frame = forecast[["timestamp", "forecast_price_eur_mwh"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["forecast_price_eur_mwh"] = pd.to_numeric(
        frame["forecast_price_eur_mwh"],
        errors="coerce",
    )
    return (
        frame.set_index("timestamp")
        .sort_index()
        .rename(columns={"forecast_price_eur_mwh": "Forecast EUR/MWh"})
    )


def price_comparison_frame(forecast: pd.DataFrame, actual: pd.Series | None) -> pd.DataFrame:
    chart = price_chart_frame(forecast, actual)
    if actual is not None:
        chart["Real EUR/MWh"] = actual.reindex(chart.index)
    return chart


def compact_forecast_table(forecast: pd.DataFrame, actual: pd.Series | None) -> pd.DataFrame:
    table = forecast[["timestamp", "forecast_price_eur_mwh"]].copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"])
    table["forecast_price_eur_mwh"] = pd.to_numeric(
        table["forecast_price_eur_mwh"],
        errors="coerce",
    )
    if actual is not None:
        actual_frame = actual.rename("actual_price_eur_mwh").reset_index()
        table = table.merge(actual_frame, on="timestamp", how="left")
        table["error_eur_mwh"] = (
            table["forecast_price_eur_mwh"] - table["actual_price_eur_mwh"]
        )

    table = table.rename(
        columns={
            "timestamp": "Timestamp",
            "forecast_price_eur_mwh": "Forecast EUR/MWh",
            "actual_price_eur_mwh": "Real EUR/MWh",
            "error_eur_mwh": "Error EUR/MWh",
        }
    )
    numeric_cols = table.select_dtypes(include=["float", "int"]).columns
    table[numeric_cols] = table[numeric_cols].round(3)
    return table


def settled_schedule(schedule: pd.DataFrame, actual: pd.Series | None) -> pd.DataFrame:
    settled = schedule.copy()
    settled["timestamp"] = pd.to_datetime(settled["timestamp"])
    settled["forecast_interval_profit_eur"] = settled["interval_profit_eur"]
    if actual is None:
        return settled

    aligned = actual.reindex(settled["timestamp"])
    if aligned.isna().any():
        return settled

    settled["actual_price_eur_mwh"] = aligned.to_numpy(dtype=float)
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
    return settled


def compact_dispatch_table(schedule: pd.DataFrame, actual: pd.Series | None) -> pd.DataFrame:
    settled = settled_schedule(schedule, actual)
    p_buy = pd.to_numeric(schedule["p_buy_mw"], errors="coerce").fillna(0.0)
    p_sell = pd.to_numeric(schedule["p_sell_mw"], errors="coerce").fillna(0.0)
    action = pd.Series("HOLD", index=schedule.index)
    action[p_buy > 1e-6] = "BUY"
    action[p_sell > 1e-6] = "SELL"

    table = pd.DataFrame(
        {
            "Timestamp": settled["timestamp"],
            "Forecast EUR/MWh": pd.to_numeric(settled["price_eur_mwh"], errors="coerce"),
            "Action": action,
            "SoC %": pd.to_numeric(settled["soc_pct"], errors="coerce") * 100.0,
        }
    )
    if "actual_price_eur_mwh" in settled.columns:
        table.insert(
            2,
            "Real EUR/MWh",
            pd.to_numeric(settled["actual_price_eur_mwh"], errors="coerce"),
        )
        table["Planned profit EUR"] = pd.to_numeric(
            settled["forecast_interval_profit_eur"],
            errors="coerce",
        )
        table["Realized profit EUR"] = pd.to_numeric(
            settled["actual_interval_profit_eur"],
            errors="coerce",
        )
    numeric_cols = table.select_dtypes(include=["float", "int"]).columns
    table[numeric_cols] = table[numeric_cols].round(3)
    return table


def scaled(summary: dict, base_key: str, park_key: str) -> float:
    if float(summary.get("scale_factor", 1.0)) != 1.0 and park_key in summary:
        return float(summary[park_key])
    return float(summary.get(base_key, 0.0))


def eur_per_mwh_discharged(value: float, discharged_mwh: float) -> str:
    if abs(discharged_mwh) < 1e-12:
        return "N/A EUR/MWh discharged"
    return f"{value / discharged_mwh:,.2f} EUR/MWh discharged"


def forecast_error_metrics(forecast: pd.DataFrame, actual: pd.Series | None) -> dict:
    if actual is None:
        return {}
    comparison = price_comparison_frame(forecast, actual).dropna()
    if comparison.empty or "Real EUR/MWh" not in comparison.columns:
        return {}
    error = comparison["Forecast EUR/MWh"] - comparison["Real EUR/MWh"]
    return {
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
    }


def settled_profit(summary: dict, schedule: pd.DataFrame, actual: pd.Series | None) -> float | None:
    if actual is None:
        return None
    settled = settled_schedule(schedule, actual)
    if "actual_interval_profit_eur" not in settled.columns:
        return None
    scale_factor = float(summary.get("scale_factor", 1.0))
    return float(settled["actual_interval_profit_eur"].sum() * scale_factor)


def show_metric_row(
    forecast: pd.DataFrame,
    summary: dict,
    schedule: pd.DataFrame,
    actual: pd.Series | None,
    actual_result: dict | None,
) -> None:
    planned_profit = scaled(summary, "net_profit_eur", "park_net_profit_eur")
    degradation_cost = scaled(
        summary,
        "degradation_cost_eur",
        "park_degradation_cost_eur",
    )
    discharged_mwh = scaled(summary, "sell_energy_mwh", "park_sell_energy_mwh")
    cycles = float(summary.get("equivalent_discharge_cycles", 0.0))
    realized_profit = settled_profit(summary, schedule, actual)
    actual_optimal_profit = None
    if actual_result is not None:
        actual_summary = actual_result["summary_dict"]
        actual_optimal_profit = scaled(
            actual_summary,
            "net_profit_eur",
            "park_net_profit_eur",
        )

    display_profit = planned_profit if realized_profit is None else realized_profit
    if realized_profit is None:
        profit_note = "planned on forecast prices"
    elif actual_optimal_profit is None:
        profit_note = f"planned EUR {planned_profit:,.0f}"
    else:
        profit_note = (
            f"planned EUR {planned_profit:,.0f} / real optimizer EUR "
            f"{actual_optimal_profit:,.0f}"
        )

    primary_kpi_grid(
        [
            (
                "Net Profit",
                f"EUR {display_profit:,.0f}",
                profit_note,
                "positive" if display_profit >= 0 else "negative",
            ),
            (
                "Degradation Cost",
                f"EUR {degradation_cost:,.0f}",
                eur_per_mwh_discharged(degradation_cost, discharged_mwh),
                "negative",
            ),
            (
                "Equivalent Full Cycles",
                f"{cycles:,.3f}",
                "cycles/day",
                "accent",
            ),
        ]
    )


inject_css()

st.title("Forecast Optimizer")
st.caption("Generate a DAM forecast, run the daily optimizer, and compare against real prices when available.")

price_files = list_price_files()
lut_files = list_lut_files()

if not price_files:
    st.error("No historical price CSV files were found under optimization/data/cleaned_data.")
    st.stop()

st.sidebar.title("Inputs")
price_labels = [display_path(path) for path in price_files]
selected_price_label = st.sidebar.selectbox("Historical price file", price_labels, index=0)
price_file = price_files[price_labels.index(selected_price_label)]

try:
    history, history_metadata, history_warnings = cached_history(str(price_file))
except ForecastingError as exc:
    st.error(str(exc))
    st.stop()

history_start, history_end = history_date_bounds(history)
default_target = latest_complete_day(history) or next_target_date(history)
min_target = min(history_start + pd.Timedelta(days=1), default_target)
max_target = max(history_end + pd.Timedelta(days=1), min_target)

target_date = st.sidebar.date_input(
    "Forecast delivery date",
    value=default_target.date(),
    min_value=min_target.date(),
    max_value=max_target.date(),
)
window_days = st.sidebar.slider("History window days", 7, 90, 30, step=1)
default_output = DEFAULT_FORECAST_OUTPUT.relative_to(PROJECT_ROOT)
output_file_text = st.sidebar.text_input("Forecast output CSV", value=str(default_output))

st.sidebar.divider()
st.sidebar.title("Battery")
installed_capacity_mw = st.sidebar.number_input(
    "Installed capacity MW",
    min_value=0.05,
    max_value=5000.0,
    value=50.0,
    step=1.0,
)
eta_ch = st.sidebar.slider("Charge efficiency", 0.50, 1.00, 0.92, step=0.01)
eta_dis = st.sidebar.slider("Discharge efficiency", 0.50, 1.00, 0.92, step=0.01)
soc_min_pct = st.sidebar.slider("SoC minimum %", 0, 95, 10, step=1)
soc_max_pct = st.sidebar.slider("SoC maximum %", 5, 100, 90, step=1)
soc_init_pct = st.sidebar.slider("Initial SoC %", 0, 100, 50, step=1)
terminal_label = st.sidebar.selectbox("Terminal SoC", ["equal to initial SoC", "free terminal SoC"])
terminal_soc_mode = "equal_initial" if terminal_label == "equal to initial SoC" else "free"

params_override = {
    "p_max": 1.0,
    "e_max": 2.0,
    "eta_ch": float(eta_ch),
    "eta_dis": float(eta_dis),
    "soc_min": float(soc_min_pct) / 100.0,
    "soc_max": float(soc_max_pct) / 100.0,
    "soc_init": float(soc_init_pct) / 100.0,
    "dt": 0.25,
    "terminal_soc_mode": terminal_soc_mode,
}

degradation_source = "pybamm_only"
degradation_lut_file = default_lut_for_source(degradation_source, lut_files)
run_clicked = st.sidebar.button("Run", type="primary")

if run_clicked:
    errors = validate_parameters(params_override)
    if degradation_lut_file is None:
        errors.append("PyBaMM degradation LUT file is missing.")

    if errors:
        st.session_state["forecast_simple_error"] = "\n".join(errors)
        st.session_state.pop("forecast_simple_result", None)
    else:
        st.session_state.pop("forecast_simple_error", None)
        try:
            with st.spinner("Running forecast and optimization..."):
                forecast, forecast_warnings = forecast_next_day(
                    history,
                    target_date=pd.Timestamp(target_date),
                    window_days=int(window_days),
                    model="seasonal",
                    created_at_utc=utc_created_at(),
                )
                actual_prices = history_day_series(history, target_date)
                forecast_path, optimizer_path = write_forecast_outputs(
                    forecast,
                    resolve_project_path(output_file_text),
                    write_optimizer_input=True,
                )
                if optimizer_path is None:
                    raise ForecastingError("Forecast optimizer input CSV was not created.")

                optimization_result = run_daily_optimization(
                    target_date=pd.Timestamp(target_date).date().isoformat(),
                    price_file=optimizer_path,
                    degradation_lut_file=degradation_lut_file,
                    params_override=params_override,
                    degradation_source=degradation_source,
                    scale_capacity_mw=float(installed_capacity_mw),
                    temperature_c=25.0,
                    terminal_soc_mode=terminal_soc_mode,
                )

                actual_optimization_result = None
                actual_warnings = []
                if actual_prices is not None:
                    try:
                        actual_optimization_result = run_daily_optimization(
                            target_date=pd.Timestamp(target_date).date().isoformat(),
                            price_file=price_file,
                            degradation_lut_file=degradation_lut_file,
                            params_override=params_override,
                            degradation_source=degradation_source,
                            scale_capacity_mw=float(installed_capacity_mw),
                            temperature_c=25.0,
                            terminal_soc_mode=terminal_soc_mode,
                        )
                        actual_warnings = actual_optimization_result.get("warnings", [])
                    except Exception as exc:
                        actual_warnings = [f"Real-price optimizer unavailable: {exc}"]

            st.session_state["forecast_simple_result"] = {
                "forecast": forecast,
                "actual_prices": actual_prices,
                "forecast_path": forecast_path,
                "optimizer_path": optimizer_path,
                "optimization_result": optimization_result,
                "actual_optimization_result": actual_optimization_result,
                "target_date": pd.Timestamp(target_date).date().isoformat(),
                "warnings": (
                    history_warnings
                    + forecast_warnings
                    + optimization_result.get("warnings", [])
                    + actual_warnings
                ),
            }
        except (ForecastingError, DashboardOptimizerError, RuntimeError, ValueError) as exc:
            st.session_state["forecast_simple_error"] = str(exc)
            st.session_state.pop("forecast_simple_result", None)

if st.session_state.get("forecast_simple_error"):
    st.error(st.session_state["forecast_simple_error"])

result = st.session_state.get("forecast_simple_result")
if result is None:
    st.info("Choose the inputs in the sidebar and click Run.")
    st.write(f"History: `{display_path(price_file)}`")
    st.write(f"Available data: `{history_start.date().isoformat()}` to `{history_end.date().isoformat()}`")
    for warning in history_warnings:
        st.warning(warning)
    st.stop()

for warning in result["warnings"]:
    st.warning(warning)

forecast = result["forecast"]
actual_prices = result["actual_prices"]
optimization_result = result["optimization_result"]
actual_optimization_result = result["actual_optimization_result"]
summary = optimization_result["summary_dict"]
schedule = optimization_result["dispatch_df"]

section_title("Basic Statistics")
show_metric_row(
    forecast=forecast,
    summary=summary,
    schedule=schedule,
    actual=actual_prices,
    actual_result=actual_optimization_result,
)
forecast_error = forecast_error_metrics(forecast, actual_prices)
if forecast_error:
    st.caption(
        f"Forecast MAE: {forecast_error['mae']:,.2f} EUR/MWh | "
        f"Forecast bias: {forecast_error['bias']:+,.2f} EUR/MWh"
    )

if actual_prices is None:
    st.info(
        "No real DAM prices are available for this delivery date, so the page shows the forecast plan only."
    )

section_title("Forecast vs Real Price" if actual_prices is not None else "Forecast Price")
st.line_chart(price_comparison_frame(forecast, actual_prices), height=280)
st.dataframe(
    compact_forecast_table(forecast, actual_prices),
    use_container_width=True,
    hide_index=True,
    height=240,
)

section_title("Forecast Dispatch")
dispatch_table = compact_dispatch_table(schedule, actual_prices)
st.dataframe(dispatch_table, use_container_width=True, hide_index=True, height=320)

st.caption(f"Forecast CSV: {display_path(Path(result['forecast_path']))}")
st.caption(f"Optimizer input CSV: {display_path(Path(result['optimizer_path']))}")

col1, col2 = st.columns(2)
col1.download_button(
    "Download forecast CSV",
    forecast.to_csv(index=False).encode("utf-8"),
    file_name=Path(result["forecast_path"]).name,
    mime="text/csv",
)
col2.download_button(
    "Download dispatch CSV",
    dispatch_table.to_csv(index=False).encode("utf-8"),
    file_name="bess_forecast_dispatch_schedule.csv",
    mime="text/csv",
)
