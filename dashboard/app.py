"""Streamlit hackathon dashboard for the BESS optimizer."""

from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

try:
    from .charts import (
        build_dispatch_chart,
        build_financial_chart,
        build_model_benchmark_chart,
        build_price_preview_chart,
    )
    from .optimizer_adapter import (
        DashboardOptimizerError,
        available_dates,
        available_years,
        default_lut_for_source,
        display_path,
        list_lut_files,
        list_price_files,
        load_price_series,
        run_annual_optimization,
        run_daily_optimization,
        validate_parameters,
    )
    from .styles import (
        compact_kpi_strip,
        config_grid,
        constraint_grid,
        executive_header,
        headline_kpi_grid,
        inject_css,
        primary_kpi_grid,
        section_title,
        system_overview,
    )
except ImportError:
    from charts import (
        build_dispatch_chart,
        build_financial_chart,
        build_model_benchmark_chart,
        build_price_preview_chart,
    )
    from optimizer_adapter import (
        DashboardOptimizerError,
        available_dates,
        available_years,
        default_lut_for_source,
        display_path,
        list_lut_files,
        list_price_files,
        load_price_series,
        run_annual_optimization,
        run_daily_optimization,
        validate_parameters,
    )
    from styles import (
        compact_kpi_strip,
        config_grid,
        constraint_grid,
        executive_header,
        headline_kpi_grid,
        inject_css,
        primary_kpi_grid,
        section_title,
        system_overview,
    )


st.set_page_config(
    page_title="BESS Greek DAM Optimizer",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_dates(path: str) -> list[pd.Timestamp]:
    return available_dates(Path(path))


@st.cache_data(show_spinner=False)
def cached_years(path: str) -> list[int]:
    return available_years(Path(path))


@st.cache_data(show_spinner=False)
def cached_price_series(path: str) -> pd.Series:
    return load_price_series(Path(path))


def available_date_bounds(dates: list[pd.Timestamp]) -> tuple[date, date]:
    ordered = sorted(pd.Timestamp(day).date() for day in dates)
    return ordered[0], ordered[-1]


def selected_date_or_fallback(preferred: pd.Timestamp, dates: list[pd.Timestamp]) -> date:
    min_date, max_date = available_date_bounds(dates)
    preferred_date = pd.Timestamp(preferred).date()
    if min_date <= preferred_date <= max_date:
        return preferred_date
    return max_date


def date_is_available(value, dates: list[pd.Timestamp]) -> bool:
    selected = pd.Timestamp(value).date()
    return selected in {pd.Timestamp(day).date() for day in dates}


def eur(value: float) -> str:
    return f"{value:,.0f}"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def scaled(summary: dict, base_key: str, park_key: str) -> float:
    if float(summary.get("scale_factor", 1.0)) != 1.0 and park_key in summary:
        return float(summary[park_key])
    return float(summary.get(base_key, 0.0))


def eur_per_mwh_discharged(value: float, discharged_mwh: float) -> str:
    if abs(discharged_mwh) < 1e-12:
        return "N/A EUR/MWh discharged"
    return f"{value / discharged_mwh:,.2f} EUR/MWh discharged"


def render_result_kpis(summary: dict, run_mode: str) -> None:
    solver_status = str(summary.get("solver_status", "Unknown"))
    net_profit = scaled(summary, "net_profit_eur", "park_net_profit_eur")
    degradation_cost = scaled(summary, "degradation_cost_eur", "park_degradation_cost_eur")
    discharged_mwh = scaled(summary, "sell_energy_mwh", "park_sell_energy_mwh")
    cycles = float(summary.get("equivalent_discharge_cycles", 0.0))
    cycles_label = "cycles/year" if run_mode == "Annual" else "cycles/day"

    primary_kpi_grid(
        [
            (
                "Net Profit",
                f"EUR {eur(net_profit)}",
                eur_per_mwh_discharged(net_profit, discharged_mwh),
                "positive" if net_profit >= 0 else "negative",
            ),
            (
                "Degradation Cost",
                f"EUR {eur(degradation_cost)}",
                eur_per_mwh_discharged(degradation_cost, discharged_mwh),
                "negative",
            ),
            (
                "Equivalent Full Cycles",
                num(cycles, 3),
                cycles_label,
                "accent",
            ),
        ]
    )

    with st.expander("Show full statistics", expanded=False):
        headline_kpi_grid(
            [
                ("Solver status", solver_status, "positive" if solver_status.lower() == "optimal" else "negative"),
                ("Gross revenue EUR", eur(scaled(summary, "gross_revenue_eur", "park_gross_revenue_eur")), "positive"),
                ("Purchase cost EUR", eur(scaled(summary, "gross_purchase_eur", "park_gross_purchase_eur")), "negative"),
                ("Bought energy MWh", num(scaled(summary, "buy_energy_mwh", "park_buy_energy_mwh"), 2), "accent"),
                ("Sold energy MWh", num(discharged_mwh, 2), "positive"),
                ("Final SoC %", num(float(summary.get("final_soc_pct", 0.0)) * 100.0, 1), "positive"),
                ("Throughput MWh", num(scaled(summary, "total_throughput_mwh", "park_total_throughput_mwh"), 2), "accent"),
                ("Charge intervals", f"{int(summary.get('buy_intervals', 0))}", "accent"),
                ("Discharge intervals", f"{int(summary.get('sell_intervals', 0))}", "positive"),
            ]
        )


def build_benchmark_display_table(comparison: pd.DataFrame) -> pd.DataFrame:
    display = comparison.copy()
    net_profit = pd.to_numeric(display["Net profit EUR"], errors="coerce")
    sold_mwh = pd.to_numeric(display["Sold MWh"], errors="coerce")
    degradation_cost = pd.to_numeric(display["Degradation cost EUR"], errors="coerce")
    display["Net profit EUR/MWh"] = net_profit.div(sold_mwh.where(sold_mwh.abs() > 1e-12))
    display["Degradation cost EUR/MWh"] = degradation_cost.div(sold_mwh.where(sold_mwh.abs() > 1e-12))
    display["Δ vs selected %"] = pd.to_numeric(
        display["Delta vs degradation-aware MILP %"],
        errors="coerce",
    )

    display = display[
        [
            "Model",
            "Net profit EUR",
            "Net profit EUR/MWh",
            "Δ vs selected %",
            "Degradation cost EUR/MWh",
            "Equivalent cycles",
        ]
    ].copy()
    numeric_cols = display.select_dtypes(include=["float", "int"]).columns
    display[numeric_cols] = display[numeric_cols].round(3)
    return display


def build_table(schedule: pd.DataFrame) -> pd.DataFrame:
    table = schedule[
        [
            "timestamp",
            "price_eur_mwh",
            "p_buy_mw",
            "p_sell_mw",
            "net_power_mw",
            "soc_pct",
            "gross_revenue_eur",
            "gross_purchase_eur",
            "degradation_cost_eur",
            "interval_profit_eur",
            "mode",
        ]
    ].copy()
    table["soc_pct"] = table["soc_pct"] * 100.0
    table = table.rename(
        columns={
            "timestamp": "timestamp",
            "price_eur_mwh": "DAM price EUR/MWh",
            "p_buy_mw": "p_buy MW",
            "p_sell_mw": "p_sell MW",
            "net_power_mw": "net_power MW",
            "soc_pct": "SoC %",
            "gross_revenue_eur": "interval revenue EUR",
            "gross_purchase_eur": "interval purchase cost EUR",
            "degradation_cost_eur": "interval degradation cost EUR",
            "interval_profit_eur": "interval net profit EUR",
            "mode": "mode",
        }
    )
    numeric_cols = table.select_dtypes(include=["float", "int"]).columns
    table[numeric_cols] = table[numeric_cols].round(4)
    return table


def build_minimal_dispatch_table(schedule: pd.DataFrame, epsilon: float = 1e-6) -> pd.DataFrame:
    p_buy = pd.to_numeric(schedule["p_buy_mw"], errors="coerce").fillna(0.0)
    p_sell = pd.to_numeric(schedule["p_sell_mw"], errors="coerce").fillna(0.0)
    buy_active = p_buy > epsilon
    sell_active = p_sell > epsilon

    actions = pd.Series("HOLD", index=schedule.index)
    actions[buy_active & ~sell_active] = "BUY"
    actions[sell_active & ~buy_active] = "SELL"
    actions[buy_active & sell_active] = "CONFLICT"

    table = pd.DataFrame(
        {
            "Timestamp": schedule["timestamp"],
            "DAM Price EUR/MWh": pd.to_numeric(schedule["price_eur_mwh"], errors="coerce"),
            "Dispatch Action": actions,
        }
    )
    table["DAM Price EUR/MWh"] = table["DAM Price EUR/MWh"].round(4)
    return table


def path_label(path: Path | str | None) -> str:
    if path is None:
        return "N/A"
    return display_path(Path(path))


def infer_system_stages(
    price_file: Path,
    degradation_lut_file: Path | None,
    available_lut_files: list[Path],
) -> list[dict[str, str]]:
    return [
        {
            "title": "Market Price Signal",
            "kind": "Data",
            "tone": "runtime",
        },
        {
            "title": "Asset Configuration",
            "kind": "Config",
            "tone": "runtime",
        },
        {
            "title": "Offline Degradation Layer",
            "kind": "Offline model",
            "tone": "offline",
        },
        {
            "title": "Optimizer Cost Curve",
            "kind": "Runtime input",
            "tone": "runtime",
        },
        {
            "title": "MILP Dispatch Engine",
            "kind": "Optimizer",
            "tone": "runtime",
        },
        {
            "title": "Schedule, KPIs, Dashboard",
            "kind": "Output",
            "tone": "output",
        },
    ]


def safe_duration(params: dict) -> float:
    power = float(params.get("p_max", 0.0))
    energy = float(params.get("e_max", 0.0))
    return energy / power if power > 0 else 0.0


def render_overview_kpis(summary: dict) -> None:
    intervals = int(summary.get("intervals", 0) or 0)
    active = int(summary.get("buy_intervals", 0) or 0) + int(summary.get("sell_intervals", 0) or 0)
    utilization = f"{100.0 * active / intervals:.1f}%" if intervals else "N/A"
    net_profit = scaled(summary, "net_profit_eur", "park_net_profit_eur")
    items = [
        ("Net profit", f"EUR {net_profit:,.0f}", "positive" if net_profit >= 0 else "negative"),
        ("Revenue", f"EUR {scaled(summary, 'gross_revenue_eur', 'park_gross_revenue_eur'):,.0f}", "positive"),
        ("Charging cost", f"EUR {scaled(summary, 'gross_purchase_eur', 'park_gross_purchase_eur'):,.0f}", "negative"),
        ("Degradation cost", f"EUR {scaled(summary, 'degradation_cost_eur', 'park_degradation_cost_eur'):,.0f}", "negative"),
        ("Bought energy", f"{scaled(summary, 'buy_energy_mwh', 'park_buy_energy_mwh'):,.2f} MWh", ""),
        ("Sold energy", f"{scaled(summary, 'sell_energy_mwh', 'park_sell_energy_mwh'):,.2f} MWh", ""),
        ("Equivalent cycles", f"{float(summary.get('equivalent_discharge_cycles', 0.0)):,.3f}", ""),
        ("Dispatch utilization", utilization, ""),
    ]
    compact_kpi_strip(items)


def render_price_preview(price_file: Path, run_mode: str, target_date, year: int) -> None:
    try:
        prices = cached_price_series(str(price_file))
        if run_mode == "Daily":
            day = pd.Timestamp(target_date).floor("D")
            preview = prices[prices.index.normalize() == day]
            title = f"DAM Price Preview - {day.date().isoformat()}"
        else:
            year_prices = prices[prices.index.year == int(year)]
            preview = year_prices.iloc[: min(len(year_prices), 96 * 7)]
            title = f"DAM Price Preview - first available week of {int(year)}"

        if preview.empty:
            st.info("No price preview is available for the selected horizon.")
            return
        fig, kind = build_price_preview_chart(preview, title)
        if fig is not None and kind == "plotly":
            st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.info(f"Price preview unavailable: {exc}")


def render_constraint_summary(summary: dict, params: dict, files: dict) -> None:
    terminal = "Equal to initial SoC" if params.get("terminal_soc_mode") == "equal_initial" else "Free terminal SoC"
    items = [
        ("Power limit", f"{float(params['p_max']):.3g} MW", True),
        ("Energy capacity", f"{float(params['e_max']):.3g} MWh", True),
        ("Duration", f"{summary.get('battery_duration_h', 0):.2f} h", True),
        ("SoC bounds", f"{float(params['soc_min']) * 100:.0f}% to {float(params['soc_max']) * 100:.0f}%", True),
        ("Initial SoC", f"{float(params['soc_init']) * 100:.0f}%", True),
        ("Terminal rule", terminal, True),
        ("Efficiency", f"charge {float(params['eta_ch']) * 100:.1f}% / discharge {float(params['eta_dis']) * 100:.1f}%", True),
        ("Degradation LUT", files.get("degradation_source") or "None", True),
        ("Intervals", f"{int(summary.get('intervals', 0))}", True),
        ("Timestep", f"{float(params.get('dt', 0.25)) * 60:.0f} minutes", True),
    ]
    constraint_grid(items)

inject_css()

price_files = list_price_files()
lut_files = list_lut_files()

if not price_files:
    st.error("No price CSV files were found under data/cleaned_data.")
    st.stop()
    raise SystemExit

st.sidebar.title("Optimizer Controls")
run_mode = st.sidebar.radio("Run mode", ["Daily", "Annual"], index=0, horizontal=True)

price_file = price_files[0]

try:
    dates = cached_dates(str(price_file))
    years = cached_years(str(price_file))
except DashboardOptimizerError as exc:
    st.sidebar.error(str(exc))
    st.stop()
    raise SystemExit

if not dates:
    st.sidebar.error("The selected price file does not contain any valid daily timestamps.")
    st.stop()
    raise SystemExit

min_available_date, max_available_date = available_date_bounds(dates)
available_date_text = f"{min_available_date.isoformat()} to {max_available_date.isoformat()}"
default_date = selected_date_or_fallback(pd.Timestamp("2025-11-01"), dates)
target_date = st.sidebar.date_input(
    "Target date",
    value=default_date,
    min_value=min_available_date,
    max_value=max_available_date,
    disabled=run_mode != "Daily",
    help=f"Calendar range from selected price data: {available_date_text}.",
)
target_date_available = date_is_available(target_date, dates)
if run_mode == "Daily" and not target_date_available:
    st.sidebar.error(f"No price data exists for {pd.Timestamp(target_date).date().isoformat()}.")

year_index = years.index(2025) if 2025 in years else max(len(years) - 1, 0)
year = st.sidebar.selectbox("Year", years, index=year_index, disabled=run_mode != "Annual")

st.sidebar.divider()
installed_capacity_mw = st.sidebar.number_input(
    "Installed capacity MW",
    min_value=0.05,
    max_value=5000.0,
    value=50.0,
    step=1.0,
)
power_mw = 1.0
duration_h = 2.0
energy_mwh = power_mw * duration_h
scale_capacity_mw = float(installed_capacity_mw)
st.sidebar.caption(
    f"Assumption: 2-hour battery duration, {installed_capacity_mw * duration_h:,.0f} MWh installed energy."
)

eta_ch = st.sidebar.slider("Charge efficiency", min_value=0.50, max_value=1.00, value=0.92, step=0.01)
eta_dis = st.sidebar.slider("Discharge efficiency", min_value=0.50, max_value=1.00, value=0.92, step=0.01)
soc_min_pct = st.sidebar.slider("SoC minimum percentage", 0, 95, 10, step=1)
soc_max_pct = st.sidebar.slider("SoC maximum percentage", 5, 100, 90, step=1)
soc_init_pct = st.sidebar.slider("Initial SoC percentage", 0, 100, 50, step=1)
terminal_label = st.sidebar.selectbox("Terminal SoC mode", ["equal to initial SoC", "free terminal SoC"])
terminal_soc_mode = "equal_initial" if terminal_label == "equal to initial SoC" else "free"

params_override = {
    "p_max": float(power_mw),
    "e_max": float(energy_mwh),
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
if degradation_lut_file is None:
    st.sidebar.error("PyBaMM degradation LUT was not found.")
temperature_c = 25.0

run_clicked = st.sidebar.button("Run Optimization", type="primary")

if run_clicked:
    errors = validate_parameters(params_override)
    if run_mode == "Daily" and not target_date_available:
        errors.append(f"No price data exists for {pd.Timestamp(target_date).date().isoformat()}.")
    if degradation_lut_file is None:
        errors.append("PyBaMM degradation LUT file is missing.")
    if errors:
        st.session_state["last_error"] = "\n".join(errors)
        st.session_state.pop("last_result", None)
    else:
        st.session_state.pop("last_error", None)
        try:
            with st.spinner("Solving MILP dispatch schedule..."):
                if run_mode == "Daily":
                    result = run_daily_optimization(
                        target_date=pd.Timestamp(target_date).date().isoformat(),
                        price_file=price_file,
                        degradation_lut_file=degradation_lut_file,
                        params_override=params_override,
                        degradation_source=degradation_source,
                        scale_capacity_mw=scale_capacity_mw,
                        temperature_c=temperature_c,
                        terminal_soc_mode=terminal_soc_mode,
                    )
                else:
                    result = run_annual_optimization(
                        year=int(year),
                        price_file=price_file,
                        degradation_lut_file=degradation_lut_file,
                        params_override=params_override,
                        degradation_source=degradation_source,
                        scale_capacity_mw=scale_capacity_mw,
                        temperature_c=temperature_c,
                        terminal_soc_mode=terminal_soc_mode,
                    )

                st.session_state["last_result"] = result
                st.session_state["last_config"] = {
                    "target_date": pd.Timestamp(target_date).date().isoformat(),
                    "price_file": price_file,
                    "degradation_lut_file": degradation_lut_file,
                    "params_override": params_override,
                    "degradation_source": degradation_source,
                    "scale_capacity_mw": scale_capacity_mw,
                    "temperature_c": temperature_c,
                    "terminal_soc_mode": terminal_soc_mode,
                    "degradation_cost_multiplier": 1.0,
                    "run_mode": run_mode,
                }
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.session_state.pop("last_result", None)

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

result = st.session_state.get("last_result")
summary = result["summary_dict"] if result is not None else None
schedule = result["dispatch_df"] if result is not None else None
params_used = result["params_used"] if result is not None else None

if result is not None:
    run_status = str(summary.get("solver_status", result.get("status", "Unknown")))
    if st.session_state.get("last_config", {}).get("run_mode") == "Annual":
        horizon_label = f"{int(summary.get('year', year))} annual horizon"
    else:
        horizon_label = str(summary.get("date", pd.Timestamp(target_date).date().isoformat()))
    header_degradation = str(result["files_used"].get("degradation_source", "N/A"))
    header_price = path_label(result["files_used"].get("price_file"))
else:
    run_status = "Not run"
    horizon_label = (
        pd.Timestamp(target_date).date().isoformat()
        if run_mode == "Daily"
        else f"{int(year)} annual horizon"
    )
    header_degradation = (
        f"PyBaMM LUT / {path_label(degradation_lut_file)}"
    )
    header_price = path_label(price_file)

executive_header(
    status=run_status,
    horizon=horizon_label,
    degradation_source=header_degradation,
    price_source=header_price,
)

if result is None:
    section_title("Strategy Overview")
    system_overview(infer_system_stages(price_file, degradation_lut_file, lut_files))

    section_title("Selected DAM Price Signal")
    preview_default_date = selected_date_or_fallback(pd.Timestamp(target_date), dates)
    preview_date = st.date_input(
        "Preview date",
        value=preview_default_date,
        min_value=min_available_date,
        max_value=max_available_date,
        help="Changes only the DAM price preview on this landing page.",
    )
    if date_is_available(preview_date, dates):
        render_price_preview(price_file, "Daily", preview_date, int(year))
    else:
        st.warning(f"No DAM price data exists for {pd.Timestamp(preview_date).date().isoformat()}.")
    st.stop()
    raise SystemExit

for warning in result.get("warnings", []):
    st.warning(warning)

if st.session_state.get("last_config", {}).get("run_mode") == "Annual":
    first_day = schedule["timestamp"].dt.normalize().min()
    chart_schedule = schedule[schedule["timestamp"].dt.normalize() == first_day].copy()
    chart_title = f"Annual Run Dispatch Excerpt - {first_day.date().isoformat()}"
else:
    chart_schedule = schedule
    chart_title = f"Daily Dispatch - {summary.get('date', '')}"

section_title("Headline KPIs")
render_result_kpis(summary, st.session_state.get("last_config", {}).get("run_mode", "Daily"))
st.caption("The above results are evaluated at 25 degrees Celsius, assuming an active cooling system.")

dispatch_fig, dispatch_kind = build_dispatch_chart(chart_schedule, params_used, chart_title)
if dispatch_kind == "plotly":
    st.plotly_chart(dispatch_fig, use_container_width=True)
else:
    st.pyplot(dispatch_fig, use_container_width=True)

financial_fig, financial_kind = build_financial_chart(summary)
if financial_fig is not None and financial_kind == "plotly":
    st.plotly_chart(financial_fig, use_container_width=True)
else:
    st.info("Install Plotly to see the interactive financial chart.")

section_title("Optimization Model Benchmark")
comparison = result.get("benchmark_comparison_df")
if comparison is None or comparison.empty:
    st.info("Benchmark comparison is unavailable for this run.")
else:
    st.dataframe(build_benchmark_display_table(comparison), use_container_width=True, hide_index=True)
    benchmark_fig, benchmark_kind = build_model_benchmark_chart(comparison)
    if benchmark_fig is not None and benchmark_kind == "plotly":
        st.plotly_chart(benchmark_fig, use_container_width=True)

section_title("Dispatch Table")
dispatch_table = build_table(schedule)
minimal_dispatch_table = build_minimal_dispatch_table(schedule)
st.dataframe(minimal_dispatch_table, use_container_width=True, hide_index=True, height=320)

with st.expander("Show full dispatch details", expanded=False):
    st.dataframe(dispatch_table, use_container_width=True, hide_index=True, height=410)

csv_schedule = dispatch_table.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download dispatch schedule CSV",
    csv_schedule,
    file_name="bess_dispatch_schedule.csv",
    mime="text/csv",
)
