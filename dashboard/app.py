"""Streamlit hackathon dashboard for the BESS optimizer."""

from __future__ import annotations

from datetime import date
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from investment import render_investment_tab

try:
    from .decision_support import (
        SCENARIOS,
        STRATEGIES,
        apply_decision_view,
        build_benchmark_chart,
        build_benchmark_table,
        build_cumulative_profit_chart,
        build_dispatch_inspection_chart,
        data_quality_checks,
        explain_dispatch_row,
    )
    from .charts import (
        build_dispatch_chart,
        build_financial_chart,
        build_price_preview_chart,
        build_scenario_chart,
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
        section_title,
        system_overview,
    )
except ImportError:
    from decision_support import (
        SCENARIOS,
        STRATEGIES,
        apply_decision_view,
        build_benchmark_chart,
        build_benchmark_table,
        build_cumulative_profit_chart,
        build_dispatch_inspection_chart,
        data_quality_checks,
        explain_dispatch_row,
    )
    from charts import (
        build_dispatch_chart,
        build_financial_chart,
        build_price_preview_chart,
        build_scenario_chart,
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


def render_kpis(summary: dict) -> None:
    solver_status = str(summary.get("solver_status", "Unknown"))
    net_profit = scaled(summary, "net_profit_eur", "park_net_profit_eur")
    values = [
        ("Solver status", solver_status, "positive" if solver_status.lower() == "optimal" else "negative"),
        ("Net profit EUR", eur(net_profit), "positive" if net_profit >= 0 else "negative"),
        ("Gross revenue EUR", eur(scaled(summary, "gross_revenue_eur", "park_gross_revenue_eur")), "positive"),
        ("Purchase cost EUR", eur(scaled(summary, "gross_purchase_eur", "park_gross_purchase_eur")), "negative"),
        ("Degradation cost EUR", eur(scaled(summary, "degradation_cost_eur", "park_degradation_cost_eur")), "negative"),
        ("Bought energy MWh", num(scaled(summary, "buy_energy_mwh", "park_buy_energy_mwh"), 2), "accent"),
        ("Sold energy MWh", num(scaled(summary, "sell_energy_mwh", "park_sell_energy_mwh"), 2), "positive"),
        ("Final SoC %", num(float(summary.get("final_soc_pct", 0.0)) * 100.0, 1), "positive"),
        ("Charge intervals", f"{int(summary.get('buy_intervals', 0))}", "accent"),
        ("Discharge intervals", f"{int(summary.get('sell_intervals', 0))}", "positive"),
        ("Equivalent cycles", num(float(summary.get("equivalent_discharge_cycles", 0.0)), 3), "accent"),
        ("Throughput MWh", num(scaled(summary, "total_throughput_mwh", "park_total_throughput_mwh"), 2), "accent"),
    ]

    headline_kpi_grid(values)


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


def scenario_row(name: str, result: dict | None, available: bool, note: str = "") -> dict:
    if not available or result is None:
        return {
            "scenario": name,
            "available": False,
            "status": "Unavailable",
            "net_profit_eur": np.nan,
            "degradation_cost_eur": np.nan,
            "bought_mwh": np.nan,
            "sold_mwh": np.nan,
            "throughput_mwh": np.nan,
            "final_soc_pct": np.nan,
            "note": note,
        }

    summary = result["summary_dict"]
    return {
        "scenario": name,
        "available": True,
        "status": str(summary.get("solver_status", result.get("status", "Unknown"))),
        "net_profit_eur": scaled(summary, "net_profit_eur", "park_net_profit_eur"),
        "degradation_cost_eur": scaled(summary, "degradation_cost_eur", "park_degradation_cost_eur"),
        "bought_mwh": scaled(summary, "buy_energy_mwh", "park_buy_energy_mwh"),
        "sold_mwh": scaled(summary, "sell_energy_mwh", "park_sell_energy_mwh"),
        "throughput_mwh": scaled(summary, "total_throughput_mwh", "park_total_throughput_mwh"),
        "final_soc_pct": float(summary.get("final_soc_pct", 0.0)) * 100.0,
        "note": note,
    }


def run_scenario_comparison(base_result: dict, config: dict, lut_files: list[Path]) -> pd.DataFrame:
    rows = [scenario_row("Base case", base_result, True, "Current selected parameters")]
    base_params = dict(config["params_override"])
    scale_capacity_mw = config["scale_capacity_mw"]
    run_mode_value = str(config.get("run_mode", "Daily"))

    pybamm_lut = default_lut_for_source("pybamm_only", lut_files)
    conservative_source = config["degradation_source"]
    conservative_lut = config["degradation_lut_file"]
    conservative_multiplier = 1.5
    conservative_note = "Same LUT with 1.5x degradation cost"
    if pybamm_lut is not None and config["degradation_source"] != "pybamm_only":
        conservative_source = "pybamm_only"
        conservative_lut = pybamm_lut
        conservative_multiplier = 1.0
        conservative_note = "PyBaMM-only degradation LUT"

    scenarios = [
        {
            "name": "Conservative degradation",
            "params": base_params,
            "source": conservative_source,
            "lut": conservative_lut,
            "multiplier": conservative_multiplier,
            "note": conservative_note,
        },
        {
            "name": "Higher power / lower duration",
            "params": {**base_params, "e_max": max(float(base_params["p_max"]) * 1.0, 0.1)},
            "source": config["degradation_source"],
            "lut": config["degradation_lut_file"],
            "multiplier": config["degradation_cost_multiplier"],
            "note": "One-hour battery at selected power",
        },
        {
            "name": "Lower power / longer duration",
            "params": {**base_params, "e_max": max(float(base_params["p_max"]) * 4.0, 0.1)},
            "source": config["degradation_source"],
            "lut": config["degradation_lut_file"],
            "multiplier": config["degradation_cost_multiplier"],
            "note": "Four-hour battery at selected power",
        },
        {
            "name": "No degradation cost",
            "params": base_params,
            "source": "zero",
            "lut": None,
            "multiplier": 1.0,
            "note": "For comparison only",
        },
    ]

    for item in scenarios:
        try:
            if run_mode_value == "Annual":
                result = run_annual_optimization(
                    year=int(config["year"]),
                    price_file=config["price_file"],
                    degradation_lut_file=item["lut"],
                    params_override=item["params"],
                    degradation_source=item["source"],
                    scale_capacity_mw=scale_capacity_mw,
                    temperature_c=config["temperature_c"],
                    terminal_soc_mode=config["terminal_soc_mode"],
                    degradation_cost_multiplier=item["multiplier"],
                )
            else:
                result = run_daily_optimization(
                    target_date=config["target_date"],
                    price_file=config["price_file"],
                    degradation_lut_file=item["lut"],
                    params_override=item["params"],
                    degradation_source=item["source"],
                    scale_capacity_mw=scale_capacity_mw,
                    temperature_c=config["temperature_c"],
                    terminal_soc_mode=config["terminal_soc_mode"],
                    degradation_cost_multiplier=item["multiplier"],
                )
            rows.append(scenario_row(item["name"], result, True, item["note"]))
        except Exception as exc:
            rows.append(scenario_row(item["name"], None, False, str(exc)))

    return pd.DataFrame(rows)


def path_label(path: Path | str | None) -> str:
    if path is None:
        return "N/A"
    return display_path(Path(path))


def infer_system_stages(
    price_file: Path,
    degradation_lut_file: Path | None,
    available_lut_files: list[Path],
) -> list[dict[str, str]]:
    root = Path(__file__).resolve().parents[1]
    pybamm_files = [
        root / "src" / "bess_twin" / "pybamm_only_dod_degradation_lut.py",
        root / "src" / "bess_twin" / "offline_pybamm_benchmark_calibrator.py",
    ]
    oxford_files = sorted((root / "src" / "oxford_data_analysis").glob("build_*.py"))
    offline_paths = [display_path(path) for path in pybamm_files if path.exists()]
    if oxford_files:
        offline_paths.append("src/oxford_data_analysis/build_*.py")

    lut_path = degradation_lut_file or default_lut_for_source("pybamm_only", available_lut_files)
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
    st.error("No price CSV files were found under optimization/data/cleaned_data.")
    st.stop()
    raise SystemExit

st.sidebar.title("Optimizer Controls")
run_mode = st.sidebar.radio("Run mode", ["Daily", "Annual"], index=0, horizontal=True)

price_options = {display_path(path): path for path in price_files}
price_label = st.sidebar.selectbox("Price input file", list(price_options.keys()))
price_file = price_options[price_label]

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
power_mw = st.sidebar.number_input("Battery power in MW", min_value=0.05, max_value=500.0, value=1.0, step=0.05)
energy_mwh = st.sidebar.number_input("Battery energy in MWh", min_value=0.05, max_value=2000.0, value=2.0, step=0.05)
duration_h = energy_mwh / power_mw if power_mw else 0.0
st.sidebar.caption(f"Duration: {duration_h:.2f} hours")

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

st.sidebar.divider()
degradation_source = st.sidebar.selectbox("Degradation source", ["pybamm_only", "lut", "dummy"], index=0)
lut_options = {display_path(path): path for path in lut_files}
preferred_lut = default_lut_for_source(degradation_source, lut_files)
lut_labels = list(lut_options.keys())
lut_index = lut_labels.index(display_path(preferred_lut)) if preferred_lut is not None and display_path(preferred_lut) in lut_labels else 0
selected_lut_label = st.sidebar.selectbox(
    "Degradation LUT file",
    lut_labels if lut_labels else ["No LUT files found"],
    index=lut_index if lut_labels else 0,
    disabled=degradation_source == "dummy" or not lut_labels,
)
degradation_lut_file = None if degradation_source == "dummy" or not lut_labels else lut_options[selected_lut_label]
temperature_c = st.sidebar.number_input("Temperature assumption C", min_value=-20.0, max_value=80.0, value=25.0, step=1.0)

st.sidebar.divider()
scale_to_capacity = st.sidebar.checkbox("Scale to installed capacity", value=True)
installed_capacity_mw = st.sidebar.number_input(
    "Installed capacity MW",
    min_value=0.05,
    max_value=5000.0,
    value=50.0,
    step=1.0,
    disabled=not scale_to_capacity,
)
scale_capacity_mw = float(installed_capacity_mw) if scale_to_capacity else None
st.sidebar.info("Solver: CBC through PuLP")
run_scenarios = st.sidebar.checkbox(
    "Run scenario comparison",
    value=run_mode == "Daily",
    help="Annual scenario comparison follows the same pipeline as daily, but runs multiple annual optimizations and can take longer.",
)

st.sidebar.divider()
with st.sidebar.expander("Decision View", expanded=False):
    decision_scenario = st.selectbox("Scenario", list(SCENARIOS.keys()), index=0)
    decision_strategy = st.selectbox("Strategy", list(STRATEGIES.keys()), index=0)
show_degradation_overlay = False
normalize_per_mwh = False

run_clicked = st.sidebar.button("Run Optimization", type="primary")

if run_clicked:
    errors = validate_parameters(params_override)
    if run_mode == "Daily" and not target_date_available:
        errors.append(f"No price data exists for {pd.Timestamp(target_date).date().isoformat()}.")
    if degradation_source != "dummy" and degradation_lut_file is None:
        errors.append("Select a valid degradation LUT file.")
    if errors:
        st.session_state["last_error"] = "\n".join(errors)
        st.session_state.pop("last_result", None)
        st.session_state.pop("investment_summary_df", None)
        st.session_state.pop("investment_source_label", None)
        st.session_state.pop("scenario_comparison", None)
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
                st.session_state["investment_summary_df"] = pd.DataFrame([result["summary_dict"]])
                st.session_state["investment_source_label"] = "current optimization KPI output"
                st.session_state["last_config"] = {
                    "target_date": pd.Timestamp(target_date).date().isoformat(),
                    "year": int(year),
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

                if run_scenarios:
                    st.session_state["scenario_comparison"] = run_scenario_comparison(
                        result,
                        st.session_state["last_config"],
                        lut_files,
                    )
                else:
                    st.session_state.pop("scenario_comparison", None)
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.session_state.pop("last_result", None)
            st.session_state.pop("investment_summary_df", None)
            st.session_state.pop("investment_source_label", None)
            st.session_state.pop("scenario_comparison", None)

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
        "synthetic dummy curve"
        if degradation_source == "dummy"
        else f"{degradation_source} / {path_label(degradation_lut_file)}"
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
    section_title("Investment Analysis")
    render_investment_tab()
    st.stop()
    raise SystemExit

for warning in result.get("warnings", []):
    st.warning(warning)

if st.session_state.get("last_config", {}).get("run_mode") == "Annual":
    st.info("Annual mode solved the full year. The dispatch chart below shows the first optimized day for readability.")
    first_day = schedule["timestamp"].dt.normalize().min()
    chart_schedule = schedule[schedule["timestamp"].dt.normalize() == first_day].copy()
    chart_title = f"Annual Run Dispatch Excerpt - {first_day.date().isoformat()}"
else:
    chart_schedule = schedule
    chart_title = f"Daily Dispatch - {summary.get('date', '')}"

decision_schedule, decision_summary = apply_decision_view(
    schedule,
    summary,
    decision_scenario,
    decision_strategy,
    normalize_per_mwh,
)
benchmark_df = build_benchmark_table(decision_schedule, decision_summary)
quality_df = data_quality_checks(schedule, params_used)

overview_tab, dispatch_tab, scenario_tab, data_tab, investment_tab = st.tabs(
    ["Overview", "Dispatch", "Scenarios", "Data", "Investment Analysis"]
)

with overview_tab:
    section_title("Headline KPIs")
    render_kpis(summary)
    section_title("Decision View Snapshot")
    compact_kpi_strip(
        [
            ("Scenario", decision_scenario, "accent"),
            ("Strategy", decision_strategy, "accent"),
            ("View net profit", f"EUR {decision_summary['view_net_profit_eur']:,.0f}", "positive" if decision_summary["view_net_profit_eur"] >= 0 else "negative"),
            ("View degradation", f"EUR {decision_summary['view_degradation_cost_eur']:,.0f}", "negative"),
        ]
    )
    cumulative_fig = build_cumulative_profit_chart(decision_schedule)
    if cumulative_fig is not None:
        st.plotly_chart(cumulative_fig, use_container_width=True)
    financial_fig, financial_kind = build_financial_chart(summary)
    if financial_fig is not None and financial_kind == "plotly":
        st.plotly_chart(financial_fig, use_container_width=True)
    else:
        st.info("Install Plotly to see the interactive financial chart.")

with dispatch_tab:
    section_title("Dispatch Schedule")
    dispatch_fig, dispatch_kind = build_dispatch_chart(chart_schedule, params_used, chart_title)
    if dispatch_kind == "plotly":
        st.plotly_chart(dispatch_fig, use_container_width=True)
    else:
        st.pyplot(dispatch_fig, use_container_width=True)

    section_title("Dispatch Inspection")
    inspection_fig = build_dispatch_inspection_chart(decision_schedule, show_degradation_overlay)
    if inspection_fig is not None:
        st.plotly_chart(inspection_fig, use_container_width=True)
    timestamp_options = decision_schedule["timestamp"].astype(str).tolist()
    selected_timestamp = st.selectbox("Inspect timestamp", timestamp_options, index=min(len(timestamp_options) // 2, max(len(timestamp_options) - 1, 0)))
    selected_row = decision_schedule[decision_schedule["timestamp"].astype(str) == selected_timestamp].iloc[0]
    explanation = explain_dispatch_row(selected_row, decision_schedule, params_used)
    explain_cols = st.columns(5)
    explain_cols[0].metric("Action", explanation["action"])
    explain_cols[1].metric("Price", explanation["price"])
    explain_cols[2].metric("SoC", explanation["soc"])
    explain_cols[3].metric("Degradation", explanation["degradation_cost"])
    explain_cols[4].metric("Net profit", explanation["net_profit"])
    constraint_grid(
        [
            ("Reason", explanation["reason"], True),
            ("Price context", explanation["price_context"], True),
            ("SoC condition", explanation["soc_condition"], True),
            ("Degradation penalty", explanation["degradation_penalty"], True),
            ("Power limit", explanation["power_limit"], True),
        ]
    )

with scenario_tab:
    section_title("Scenario Comparison")
    comparison = st.session_state.get("scenario_comparison")
    if comparison is None:
        if st.session_state.get("last_config", {}).get("run_mode") == "Annual":
            st.info("Scenario comparison is available in Daily mode to keep the demo responsive.")
        else:
            st.info("Enable scenario comparison in the sidebar and run the optimizer.")
    else:
        display_comparison = comparison.copy()
        for col in ["net_profit_eur", "degradation_cost_eur", "bought_mwh", "sold_mwh", "throughput_mwh", "final_soc_pct"]:
            display_comparison[col] = display_comparison[col].round(3)
        st.dataframe(display_comparison, use_container_width=True, hide_index=True)
        scenario_fig, scenario_kind = build_scenario_chart(comparison)
        if scenario_fig is not None and scenario_kind == "plotly":
            st.plotly_chart(scenario_fig, use_container_width=True)

    section_title("Benchmark Module")
    benchmark_display = benchmark_df.copy()
    for col in ["net_profit", "degradation_cost", "profit_after_degradation"]:
        benchmark_display[col] = benchmark_display[col].round(2)
    benchmark_display["equivalent_cycles"] = benchmark_display["equivalent_cycles"].round(3)
    benchmark_display["final_soh_proxy"] = (benchmark_display["final_soh_proxy"] * 100.0).round(2)
    st.dataframe(benchmark_display, use_container_width=True, hide_index=True)
    benchmark_fig = build_benchmark_chart(benchmark_df)
    if benchmark_fig is not None:
        st.plotly_chart(benchmark_fig, use_container_width=True)

with data_tab:
    section_title("Dispatch Table")
    dispatch_table = build_table(schedule)
    st.dataframe(dispatch_table, use_container_width=True, hide_index=True, height=410)

    csv_schedule = dispatch_table.to_csv(index=False).encode("utf-8")
    csv_summary = pd.DataFrame([summary]).to_csv(index=False).encode("utf-8")
    download_a, download_b = st.columns(2)
    download_a.download_button(
        "Download dispatch schedule CSV",
        csv_schedule,
        file_name="bess_dispatch_schedule.csv",
        mime="text/csv",
    )
    download_b.download_button(
        "Download summary KPIs CSV",
        csv_summary,
        file_name="bess_summary_kpis.csv",
        mime="text/csv",
    )
    section_title("Data Quality Panel")
    st.dataframe(quality_df, use_container_width=True, hide_index=True)

with investment_tab:
    render_investment_tab()
