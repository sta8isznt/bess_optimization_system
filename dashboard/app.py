"""Streamlit hackathon dashboard for the BESS optimizer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from charts import build_dispatch_chart, build_financial_chart, build_scenario_chart
from optimizer_adapter import (
    DashboardOptimizerError,
    available_dates,
    available_years,
    default_lut_for_source,
    display_path,
    list_lut_files,
    list_price_files,
    run_annual_optimization,
    run_daily_optimization,
    validate_parameters,
)
from styles import constraint_grid, hero, inject_css, section_title


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


def eur(value: float) -> str:
    return f"{value:,.0f}"


def num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def scaled(summary: dict, base_key: str, park_key: str) -> float:
    if float(summary.get("scale_factor", 1.0)) != 1.0 and park_key in summary:
        return float(summary[park_key])
    return float(summary.get(base_key, 0.0))


def render_kpis(summary: dict) -> None:
    values = [
        ("Solver status", str(summary.get("solver_status", "Unknown"))),
        ("Net profit EUR", eur(scaled(summary, "net_profit_eur", "park_net_profit_eur"))),
        ("Gross revenue EUR", eur(scaled(summary, "gross_revenue_eur", "park_gross_revenue_eur"))),
        ("Purchase cost EUR", eur(scaled(summary, "gross_purchase_eur", "park_gross_purchase_eur"))),
        ("Degradation cost EUR", eur(scaled(summary, "degradation_cost_eur", "park_degradation_cost_eur"))),
        ("Bought energy MWh", num(scaled(summary, "buy_energy_mwh", "park_buy_energy_mwh"), 2)),
        ("Sold energy MWh", num(scaled(summary, "sell_energy_mwh", "park_sell_energy_mwh"), 2)),
        ("Final SoC %", num(float(summary.get("final_soc_pct", 0.0)) * 100.0, 1)),
        ("Charge intervals", f"{int(summary.get('buy_intervals', 0))}"),
        ("Discharge intervals", f"{int(summary.get('sell_intervals', 0))}"),
        ("Equivalent cycles", num(float(summary.get("equivalent_discharge_cycles", 0.0)), 3)),
        ("Throughput MWh", num(scaled(summary, "total_throughput_mwh", "park_total_throughput_mwh"), 2)),
    ]

    for start in range(0, len(values), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, values[start:start + 4]):
            col.metric(label, value)


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


def data_scarcity_story() -> None:
    section_title("Why This Works Under Data Scarcity")
    st.markdown(
        """
        <div class="story-panel">
        <ul>
          <li>We do not need historical battery operation data for every possible dispatch.</li>
          <li>The MILP uses known asset constraints: power, energy, SoC limits, and efficiencies.</li>
          <li>DAM prices provide the market signal for arbitrage decisions.</li>
          <li>PyBaMM/Oxford-derived LUTs approximate marginal degradation cost offline.</li>
          <li>The optimizer produces physically feasible schedules even with limited telemetry.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_css()
hero()

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

default_date = pd.Timestamp("2025-11-01")
date_index = dates.index(default_date) if default_date in dates else max(len(dates) - 1, 0)
target_date = st.sidebar.selectbox(
    "Target date",
    dates,
    index=date_index,
    format_func=lambda d: pd.Timestamp(d).date().isoformat(),
    disabled=run_mode != "Daily",
)

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
run_scenarios = st.sidebar.checkbox("Run scenario comparison", value=True, disabled=run_mode != "Daily")

run_clicked = st.sidebar.button("Run Optimization", type="primary")

if run_clicked:
    errors = validate_parameters(params_override)
    if degradation_source != "dummy" and degradation_lut_file is None:
        errors.append("Select a valid degradation LUT file.")
    if errors:
        st.session_state["last_error"] = "\n".join(errors)
        st.session_state.pop("last_result", None)
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

                if run_mode == "Daily" and run_scenarios:
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
            st.session_state.pop("scenario_comparison", None)

if st.session_state.get("last_error"):
    st.error(st.session_state["last_error"])

result = st.session_state.get("last_result")

if result is None:
    section_title("Demo Workflow")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Price data", f"{len(dates)} days")
    col_b.metric("LUT files", f"{len(lut_files)} available")
    col_c.metric("Default asset", "1 MW / 2 MWh")
    st.markdown(
        """
        <div class="soft-panel">
        Select a date, choose the degradation source, then run the optimizer. The dashboard will show
        the dispatch schedule, degradation-aware economics, constraints, downloadable outputs, and
        what-if scenario comparison.
        </div>
        """,
        unsafe_allow_html=True,
    )
    data_scarcity_story()
    st.stop()
    raise SystemExit

summary = result["summary_dict"]
schedule = result["dispatch_df"]
params_used = result["params_used"]

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

section_title("Headline KPIs")
render_kpis(summary)

section_title("Dispatch Schedule")
dispatch_fig, dispatch_kind = build_dispatch_chart(chart_schedule, params_used, chart_title)
if dispatch_kind == "plotly":
    st.plotly_chart(dispatch_fig, use_container_width=True)
else:
    st.pyplot(dispatch_fig, use_container_width=True)

left, right = st.columns([1.05, 0.95])
with left:
    section_title("Financial Breakdown")
    financial_fig, financial_kind = build_financial_chart(summary)
    if financial_fig is not None and financial_kind == "plotly":
        st.plotly_chart(financial_fig, use_container_width=True)
    else:
        st.info("Install Plotly to see the interactive financial chart.")

with right:
    section_title("Constraint Summary")
    render_constraint_summary(summary, params_used, result["files_used"])

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

data_scarcity_story()
