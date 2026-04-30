from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .investment_model import (
    compute_cashflows,
    compute_investment_metrics,
    load_financial_inputs,
    metric_from_summary,
)
from .scenarios import SCENARIOS, apply_scenario
from .sensitivity import irr_vs_capex, npv_vs_discount_rate


DATA_PATH = Path(__file__).resolve().parents[1] / "data_final"
FINANCIAL_SUMMARY_CSV = DATA_PATH / "bess_financial_summary_fixed.csv"


@st.cache_data(show_spinner=False)
def _load_financial_summary(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _summary_frame(summary: dict | pd.DataFrame | None) -> pd.DataFrame | None:
    if summary is None:
        return None
    if isinstance(summary, pd.DataFrame):
        return summary.copy()
    return pd.DataFrame([summary])


def _resolve_summary(summary: dict | pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    session_summary = st.session_state.get("investment_summary_df")
    if isinstance(session_summary, pd.DataFrame) and not session_summary.empty:
        return session_summary.copy(), str(
            st.session_state.get("investment_source_label", "current optimization KPI output")
        )

    summary_df = _summary_frame(summary)
    if summary_df is not None and not summary_df.empty:
        return summary_df, "current optimization run"
    if FINANCIAL_SUMMARY_CSV.exists():
        return _load_financial_summary(str(FINANCIAL_SUMMARY_CSV)), str(FINANCIAL_SUMMARY_CSV)
    return None, str(FINANCIAL_SUMMARY_CSV)


def _style_chart(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="#08111f",
        plot_bgcolor="#0d1726",
        font={"color": "#e5eefb", "family": "Inter, Arial, sans-serif"},
        margin={"l": 48, "r": 24, "t": 48, "b": 36},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(148, 163, 184, 0.16)", zerolinecolor="rgba(148, 163, 184, 0.16)")
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.16)", zerolinecolor="rgba(148, 163, 184, 0.16)")
    return fig


def _money(value: float) -> str:
    return f"EUR {value:,.0f}"


def _percent(value: float) -> str:
    if value != value:
        return "N/A"
    return f"{value * 100.0:,.1f}%"


def _scenario_metrics_table(summary_df: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    base = compute_investment_metrics(summary_df, inputs)
    rows = []
    degradation_cost = metric_from_summary(
        summary_df,
        ["degradation_cost_eur", "park_degradation_cost_eur", "degradation_cost"],
    )
    for scenario_name in SCENARIOS:
        adjusted = apply_scenario({**base, "degradation_cost": degradation_cost}, scenario_name)
        scenario_inputs = dict(inputs)
        cashflows = compute_cashflows(float(adjusted["annual_profit"]), scenario_inputs)
        metrics = compute_investment_metrics(
            pd.DataFrame([{"net_profit_eur": float(adjusted["annual_profit"]) / 365.0}]),
            scenario_inputs,
        )
        rows.append(
            {
                "scenario": scenario_name,
                "annual_profit": float(adjusted["annual_profit"]),
                "degradation_cost": float(adjusted["degradation_cost"]),
                "npv": metrics["npv"],
                "irr": metrics["irr"],
                "payback": metrics["payback"],
                "terminal_cashflow": cashflows[-1],
            }
        )
    return pd.DataFrame(rows)


def _npv_chart(npv_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=npv_df["discount_rate"] * 100.0,
            y=npv_df["npv"],
            mode="lines",
            line={"color": "#60a5fa", "width": 3},
            hovertemplate="Discount rate: %{x:.1f}%<br>NPV: EUR %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(title="NPV vs Discount Rate", xaxis_title="Discount rate (%)", yaxis_title="NPV EUR")
    return _style_chart(fig)


def _irr_chart(irr_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=irr_df["CAPEX_per_MWh"],
            y=irr_df["irr"] * 100.0,
            mode="lines",
            line={"color": "#22c55e", "width": 3},
            hovertemplate="CAPEX: EUR %{x:,.0f}/MWh<br>IRR: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(title="IRR vs CAPEX", xaxis_title="CAPEX EUR/MWh", yaxis_title="IRR (%)")
    return _style_chart(fig)


def render_investment_tab(summary: dict | pd.DataFrame | None = None) -> None:
    st.subheader("Investment Analysis")
    uploaded_df = None
    with st.expander("Manual KPI CSV override", expanded=False):
        uploaded_summary = st.file_uploader(
            "Upload KPI summary CSV",
            type=["csv"],
            help="Optional fallback for analyzing a previously downloaded summary KPI CSV.",
        )
        if uploaded_summary is not None:
            uploaded_df = pd.read_csv(uploaded_summary)

    if summary is None:
        result = st.session_state.get("last_result")
        if isinstance(result, dict):
            summary = result.get("summary_dict")
    summary_df, source_label = _resolve_summary(summary)
    if uploaded_df is not None:
        summary_df, source_label = uploaded_df, uploaded_summary.name

    if summary_df is None:
        st.info(
            "Run an optimization to automatically analyze the in-memory KPI summary, "
            "or upload the KPI summary CSV downloaded from the Data tab."
        )
        return
    st.caption(f"Financial basis: {source_label}")
    defaults = load_financial_inputs()

    with st.sidebar.expander("Investment Inputs", expanded=False):
        capex_per_mwh = st.number_input(
            "CAPEX per MWh",
            min_value=0.0,
            value=float(defaults["CAPEX_per_MWh"]),
            step=25000.0,
        )
        discount_rate = st.slider(
            "Discount rate",
            min_value=0.0,
            max_value=0.25,
            value=float(defaults["discount_rate"]),
            step=0.005,
            format="%.3f",
        )
        lifetime_years = st.number_input(
            "Lifetime years",
            min_value=1,
            max_value=40,
            value=int(defaults["lifetime_years"]),
            step=1,
        )
        battery_capacity_mwh = st.number_input(
            "Battery capacity MWh",
            min_value=0.1,
            value=float(defaults["battery_capacity_MWh"]),
            step=0.5,
        )
        scenario_name = st.selectbox("Investment scenario", list(SCENARIOS), index=0)

    inputs = {
        **defaults,
        "CAPEX_per_MWh": float(capex_per_mwh),
        "discount_rate": float(discount_rate),
        "lifetime_years": int(lifetime_years),
        "battery_capacity_MWh": float(battery_capacity_mwh),
    }

    metrics = compute_investment_metrics(summary_df, inputs)
    scenario_adjusted = apply_scenario(metrics, scenario_name)
    scenario_summary = pd.DataFrame(
        [{"net_profit_eur": float(scenario_adjusted["annual_profit"]) / 365.0}]
    )
    display_metrics = compute_investment_metrics(scenario_summary, inputs)
    cashflows = compute_cashflows(display_metrics["annual_profit"], inputs)

    cols = st.columns(4)
    cols[0].metric("IRR", _percent(display_metrics["irr"]))
    cols[1].metric("NPV", _money(display_metrics["npv"]))
    cols[2].metric("Payback", f"{display_metrics['payback']:,.2f} years")
    cols[3].metric("Annual Profit", _money(display_metrics["annual_profit"]))

    scenario_df = _scenario_metrics_table(summary_df, inputs)
    st.dataframe(
        scenario_df.assign(
            annual_profit=lambda df: df["annual_profit"].round(0),
            degradation_cost=lambda df: df["degradation_cost"].round(0),
            npv=lambda df: df["npv"].round(0),
            irr=lambda df: (df["irr"] * 100.0).round(2),
            payback=lambda df: df["payback"].round(2),
            terminal_cashflow=lambda df: df["terminal_cashflow"].round(0),
        ),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_npv_chart(npv_vs_discount_rate(cashflows)), use_container_width=True)
    with right:
        capex_range = [capex_per_mwh * factor for factor in (0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4)]
        irr_inputs = {**inputs, "annual_profit": display_metrics["annual_profit"]}
        st.plotly_chart(_irr_chart(irr_vs_capex(capex_range, irr_inputs)), use_container_width=True)
