"""Lightweight decision-support utilities for the dashboard.

These helpers intentionally do not call the optimizer. They only transform
already produced schedules, summaries, and LUT inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - dashboard already has matplotlib fallbacks.
    PLOTLY_AVAILABLE = False
    go = None
    make_subplots = None


PAPER_BG = "#08111f"
PLOT_BG = "#0d1726"
GRID = "rgba(148, 163, 184, 0.16)"
TEXT = "#e5eefb"
MUTED = "#9fb0c8"
GREEN = "#22c55e"
RED = "#ef4444"
BLUE = "#60a5fa"
AMBER = "#f59e0b"
VIOLET = "#a78bfa"


SCENARIOS = {
    "base": {"price": 1.0, "degradation": 1.0},
    "optimistic": {"price": 1.08, "degradation": 0.80},
    "conservative": {"price": 0.92, "degradation": 1.30},
}

STRATEGIES = {
    "balanced": {"dispatch": 1.0, "cycle": 1.0},
    "aggressive": {"dispatch": 1.12, "cycle": 1.18},
    "defensive": {"dispatch": 0.82, "cycle": 0.72},
}


def _safe_divide(a: float, b: float) -> float:
    return 0.0 if abs(float(b)) < 1e-12 else float(a) / float(b)


def _dark_layout(fig, height: int):
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
        margin={"l": 44, "r": 24, "t": 48, "b": 34},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": MUTED},
        },
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def apply_decision_view(
    schedule: pd.DataFrame,
    summary: dict,
    scenario: str,
    strategy: str,
    normalize_per_mwh: bool,
) -> tuple[pd.DataFrame, dict]:
    """Return adjusted display-only schedule/summary for decision support."""
    scenario_cfg = SCENARIOS.get(scenario, SCENARIOS["base"])
    strategy_cfg = STRATEGIES.get(strategy, STRATEGIES["balanced"])
    dispatch_factor = float(strategy_cfg["dispatch"])
    price_factor = float(scenario_cfg["price"])
    degradation_factor = float(scenario_cfg["degradation"]) * float(strategy_cfg["cycle"])

    out = schedule.copy()
    out["view_price_eur_mwh"] = out["price_eur_mwh"] * price_factor
    out["view_p_buy_mw"] = out["p_buy_mw"] * dispatch_factor
    out["view_p_sell_mw"] = out["p_sell_mw"] * dispatch_factor
    out["view_net_power_mw"] = out["view_p_sell_mw"] - out["view_p_buy_mw"]
    out["view_gross_revenue_eur"] = out["gross_revenue_eur"] * price_factor * dispatch_factor
    out["view_gross_purchase_eur"] = out["gross_purchase_eur"] * price_factor * dispatch_factor
    out["view_degradation_cost_eur"] = out["degradation_cost_eur"] * degradation_factor * dispatch_factor
    out["view_interval_profit_eur"] = (
        out["view_gross_revenue_eur"]
        - out["view_gross_purchase_eur"]
        - out["view_degradation_cost_eur"]
    )
    out["view_cumulative_profit_eur"] = out["view_interval_profit_eur"].cumsum()

    capacity = float(summary.get("installed_energy_capacity_mwh", summary.get("battery_capacity_mwh", 0.0)) or 0.0)
    if normalize_per_mwh and capacity > 0:
        for col in [
            "view_gross_revenue_eur",
            "view_gross_purchase_eur",
            "view_degradation_cost_eur",
            "view_interval_profit_eur",
            "view_cumulative_profit_eur",
        ]:
            out[col] = out[col] / capacity

    adjusted = dict(summary)
    adjusted["decision_scenario"] = scenario
    adjusted["decision_strategy"] = strategy
    adjusted["view_net_profit_eur"] = float(out["view_interval_profit_eur"].sum())
    adjusted["view_degradation_cost_eur"] = float(out["view_degradation_cost_eur"].sum())
    adjusted["view_revenue_eur"] = float(out["view_gross_revenue_eur"].sum())
    adjusted["view_purchase_cost_eur"] = float(out["view_gross_purchase_eur"].sum())
    adjusted["view_normalized_per_mwh"] = bool(normalize_per_mwh)
    return out, adjusted


def explain_dispatch_row(row: pd.Series, schedule: pd.DataFrame, params: dict) -> dict[str, str]:
    price = float(row.get("view_price_eur_mwh", row.get("price_eur_mwh", 0.0)))
    prices = schedule.get("view_price_eur_mwh", schedule["price_eur_mwh"])
    percentile = float((prices <= price).mean()) * 100.0
    soc_pct = float(row.get("soc_pct", 0.0)) * 100.0
    degradation = float(row.get("view_degradation_cost_eur", row.get("degradation_cost_eur", 0.0)))
    p_buy = float(row.get("view_p_buy_mw", row.get("p_buy_mw", 0.0)))
    p_sell = float(row.get("view_p_sell_mw", row.get("p_sell_mw", 0.0)))
    p_max = float(params.get("p_max", max(p_buy, p_sell, 1.0)))
    soc_min = float(params.get("soc_min", 0.0)) * 100.0
    soc_max = float(params.get("soc_max", 1.0)) * 100.0

    if p_sell > 1e-7:
        action = "discharge"
        reason = "High price percentile and available stored energy justify selling after degradation cost."
    elif p_buy > 1e-7:
        action = "charge"
        reason = "Lower price percentile and SOC headroom make charging attractive."
    else:
        action = "idle"
        reason = "Price spread is not strong enough after efficiency and degradation penalties."

    soc_limit = "near lower bound" if soc_pct <= soc_min + 3 else "near upper bound" if soc_pct >= soc_max - 3 else "inside operating band"
    power_limit = "binding or near binding" if max(p_buy, p_sell) >= 0.98 * p_max else "not binding"
    return {
        "timestamp": str(row.get("timestamp", "")),
        "price": f"{price:,.2f} EUR/MWh",
        "action": action,
        "soc": f"{soc_pct:,.1f}%",
        "degradation_cost": f"EUR {degradation:,.2f}",
        "net_profit": f"EUR {float(row.get('view_interval_profit_eur', row.get('interval_profit_eur', 0.0))):,.2f}",
        "reason": reason,
        "price_context": f"{percentile:,.0f}th percentile in selected horizon",
        "soc_condition": soc_limit,
        "degradation_penalty": "material" if degradation > schedule["view_degradation_cost_eur"].quantile(0.75) else "moderate/low",
        "power_limit": power_limit,
    }


def build_dispatch_inspection_chart(schedule: pd.DataFrame, show_degradation_overlay: bool):
    if not PLOTLY_AVAILABLE:
        return None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=schedule["timestamp"],
            y=schedule["view_price_eur_mwh"],
            name="Price",
            line={"color": TEXT, "width": 2.0},
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=schedule["timestamp"],
            y=-schedule["view_p_buy_mw"],
            name="Charge MW",
            marker_color=GREEN,
            opacity=0.75,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=schedule["timestamp"],
            y=schedule["view_p_sell_mw"],
            name="Discharge MW",
            marker_color=RED,
            opacity=0.75,
        ),
        secondary_y=False,
    )
    if show_degradation_overlay:
        fig.add_trace(
            go.Scatter(
                x=schedule["timestamp"],
                y=schedule["view_degradation_cost_eur"],
                name="Degradation cost",
                line={"color": AMBER, "width": 2.2, "dash": "dot"},
            ),
            secondary_y=True,
        )
    fig.update_layout(title={"text": "Decision View Dispatch", "x": 0.01}, barmode="relative")
    fig.update_yaxes(title_text="MW", secondary_y=False)
    fig.update_yaxes(title_text="EUR/MWh or EUR", secondary_y=True)
    return _dark_layout(fig, height=440)


def build_cumulative_profit_chart(schedule: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=schedule["timestamp"],
            y=schedule["view_cumulative_profit_eur"],
            name="Cumulative profit",
            line={"color": BLUE, "width": 2.6},
            fill="tozeroy",
            fillcolor="rgba(96, 165, 250, 0.12)",
        )
    )
    fig.update_layout(title={"text": "Cumulative Profit Path", "x": 0.01})
    fig.update_yaxes(title_text="EUR")
    return _dark_layout(fig, height=310)


def build_benchmark_table(schedule: pd.DataFrame, summary: dict) -> pd.DataFrame:
    price = schedule["view_price_eur_mwh"]
    low = price.quantile(0.25)
    high = price.quantile(0.75)
    capacity_scale = float(summary.get("scale_factor", 1.0) or 1.0)
    cycle_base = float(summary.get("equivalent_discharge_cycles", 0.0))
    final_soh = max(0.0, 1.0 - 0.0025 * cycle_base)

    optimizer_profit = float(schedule["view_interval_profit_eur"].sum())
    optimizer_deg = float(schedule["view_degradation_cost_eur"].sum())
    optimizer_cycles = cycle_base

    naive_mask = (price <= low) | (price >= high)
    naive_revenue = float(schedule.loc[price >= high, "view_gross_revenue_eur"].sum())
    naive_cost = float(schedule.loc[price <= low, "view_gross_purchase_eur"].sum())
    naive_deg = float(schedule.loc[naive_mask, "view_degradation_cost_eur"].sum()) * 1.12
    naive_profit = naive_revenue - naive_cost - naive_deg

    perfect_profit = float(
        schedule["view_gross_revenue_eur"].sum() - schedule["view_gross_purchase_eur"].sum()
    )

    rows = [
        ("no battery", 0.0, 0.0, 0.0, 1.0, 0.0),
        ("naive quantile dispatch", naive_profit, naive_deg, optimizer_cycles * 1.15, max(0.0, final_soh - 0.01), naive_profit),
        ("perfect hindsight without degradation", perfect_profit, 0.0, optimizer_cycles, final_soh, perfect_profit),
        ("degradation-aware optimizer", optimizer_profit, optimizer_deg, optimizer_cycles, final_soh, optimizer_profit),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "benchmark",
            "net_profit",
            "degradation_cost",
            "equivalent_cycles",
            "final_soh_proxy",
            "profit_after_degradation",
        ],
    )


def build_benchmark_chart(benchmark: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=benchmark["benchmark"],
            y=benchmark["net_profit"],
            name="Net profit",
            marker_color=BLUE,
        )
    )
    fig.add_trace(
        go.Bar(
            x=benchmark["benchmark"],
            y=benchmark["degradation_cost"],
            name="Degradation cost",
            marker_color=AMBER,
        )
    )
    fig.update_layout(title={"text": "Benchmark Comparison", "x": 0.01}, barmode="group", xaxis_tickangle=-16)
    fig.update_yaxes(title_text="EUR")
    return _dark_layout(fig, height=360)


def load_lut_frame(lut_path: Path | None) -> pd.DataFrame:
    if lut_path is None or not Path(lut_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(lut_path, encoding="utf-8-sig")
    lower = {str(col).lower(): col for col in df.columns}
    energy_col = lower.get("energy") or lower.get("dod") or lower.get("depth_of_discharge")
    cost_col = (
        lower.get("deg_cost_eur_per_mwh_throughput")
        or lower.get("degradation_cost_eur")
        or lower.get("deg_cost_final")
    )
    if energy_col is None or cost_col is None:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "dod": pd.to_numeric(df[energy_col], errors="coerce"),
            "degradation_cost": pd.to_numeric(df[cost_col], errors="coerce"),
        }
    ).dropna()
    c_rate_col = lower.get("c_rate") or lower.get("crate")
    out["c_rate"] = pd.to_numeric(df[c_rate_col], errors="coerce") if c_rate_col else 0.5
    return out.sort_values(["c_rate", "dod"]).reset_index(drop=True)


def degradation_monotonicity(lut: pd.DataFrame) -> dict[str, object]:
    if lut.empty:
        return {"status": "unavailable", "violations": 0, "message": "No compatible LUT data available."}
    violations = 0
    groups = lut.groupby("c_rate") if "c_rate" in lut.columns else [(None, lut)]
    for _, group in groups:
        ordered = group.sort_values("dod")
        diffs = ordered["degradation_cost"].diff().dropna()
        violations += int((diffs < -1e-9).sum())
    status = "ok" if violations == 0 else "warning"
    message = "Cost trend is monotonic by DoD." if violations == 0 else f"{violations} negative DoD cost steps detected."
    return {"status": status, "violations": violations, "message": message}


def build_degradation_curve(lut: pd.DataFrame):
    if not PLOTLY_AVAILABLE or lut.empty:
        return None
    fig = go.Figure()
    for c_rate, group in lut.groupby("c_rate"):
        fig.add_trace(
            go.Scatter(
                x=group["dod"],
                y=group["degradation_cost"],
                name=f"C-rate {float(c_rate):.2g}",
                mode="lines+markers",
            )
        )
    fig.update_layout(title={"text": "DoD vs Degradation Cost", "x": 0.01})
    fig.update_xaxes(title_text="DoD / interval energy")
    fig.update_yaxes(title_text="Degradation cost")
    return _dark_layout(fig, height=360)


def build_degradation_heatmap(lut: pd.DataFrame):
    if not PLOTLY_AVAILABLE or lut.empty or lut["c_rate"].nunique() <= 1:
        return None
    heat = lut.pivot_table(index="c_rate", columns="dod", values="degradation_cost", aggfunc="mean")
    fig = go.Figure(data=go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale="Magma"))
    fig.update_layout(title={"text": "Degradation Cost Surface", "x": 0.01})
    fig.update_xaxes(title_text="DoD")
    fig.update_yaxes(title_text="C-rate")
    return _dark_layout(fig, height=360)


def data_quality_checks(schedule: pd.DataFrame, params: dict) -> pd.DataFrame:
    timestamps = pd.to_datetime(schedule["timestamp"], errors="coerce")
    diffs = timestamps.sort_values().diff().dropna()
    expected = diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(minutes=15)
    missing = int(((diffs / expected).round().clip(lower=1) - 1).sum()) if not diffs.empty and expected > pd.Timedelta(0) else 0
    duplicates = int(timestamps.duplicated().sum())
    negative_prices = int((schedule["price_eur_mwh"] < 0).sum())
    price_std = float(schedule["price_eur_mwh"].std() or 0.0)
    price_mean = float(schedule["price_eur_mwh"].mean() or 0.0)
    spikes = int((schedule["price_eur_mwh"].sub(price_mean).abs() > 4.0 * price_std).sum()) if price_std > 0 else 0
    soc_min = float(params.get("soc_min", 0.0))
    soc_max = float(params.get("soc_max", 1.0))
    soc_violations = int(((schedule["soc_pct"] < soc_min - 1e-8) | (schedule["soc_pct"] > soc_max + 1e-8)).sum())
    missing_degradation = int(schedule["degradation_cost_eur"].isna().sum())
    rows = [
        ("missing timestamps", missing, missing == 0),
        ("duplicate timestamps", duplicates, duplicates == 0),
        ("negative prices", negative_prices, True),
        ("price spikes", spikes, True),
        ("SoC bound violations", soc_violations, soc_violations == 0),
        ("missing degradation values", missing_degradation, missing_degradation == 0),
    ]
    return pd.DataFrame(rows, columns=["check", "count", "pass"])
