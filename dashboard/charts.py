"""Chart builders for the Streamlit BESS dashboard."""

from __future__ import annotations

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except Exception:
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


def _operation_from_row(row) -> str:
    buy_mw = float(getattr(row, "p_buy_mw", 0.0) or 0.0)
    sell_mw = float(getattr(row, "p_sell_mw", 0.0) or 0.0)
    if sell_mw > 1e-7:
        return "sell"
    if buy_mw > 1e-7:
        return "buy"
    return str(getattr(row, "operation", "idle")).lower()


def _dark_layout(fig, height: int):
    fig.update_layout(
        height=height,
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font={"color": TEXT, "family": "Inter, Arial, sans-serif"},
        margin={"l": 52, "r": 26, "t": 54, "b": 36},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": MUTED},
        },
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def build_dispatch_chart(schedule: pd.DataFrame, params: dict, title: str = "Daily Dispatch"):
    if not PLOTLY_AVAILABLE:
        return build_dispatch_chart_matplotlib(schedule, params, title), "matplotlib"

    df = schedule.copy()
    df["soc_percent"] = df["soc_pct"] * 100.0
    dt_minutes = int(round(float(params.get("dt", 0.25)) * 60))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        row_heights=[0.43, 0.30, 0.27],
    )

    for row in df.itertuples(index=False):
        operation = _operation_from_row(row)
        if operation not in {"buy", "sell"}:
            continue
        color = "rgba(34, 197, 94, 0.24)" if operation == "buy" else "rgba(239, 68, 68, 0.24)"
        fig.add_shape(
            type="rect",
            x0=row.timestamp,
            x1=row.timestamp + pd.Timedelta(minutes=dt_minutes),
            y0=0,
            y1=1,
            xref="x",
            yref="y domain",
            fillcolor=color,
            line_width=0,
            layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["price_eur_mwh"],
            mode="lines",
            name="DAM price",
            line={"color": "#e2e8f0", "width": 2.4},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", name="Buy energy", marker={"color": GREEN, "size": 10}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", name="Sell energy", marker={"color": RED, "size": 10}),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=-df["p_buy_mw"],
            name="Charge MW",
            marker_color=GREEN,
            opacity=0.82,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["p_sell_mw"],
            name="Discharge MW",
            marker_color=RED,
            opacity=0.82,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_color="rgba(226, 232, 240, 0.38)", line_width=1, row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["soc_percent"],
            mode="lines",
            name="SoC",
            line={"color": BLUE, "width": 2.5},
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=float(params["soc_min"]) * 100,
        line_dash="dash",
        line_color="rgba(96, 165, 250, 0.55)",
        row=3,
        col=1,
    )
    fig.add_hline(
        y=float(params["soc_max"]) * 100,
        line_dash="dash",
        line_color="rgba(96, 165, 250, 0.55)",
        row=3,
        col=1,
    )

    fig.update_yaxes(title_text="EUR/MWh", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="SoC %", row=3, col=1, range=[0, 100])
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 19, "color": TEXT},
        },
        barmode="relative",
        hovermode="x unified",
        margin={"l": 52, "r": 26, "t": 118, "b": 36},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.025,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": MUTED},
        },
    )
    fig = _dark_layout(fig, height=760)
    fig.update_layout(
        margin={"l": 52, "r": 26, "t": 118, "b": 36},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.025,
            "xanchor": "center",
            "x": 0.5,
            "font": {"color": MUTED},
        },
    )
    return fig, "plotly"


def build_financial_chart(summary: dict):
    if not PLOTLY_AVAILABLE:
        return None, "matplotlib"

    scale_factor = float(summary.get("scale_factor", 1.0))
    suffix = "park" if scale_factor != 1.0 else "base module"
    revenue = float(summary.get("park_gross_revenue_eur", summary.get("gross_revenue_eur", 0.0)))
    purchase = float(summary.get("park_gross_purchase_eur", summary.get("gross_purchase_eur", 0.0)))
    degradation = float(summary.get("park_degradation_cost_eur", summary.get("degradation_cost_eur", 0.0)))
    net = float(summary.get("park_net_profit_eur", summary.get("net_profit_eur", 0.0)))

    fig = go.Figure(
        go.Waterfall(
            name=suffix,
            orientation="v",
            measure=["relative", "relative", "relative", "total"],
            x=["Revenue", "Purchase cost", "Degradation", "Net profit"],
            y=[revenue, -purchase, -degradation, net],
            connector={"line": {"color": "rgba(226,232,240,0.35)"}},
            increasing={"marker": {"color": GREEN}},
            decreasing={"marker": {"color": RED}},
            totals={"marker": {"color": BLUE}},
        )
    )
    fig.update_layout(
        title={
            "text": "Financial Breakdown",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 19, "color": TEXT},
        }
    )
    fig.update_yaxes(title_text="EUR")
    return _dark_layout(fig, height=410), "plotly"


def build_model_benchmark_chart(comparison: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        return None, "matplotlib"

    if comparison is None or comparison.empty:
        return None, "plotly"

    available_mask = (
        comparison["available"].astype(bool)
        if "available" in comparison.columns
        else pd.Series(True, index=comparison.index)
    )
    available = comparison[available_mask].copy()
    if available.empty:
        return None, "plotly"

    model_ids = available.get("model_id", pd.Series("", index=available.index)).astype(str)
    base_rows = available[model_ids == "degradation_aware_milp"]
    if base_rows.empty:
        base_profit = float(pd.to_numeric(available["Net profit EUR"], errors="coerce").dropna().iloc[0])
    else:
        base_profit = float(base_rows["Net profit EUR"].iloc[0])

    color_map = {
        "degradation_aware_milp": BLUE,
        "perfect": GREEN,
        "naive": AMBER,
    }
    short_label_map = {
        "degradation_aware_milp": "Selected MILP",
        "perfect": "Perfect MILP",
        "naive": "EMA heuristic",
    }
    bar_colors = model_ids.map(color_map).fillna(BLUE)
    chart_labels = model_ids.map(short_label_map).fillna(available["Model"])
    annotations = []
    for _, row in available.iterrows():
        delta = row.get("Delta vs degradation-aware MILP EUR")
        if row.get("model_id") == "degradation_aware_milp" or pd.isna(delta):
            annotations.append("")
        else:
            annotations.append(f"{float(delta):+,.0f} EUR")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=chart_labels,
            y=available["Net profit EUR"],
            text=annotations,
            textposition="outside",
            cliponaxis=False,
            name="Net profit",
            marker_color=bar_colors,
        )
    )
    fig.add_hline(
        y=base_profit,
        line_dash="dash",
        line_color="rgba(226, 232, 240, 0.45)",
        annotation_text="Selected baseline",
        annotation_position="top left",
    )
    fig.update_layout(
        title={
            "text": "Optimization Model Benchmark",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 19, "color": TEXT},
        },
        showlegend=False,
        xaxis_tickangle=0,
    )
    fig.update_yaxes(title_text="EUR")
    return _dark_layout(fig, height=430), "plotly"


def build_price_preview_chart(prices: pd.Series, title: str = "DAM Price Preview"):
    if not PLOTLY_AVAILABLE:
        return None, "matplotlib"

    preview = prices.dropna().copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preview.index,
            y=preview.values,
            mode="lines",
            name="DAM price",
            line={"color": "#e2e8f0", "width": 2.0},
            fill="tozeroy",
            fillcolor="rgba(96, 165, 250, 0.10)",
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.01},
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_yaxes(title_text="EUR/MWh")
    return _dark_layout(fig, height=280), "plotly"


def build_dispatch_chart_matplotlib(schedule: pd.DataFrame, params: dict, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = schedule.copy()
    dt_minutes = int(round(float(params.get("dt", 0.25)) * 60))
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.patch.set_facecolor(PAPER_BG)
    for ax in axes:
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.grid(color=GRID, alpha=0.55)

    for row in df.itertuples(index=False):
        if row.operation == "buy":
            axes[0].axvspan(row.timestamp, row.timestamp + pd.Timedelta(minutes=dt_minutes), color=GREEN, alpha=0.16)
        elif row.operation == "sell":
            axes[0].axvspan(row.timestamp, row.timestamp + pd.Timedelta(minutes=dt_minutes), color=RED, alpha=0.16)

    axes[0].plot(df["timestamp"], df["price_eur_mwh"], color="#e2e8f0", linewidth=2)
    axes[0].set_ylabel("EUR/MWh")
    axes[0].set_title(title, color=TEXT)
    axes[1].bar(df["timestamp"], df["p_sell_mw"], color=RED, width=0.009, label="Discharge")
    axes[1].bar(df["timestamp"], -df["p_buy_mw"], color=GREEN, width=0.009, label="Charge")
    axes[1].axhline(0, color=TEXT, linewidth=0.8, alpha=0.4)
    axes[1].set_ylabel("MW")
    axes[2].plot(df["timestamp"], df["soc_pct"] * 100, color=BLUE, linewidth=2)
    axes[2].axhline(float(params["soc_min"]) * 100, linestyle="--", color=BLUE, alpha=0.55)
    axes[2].axhline(float(params["soc_max"]) * 100, linestyle="--", color=BLUE, alpha=0.55)
    axes[2].set_ylabel("SoC %")
    fig.tight_layout()
    return fig
