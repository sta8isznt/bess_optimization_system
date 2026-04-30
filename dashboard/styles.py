"""Dark Streamlit styling for the BESS dashboard."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import streamlit as st


def _asset_data_uri(filename: str) -> str | None:
    path = Path(__file__).resolve().parent / filename
    if not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    suffix = path.suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{encoded}"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Nunito+Sans:wght@700;800;900&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(37, 99, 235, 0.20), transparent 30%),
                linear-gradient(135deg, #07111f 0%, #0a1020 42%, #101623 100%);
            color: #e5eefb;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 2.75rem !important;
        }

        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stAppDeployButton"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        div[data-testid="stToolbar"] {
            background: transparent !important;
        }

        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #080f1c 0%, #0d1422 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        section[data-testid="stSidebar"] * {
            color: #dce8f8;
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.4rem;
            max-width: 1560px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
        }

        div[data-testid="stMetricLabel"] {
            color: #9fb0c8;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 1.46rem;
        }

        .headline-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.82rem 1rem;
            margin: 0.4rem 0 1rem 0;
        }

        .headline-kpi-card {
            min-height: 108px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.91), rgba(12, 19, 34, 0.90));
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 0.88rem 0.95rem 0.82rem 0.95rem;
            box-shadow: 0 16px 36px rgba(0, 0, 0, 0.22);
        }

        .headline-kpi-card.positive {
            border-color: rgba(74, 222, 128, 0.28);
        }

        .headline-kpi-card.negative {
            border-color: rgba(248, 113, 113, 0.28);
        }

        .headline-kpi-card.accent {
            border-color: rgba(96, 165, 250, 0.26);
        }

        .headline-kpi-label {
            color: #d7e0ee;
            font-size: 0.86rem;
            line-height: 1.2;
            font-weight: 700;
            margin-bottom: 0.52rem;
        }

        .headline-kpi-value {
            font-family: 'Nunito Sans', ui-rounded, 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            color: #f8fafc;
            font-size: clamp(1.72rem, 2.25vw, 2.35rem);
            line-height: 0.98;
            font-weight: 850;
            letter-spacing: 0;
            font-variant-numeric: tabular-nums;
        }

        .headline-kpi-value.positive {
            color: #86efac;
            text-shadow: 0 0 18px rgba(34, 197, 94, 0.12);
        }

        .headline-kpi-value.negative {
            color: #fca5a5;
            text-shadow: 0 0 18px rgba(239, 68, 68, 0.10);
        }

        .headline-kpi-value.accent {
            color: #93c5fd;
        }

        .headline-kpi-grid.primary-kpi-grid {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }

        .headline-kpi-subvalue {
            margin-top: 0.64rem;
            color: #9fb0c8;
            font-size: 0.83rem;
            line-height: 1.25;
            font-weight: 700;
        }

        .headline-kpi-subvalue strong {
            color: #e5eefb;
            font-weight: 850;
        }

        .hero {
            padding: 1.1rem 1.2rem 1.05rem 1.2rem;
            border: 1px solid rgba(96, 165, 250, 0.24);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.66));
            border-radius: 8px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.24);
            margin-bottom: 1.0rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2.15rem;
            color: #f8fafc;
            font-weight: 800;
        }

        .hero p {
            margin: 0.45rem 0 0 0;
            color: #aab9cf;
            font-size: 1rem;
        }

        .exec-header {
            display: grid;
            grid-template-columns: minmax(0, 1fr) 230px;
            gap: 1rem;
            align-items: start;
            margin-bottom: 0.95rem;
        }

        .exec-title {
            background: rgba(8, 17, 31, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 0.95rem 1rem;
        }

        .exec-title h1 {
            margin: 0;
            color: #f8fafc;
            font-size: 1.58rem;
            line-height: 1.15;
            font-weight: 800;
        }

        .exec-title p {
            margin: 0.38rem 0 0 0;
            color: #9fb0c8;
            font-size: 0.92rem;
        }

        .exec-facts {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
        }

        .exec-fact {
            background: rgba(15, 23, 42, 0.80);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            padding: 0.72rem 0.78rem;
        }

        .exec-fact-label {
            color: #8496b1;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            margin-bottom: 0.22rem;
        }

        .exec-fact-value {
            color: #f8fafc;
            font-size: 0.86rem;
            font-weight: 700;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }

        .team-logo-slot {
            height: 118px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 8px;
            background: rgba(8, 17, 31, 0.52);
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.42rem;
            overflow: hidden;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
        }

        .team-logo-image {
            width: 100%;
            height: 100%;
            object-fit: contain;
            border-radius: 7px;
            display: block;
        }

        .team-logo-fallback {
            color: #71829a;
            font-size: 0.72rem;
            font-weight: 750;
            letter-spacing: 0.055em;
            text-transform: uppercase;
        }

        .strategy-panel {
            background: rgba(8, 17, 31, 0.66);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 1.8rem 2rem;
            margin-bottom: 0.8rem;
        }

        .overview-flow {
            display: grid;
            grid-template-columns: repeat(3, minmax(230px, 1fr));
            grid-template-rows: auto 4.2rem auto;
            column-gap: 3.2rem;
            row-gap: 0.9rem;
            align-items: center;
        }

        .flow-node {
            position: relative;
            display: flex;
            justify-content: center;
            min-width: 0;
        }

        .flow-node-0 {
            grid-column: 1;
            grid-row: 3;
            transform: translateY(0.25rem);
        }

        .flow-node-1 {
            grid-column: 1;
            grid-row: 1;
            transform: translateY(-0.1rem);
        }

        .flow-node-2 {
            grid-column: 2;
            grid-row: 1;
            transform: translateY(0.2rem);
        }

        .flow-node-3 {
            grid-column: 2;
            grid-row: 3;
            transform: translateY(-0.15rem);
        }

        .flow-node-4 {
            grid-column: 3;
            grid-row: 3;
            transform: translateY(0.25rem);
        }

        .flow-node-5 {
            grid-column: 3;
            grid-row: 1;
            transform: translateY(-0.2rem);
        }

        .flow-node-1::after,
        .flow-node-3::after {
            content: "→";
            position: absolute;
            top: 50%;
            right: -1.72rem;
            transform: translate(50%, -50%);
            color: #6f819a;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1;
        }

        .flow-node-0::after {
            content: "↑";
            position: absolute;
            top: -2.65rem;
            left: 50%;
            transform: translateX(-50%);
            color: #6f819a;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1;
        }

        .flow-node-4::after {
            content: "↑";
            position: absolute;
            top: -2.65rem;
            left: 50%;
            transform: translateX(-50%);
            color: #6f819a;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1;
        }

        .flow-node-2::after {
            content: "↓";
            position: absolute;
            bottom: -1.95rem;
            left: 50%;
            transform: translateX(-50%);
            color: #6f819a;
            font-size: 1.55rem;
            font-weight: 900;
            line-height: 1;
        }

        .flow-card {
            width: min(100%, 258px);
            min-height: 158px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            background:
                radial-gradient(circle at 50% 0%, rgba(96, 165, 250, 0.13), transparent 58%),
                rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 999px;
            padding: 1.2rem 1.3rem;
            box-shadow:
                inset 0 0 0 1px rgba(255, 255, 255, 0.018),
                0 16px 34px rgba(0, 0, 0, 0.20);
        }

        .flow-card.runtime {
            border-color: rgba(96, 165, 250, 0.52);
        }

        .flow-card.offline {
            border-color: rgba(245, 158, 11, 0.58);
            background:
                radial-gradient(circle at 50% 0%, rgba(245, 158, 11, 0.13), transparent 58%),
                rgba(15, 23, 42, 0.88);
        }

        .flow-card.output {
            border-color: rgba(34, 197, 94, 0.54);
            background:
                radial-gradient(circle at 50% 0%, rgba(34, 197, 94, 0.13), transparent 58%),
                rgba(15, 23, 42, 0.88);
        }

        .flow-arrow {
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 1.15rem;
            color: #64748b;
            font-size: 1.15rem;
            font-weight: 800;
        }

        .flow-kind {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.08rem 0.46rem;
            border: 1px solid rgba(148, 163, 184, 0.23);
            background: rgba(8, 17, 31, 0.76);
            color: #9fb0c8;
            font-size: 0.72rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.035em;
            margin-bottom: 0.5rem;
        }

        .flow-title {
            color: #f8fafc;
            font-size: 1.06rem;
            line-height: 1.2;
            font-weight: 800;
            margin-bottom: 0;
        }

        .flow-desc {
            color: #aab9cf;
            font-size: 0.78rem;
            line-height: 1.35;
            margin-bottom: 0.54rem;
        }

        .flow-path {
            color: #6f819a;
            font-size: 0.68rem;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }

        .config-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
            gap: 0.58rem;
            margin-top: 0.3rem;
        }

        .config-item {
            background: rgba(8, 17, 31, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            padding: 0.68rem 0.72rem;
        }

        .config-label {
            color: #8496b1;
            font-size: 0.72rem;
            margin-bottom: 0.20rem;
        }

        .config-value {
            color: #f8fafc;
            font-weight: 750;
            font-size: 0.84rem;
            overflow-wrap: anywhere;
        }

        .compact-kpi-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.55rem;
            margin: 0.35rem 0 0.85rem 0;
        }

        .compact-kpi {
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            padding: 0.66rem 0.72rem;
        }

        .compact-kpi-label {
            color: #8496b1;
            font-size: 0.71rem;
            margin-bottom: 0.18rem;
        }

        .compact-kpi-value {
            color: #f8fafc;
            font-weight: 800;
            font-size: 0.98rem;
        }

        .compact-kpi-value.positive {
            color: #bbf7d0;
        }

        .compact-kpi-value.negative {
            color: #fecaca;
        }

        @media (max-width: 1180px) {
            .exec-header {
                grid-template-columns: minmax(0, 1fr) 190px;
            }

            .headline-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .headline-kpi-grid.primary-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .team-logo-slot {
                height: 102px;
            }

            .overview-flow {
                column-gap: 1rem;
            }

            .flow-card {
                width: min(100%, 178px);
                min-height: 118px;
                padding: 0.9rem 0.95rem;
            }
        }

        @media (max-width: 720px) {
            .headline-kpi-grid {
                grid-template-columns: 1fr;
            }

            .headline-kpi-grid.primary-kpi-grid {
                grid-template-columns: 1fr;
            }

            .headline-kpi-card {
                min-height: 98px;
            }

            .strategy-panel {
                padding: 0.9rem;
            }

            .overview-flow {
                grid-template-columns: 1fr;
                grid-template-rows: none;
                row-gap: 1.45rem;
            }

            .flow-node,
            .flow-node-0,
            .flow-node-1,
            .flow-node-2,
            .flow-node-3,
            .flow-node-4,
            .flow-node-5 {
                grid-column: 1;
                grid-row: auto;
                transform: none;
            }

            .flow-node-0::after,
            .flow-node-1::after,
            .flow-node-2::after,
            .flow-node-3::after,
            .flow-node-4::after {
                content: "↓";
                top: auto;
                right: auto;
                bottom: -1.05rem;
                left: 50%;
                transform: translateX(-50%);
                font-size: 1.1rem;
            }

            .flow-card {
                width: min(100%, 260px);
                min-height: 112px;
            }
        }

        .section-title {
            margin: 1.3rem 0 0.55rem 0;
            color: #f8fafc;
            font-size: 1.05rem;
            font-weight: 750;
        }

        .soft-panel {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            padding: 1rem 1rem;
        }

        .constraint-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 0.72rem;
            margin-top: 0.3rem;
        }

        .constraint-item {
            background: rgba(8, 17, 31, 0.82);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            padding: 0.78rem 0.82rem;
        }

        .constraint-label {
            color: #8ea3be;
            font-size: 0.78rem;
            margin-bottom: 0.22rem;
        }

        .constraint-value {
            color: #f8fafc;
            font-weight: 700;
            font-size: 0.94rem;
        }

        .ok-badge {
            display: inline-block;
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.13);
            border: 1px solid rgba(34, 197, 94, 0.32);
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 0.12rem 0.5rem;
            margin-bottom: 0.35rem;
        }

        .story-panel {
            background: rgba(8, 17, 31, 0.70);
            border-left: 3px solid #60a5fa;
            border-radius: 8px;
            padding: 1rem 1.1rem;
        }

        .story-panel li {
            margin-bottom: 0.3rem;
            color: #cbd7e8;
        }

        .stButton > button {
            background: linear-gradient(135deg, #2563eb, #0ea5e9);
            color: #ffffff;
            border: 0;
            border-radius: 8px;
            font-weight: 800;
            padding: 0.62rem 1rem;
            width: 100%;
        }

        .stDownloadButton > button {
            border-radius: 8px;
            border-color: rgba(96, 165, 250, 0.46);
            background: rgba(15, 23, 42, 0.82);
            color: #e5eefb;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Battery Optimization in the Greek Electricity Market</h1>
          <p>Day-ahead BESS dispatch under data scarcity, physical degradation costs, and operational constraints.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def executive_header(
    status: str,
    horizon: str,
    degradation_source: str,
    price_source: str,
) -> None:
    logo_uri = _asset_data_uri("team_logo_header.png")
    logo_markup = (
        f'<img class="team-logo-image" src="{html.escape(logo_uri)}" alt="Cache Me If You Can team logo">'
        if logo_uri
        else '<div class="team-logo-fallback">Team logo</div>'
    )
    st.markdown(
        f"""
        <div class="exec-header">
          <div class="exec-title">
            <h1>Battery Optimization in the Greek Electricity Market</h1>
            <p>Day-ahead BESS dispatch under data scarcity, physical degradation costs, and operational constraints.</p>
          </div>
          <div class="team-logo-slot">{logo_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def system_overview(stages: list[dict[str, str]]) -> None:
    cards = []
    for index, stage in enumerate(stages):
        card_class = " ".join(["flow-card", stage.get("tone", "runtime")])
        cards.append(
            f'<div class="flow-node flow-node-{index}">'
            f'<div class="{html.escape(card_class)}">'
            f'<div class="flow-kind">{html.escape(stage.get("kind", ""))}</div>'
            f'<div class="flow-title">{html.escape(stage.get("title", ""))}</div>'
            "</div>"
            "</div>"
        )

    st.markdown(
        '<div class="strategy-panel">'
        '<div class="overview-flow">'
        + "".join(cards)
        + "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def config_grid(items: list[tuple[str, str]]) -> None:
    body = "".join(
        '<div class="config-item">'
        f'<div class="config-label">{html.escape(label)}</div>'
        f'<div class="config-value">{html.escape(value)}</div>'
        "</div>"
        for label, value in items
    )
    st.markdown(f'<div class="config-grid">{body}</div>', unsafe_allow_html=True)


def headline_kpi_grid(items: list[tuple[str, str, str]]) -> None:
    allowed_tones = {"positive", "negative", "accent", "neutral"}
    cards = []
    for label, value, tone in items:
        tone_class = tone if tone in allowed_tones else "neutral"
        cards.append(
            f'<div class="headline-kpi-card {html.escape(tone_class)}">'
            f'<div class="headline-kpi-label">{html.escape(label)}</div>'
            f'<div class="headline-kpi-value {html.escape(tone_class)}">{html.escape(value)}</div>'
            "</div>"
        )
    st.markdown(f'<div class="headline-kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def primary_kpi_grid(items: list[tuple[str, str, str, str]]) -> None:
    allowed_tones = {"positive", "negative", "accent", "neutral"}
    cards = []
    for label, value, secondary, tone in items:
        tone_class = tone if tone in allowed_tones else "neutral"
        cards.append(
            f'<div class="headline-kpi-card {html.escape(tone_class)}">'
            f'<div class="headline-kpi-label">{html.escape(label)}</div>'
            f'<div class="headline-kpi-value {html.escape(tone_class)}">{html.escape(value)}</div>'
            f'<div class="headline-kpi-subvalue">{html.escape(secondary)}</div>'
            "</div>"
        )
    st.markdown(
        f'<div class="headline-kpi-grid primary-kpi-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def compact_kpi_strip(items: list[tuple[str, str, str]]) -> None:
    body = "".join(
        '<div class="compact-kpi">'
        f'<div class="compact-kpi-label">{html.escape(label)}</div>'
        f'<div class="compact-kpi-value {html.escape(tone)}">{html.escape(value)}</div>'
        "</div>"
        for label, value, tone in items
    )
    st.markdown(f'<div class="compact-kpi-strip">{body}</div>', unsafe_allow_html=True)


def constraint_item(label: str, value: str, ok: bool = True) -> str:
    badge = '<span class="ok-badge">OK</span>' if ok else '<span class="ok-badge">CHECK</span>'
    return (
        '<div class="constraint-item">'
        f"{badge}"
        f'<div class="constraint-label">{html.escape(label)}</div>'
        f'<div class="constraint-value">{html.escape(value)}</div>'
        "</div>"
    )


def constraint_grid(items: list[tuple[str, str, bool]]) -> None:
    body = "".join(constraint_item(label, value, ok) for label, value, ok in items)
    st.markdown(f'<div class="constraint-grid">{body}</div>', unsafe_allow_html=True)
