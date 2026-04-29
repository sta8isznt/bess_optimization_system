"""Dark Streamlit styling for the BESS dashboard."""

from __future__ import annotations

import html

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 18% 0%, rgba(37, 99, 235, 0.20), transparent 30%),
                linear-gradient(135deg, #07111f 0%, #0a1020 42%, #101623 100%);
            color: #e5eefb;
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


def section_title(text: str) -> None:
    st.markdown(f'<div class="section-title">{html.escape(text)}</div>', unsafe_allow_html=True)


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
