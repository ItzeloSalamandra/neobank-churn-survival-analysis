"""
Shared styling utilities: vivid light-theme palette, CSS injection,
APA-inspired figure captions, and a genuine 3D bar chart helper.
"""

import streamlit as st
import plotly.graph_objects as go

COLOR_BLUE = "#4A90D9"
COLOR_RED = "#F2637A"
COLOR_ORANGE = "#FDBA45"
COLOR_GREEN = "#17A398"
COLOR_YELLOW = "#FFD166"
COLOR_PURPLE = "#6C63FF"

COLOR_BG = "#FFFFFF"
COLOR_BG_CARD = "#F5F7FA"
COLOR_TEXT = "#2D3648"
COLOR_TEXT_MUTED = "#6B7280"

PLOTLY_TEMPLATE = "plotly_white"

CHURN_COLOR_MAP = {"Active": COLOR_GREEN, "Churned": COLOR_RED}
CATEGORY_COLORWAY = [COLOR_RED, COLOR_GREEN, COLOR_ORANGE, COLOR_BLUE, COLOR_YELLOW, COLOR_PURPLE]


def inject_custom_css():
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 1rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
        }}
        .stMetric {{
            background-color: {COLOR_BG_CARD};
            border-radius: 12px;
            padding: 10px 12px;
            border: 1px solid rgba(0,0,0,0.06);
        }}
        h1 {{
            font-weight: 700 !important;
            color: {COLOR_TEXT};
            margin-top: 0rem !important;
            margin-bottom: 0.3rem !important;
            font-size: 1.9rem !important;
        }}
        h2, h3 {{
            font-weight: 700 !important;
            color: {COLOR_TEXT};
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
        }}
        hr {{
            margin: 0.6rem 0 !important;
        }}
        .takeaway-box {{
            background: linear-gradient(135deg, {COLOR_BLUE}15, {COLOR_ORANGE}15);
            border-left: 5px solid {COLOR_ORANGE};
            border-radius: 8px;
            padding: 12px 20px;
            margin-top: 8px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def takeaway_box(title, text):
    st.markdown(
        f"""
        <div class="takeaway-box">
            <strong style="font-size:1.05em;">{title}</strong><br>
            <span style="font-size:0.92em;">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(icon, title, text, color=COLOR_BLUE):
    st.markdown(
        f"""
        <div style="background:{color}18; border-left:4px solid {color};
                    border-radius:8px; padding:10px 14px; margin-bottom:8px;">
            <span style="font-size:1.3em;">{icon}</span>
            <strong style="font-size:0.95em;"> {title}</strong><br>
            <span style="font-size:0.85em; color:{COLOR_TEXT_MUTED};">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scorecard(label, value, color=COLOR_BLUE, delta=None):
    delta_html = f"<div style='font-size:0.8em; opacity:0.85;'>{delta}</div>" if delta else ""
    st.markdown(
        f"""
        <div style="background:{color}; border-radius:12px; padding:14px 16px;
                    color:white; min-height:80px;">
            <div style="font-size:0.8em; opacity:0.9;">{label}</div>
            <div style="font-size:1.8em; font-weight:700; line-height:1.2;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(text):
    st.markdown(
        f"""
        <div style="margin-top:14px; margin-bottom:6px;
                    font-size:1rem; font-weight:700; color:{COLOR_TEXT};">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def figure_header(number, title):
    """APA-inspired figure heading: 'Figure N' label above a bold title (no italics)."""
    st.markdown(
        f"""
        <div style="margin-top:14px; margin-bottom:6px;">
            <div style="font-size:0.85rem; color:{COLOR_TEXT_MUTED};">
                Figure {number}
            </div>
            <div style="font-weight:700; font-size:1rem; color:{COLOR_TEXT};">
                {title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def figure_note(text):
    """APA-inspired figure note, placed below a chart/section."""
    st.markdown(
        f"""
        <div style="font-size:0.78rem; color:{COLOR_TEXT_MUTED}; margin-top:4px; line-height:1.35;">
            <i>Note.</i> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_fig(fig, height=340):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT, family="sans-serif"),
        colorway=CATEGORY_COLORWAY,
        height=height,
        margin=dict(t=20, b=30, l=40, r=20),
    )
    return fig


def _cuboid_mesh(x0, x1, y0, y1, z0, z1, color, name):
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    xs, ys, zs = zip(*vertices)
    i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]
    return go.Mesh3d(
        x=xs, y=ys, z=zs, i=i, j=j, k=k,
        color=color, opacity=1.0, flatshading=True, name=name, showlegend=True,
    )


def bar3d_chart(categories, values, colors=None, bar_width=0.5, title="", height=340):
    colors = colors or (CATEGORY_COLORWAY * (len(categories) // len(CATEGORY_COLORWAY) + 1))
    fig = go.Figure()
    for idx, (cat, val) in enumerate(zip(categories, values)):
        x0 = idx - bar_width / 2
        x1 = idx + bar_width / 2
        fig.add_trace(_cuboid_mesh(x0, x1, 0, bar_width, 0, val, colors[idx], cat))

    fig.update_layout(
        scene=dict(
            xaxis=dict(ticktext=categories, tickvals=list(range(len(categories))), title=""),
            yaxis=dict(showticklabels=False, title=""),
            zaxis=dict(title="Value"),
        ),
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(t=20, b=10, l=10, r=10),
        showlegend=False,
    )
    return fig
