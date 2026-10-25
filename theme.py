"""Visual theme: sun, sky, wind and foliage palette for Streamlit and Plotly."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PALETTE: dict[str, str] = {
    "gold": "#F5A623",
    "amber": "#D97706",
    "slate": "#1B2A3B",
    "slate_soft": "#3E5063",
    "sky": "#4FB3E8",
    "sky_light": "#E6F4FC",
    "sage": "#6FAE8A",
    "mint": "#DDF1E6",
    "sand": "#FFF6E5",
    "muted": "#6B7A8C",
    "border": "#E1E8EF",
    "page": "#F5F8FA",
}

COLORWAY: list[str] = [
    PALETTE["gold"], PALETTE["sky"], PALETTE["sage"],
    PALETTE["amber"], PALETTE["slate_soft"], "#9AD1EE",
]
SUN_SCALE: list[list[Any]] = [
    [0.0, PALETTE["sky"]], [0.5, PALETTE["gold"]], [1.0, "#B45309"],
]
CORR_SCALE: list[list[Any]] = [
    [0.0, PALETTE["sky"]], [0.5, "#FFFFFF"], [1.0, PALETTE["gold"]],
]
FONT_STACK = "Inter, 'Segoe UI', Roboto, sans-serif"
HEADING_STACK = "Poppins, Inter, 'Segoe UI', sans-serif"


def _register_plotly_template() -> None:
    axis = dict(
        gridcolor=PALETTE["border"],
        linecolor=PALETTE["border"],
        zeroline=False,
        ticks="outside",
        tickcolor=PALETTE["border"],
        title=dict(standoff=10),
    )
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_STACK, color=PALETTE["slate"], size=13),
        title=dict(
            font=dict(family=HEADING_STACK, size=16, color=PALETTE["slate"]),
            x=0.01, xanchor="left",
        ),
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=12, r=12, t=58, b=12),
        xaxis=axis,
        yaxis=axis,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="white", bordercolor=PALETTE["border"],
            font=dict(family=FONT_STACK, size=12, color=PALETTE["slate"]),
        ),
    )
    pio.templates["solar"] = template
    pio.templates.default = "solar"


_register_plotly_template()

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: __FONT__; color: __SLATE__; }
.stApp { background: __PAGE__; }
h1, h2, h3, h4 { font-family: __HEADING__; color: __SLATE__; letter-spacing: -0.01em; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] { background: linear-gradient(180deg, __SLATE__ 0%, #223750 100%); }
[data-testid="stSidebar"] * { color: #EAF2F8; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: 0.25rem; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 0.55rem 0.8rem; border-radius: 10px; width: 100%; transition: background 0.15s;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(255,255,255,0.08); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(245,166,35,0.20); box-shadow: inset 3px 0 0 __GOLD__;
}
.brand { display: flex; align-items: center; gap: 0.7rem; padding: 0.4rem 0 1rem 0; }
.brand .sun { width: 38px; height: 38px; border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, #FFD27A, __GOLD__ 60%, __AMBER__);
  box-shadow: 0 0 0 5px rgba(245,166,35,0.22); }
.brand .name { font-family: __HEADING__; font-weight: 600; font-size: 1.05rem; line-height: 1.15; }
.brand .tag { font-size: 0.78rem; opacity: 0.7; }
.chip { display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.74rem;
  font-weight: 500; margin: 0.15rem 0.25rem 0.15rem 0; }
.chip.ok { background: rgba(111,174,138,0.28); color: #CFF0DC !important; }
.chip.warn { background: rgba(245,166,35,0.25); color: #FFE0A8 !important; }

.hero { position: relative; overflow: hidden; padding: 1.7rem 2rem; border-radius: 20px; margin-bottom: 1.3rem;
  background: radial-gradient(circle at 90% 18%, rgba(245,166,35,0.60), transparent 36%),
              linear-gradient(120deg, __SLATE__ 0%, #285472 62%, #3C8DB5 100%);
  box-shadow: 0 10px 30px rgba(27,42,59,0.16); }
.hero::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='600' height='160' viewBox='0 0 600 160'%3E%3Cpath d='M0 110 C100 70 200 150 300 110 S500 70 600 110' fill='none' stroke='white' stroke-opacity='.13' stroke-width='2'/%3E%3Cpath d='M0 136 C100 96 200 176 300 136 S500 96 600 136' fill='none' stroke='white' stroke-opacity='.08' stroke-width='2'/%3E%3C/svg%3E") repeat-x bottom / 600px 160px; }
.hero h1 { color: #FFFFFF; margin: 0; font-size: 2rem; font-weight: 700; position: relative; }
.hero p { color: #DCEBF5; margin: 0.45rem 0 0 0; font-size: 1.02rem; max-width: 46rem; position: relative; }
.hero .badges { margin-top: 0.9rem; position: relative; }
.hero .badge { display: inline-block; padding: 0.22rem 0.75rem; margin-right: 0.45rem; border-radius: 999px;
  background: rgba(255,255,255,0.14); color: #FFFFFF; font-size: 0.8rem; border: 1px solid rgba(255,255,255,0.22); }

.section-title { font-family: __HEADING__; font-weight: 600; font-size: 1.15rem; margin: 1.4rem 0 0.15rem 0; }
.section-caption { color: __MUTED__; font-size: 0.9rem; margin-bottom: 0.7rem; }

.kpi { background: #FFFFFF; border: 1px solid __BORDER__; border-radius: 14px; padding: 0.95rem 1.1rem;
  border-left: 5px solid var(--accent); min-height: 104px; box-shadow: 0 1px 2px rgba(27,42,59,0.04); }
.kpi .label { color: __MUTED__; font-size: 0.82rem; font-weight: 500; }
.kpi .value { font-family: __HEADING__; font-size: 1.65rem; font-weight: 600; line-height: 1.25; margin-top: 0.1rem; }
.kpi .hint { color: __MUTED__; font-size: 0.78rem; margin-top: 0.1rem; }

.panel { background: #FFFFFF; border: 1px solid __BORDER__; border-radius: 14px; padding: 1rem 1.2rem; }
div[data-testid="stForm"] { background: #FFFFFF; border: 1px solid __BORDER__; border-radius: 14px; padding: 1.1rem 1.3rem; }

.stButton > button, div[data-testid="stFormSubmitButton"] > button {
  background: linear-gradient(135deg, __GOLD__, __AMBER__); color: #FFFFFF; border: 0; border-radius: 10px;
  font-weight: 600; padding: 0.55rem 1.3rem; }
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { filter: brightness(1.06); color: #FFFFFF; }
.stButton > button:focus-visible, div[data-testid="stFormSubmitButton"] > button:focus-visible {
  outline: 3px solid __SKY__; outline-offset: 2px; }
button[data-baseweb="tab"] { font-weight: 500; }
button[data-baseweb="tab"][aria-selected="true"] { color: __AMBER__; }
div[data-baseweb="tab-highlight"] { background-color: __GOLD__; }
div[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid __BORDER__; border-radius: 14px; padding: 0.3rem; }
@media (max-width: 760px) { .hero { padding: 1.2rem; } .hero h1 { font-size: 1.5rem; } }
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet."""
    css = _CSS
    replacements = {
        "__FONT__": FONT_STACK, "__HEADING__": HEADING_STACK,
        "__SLATE__": PALETTE["slate"], "__PAGE__": PALETTE["page"],
        "__GOLD__": PALETTE["gold"], "__AMBER__": PALETTE["amber"],
        "__SKY__": PALETTE["sky"], "__MUTED__": PALETTE["muted"],
        "__BORDER__": PALETTE["border"],
    }
    for token, value in replacements.items():
        css = css.replace(token, value)
    st.markdown(css, unsafe_allow_html=True)


def hero(title: str, subtitle: str, badges: Sequence[str] = ()) -> None:
    badge_html = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>'
        f'<div class="badges">{badge_html}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, caption: str = "") -> None:
    html = f'<div class="section-title">{title}</div>'
    if caption:
        html += f'<div class="section-caption">{caption}</div>'
    st.markdown(html, unsafe_allow_html=True)


def kpi_card(label: str, value: str, hint: str = "", accent: str = "gold") -> str:
    color = PALETTE.get(accent, accent)
    return (
        f'<div class="kpi" style="--accent:{color}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="hint">{hint}</div></div>'
    )


def kpi_row(cards: Sequence[tuple[str, str, str, str]]) -> None:
    """Render cards given as (label, value, hint, accent) tuples."""
    for column, (label, value, hint, accent) in zip(st.columns(len(cards)), cards):
        column.markdown(kpi_card(label, value, hint, accent), unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, template="solar")
    return fig


def plot(fig: go.Figure, height: int = 380) -> None:
    """Render a themed Plotly figure (works across Streamlit versions)."""
    style_figure(fig, height)
    try:
        st.plotly_chart(fig, width="stretch", theme=None)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, theme=None)


def table(data: pd.DataFrame, **kwargs: Any) -> None:
    """Render a dataframe full-width (works across Streamlit versions)."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)