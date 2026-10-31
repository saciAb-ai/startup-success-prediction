"""Identité visuelle de l'application : palette, graphiques et blocs d'interface.

Le thème Streamlit (.streamlit/config.toml) habille les widgets ; ce module
habille les graphiques Plotly et les quelques blocs HTML maison, avec la même
palette. Les couleurs portent un sens : bleu = sortie réussie (IPO ou
acquisition), vermillon = fermeture, violet = ce sur quoi on agit.
"""

from __future__ import annotations

import html

import streamlit as st

import app_data

INK = "#151B2E"
MUTED = "#5B6478"
GRID = "#E3E8F0"
ACCENT = "#5B3CC4"
SUCCESS = app_data.ISSUE_COLORS[app_data.LABEL_SUCCESS]
FAILURE = app_data.ISSUE_COLORS[app_data.LABEL_FAILURE]
BODY_FONT = "IBM Plex Sans, Source Sans Pro, sans-serif"
HEADING_FONT = "Sora, IBM Plex Sans, sans-serif"
DIVERGING_SCALE = [[0.0, FAILURE], [0.5, "#F4F4F6"], [1.0, SUCCESS]]
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

_CSS = """
<style>
.ux-split {display: flex; height: 54px; border-radius: 12px; overflow: hidden; margin: 0.5rem 0 0.35rem 0;}
.ux-split > div {display: flex; flex-direction: column; justify-content: center; min-width: 110px; padding: 0 16px; color: #FFFFFF;}
.ux-split span {font-size: 0.78rem; opacity: 0.92;}
.ux-split b {font-family: Sora, sans-serif; font-size: 1.15rem; font-weight: 700; line-height: 1.25;}
.ux-split-caption {display: flex; justify-content: space-between; gap: 1rem; color: #5B6478; font-size: 0.85rem; margin-bottom: 1.1rem;}
.ux-note {border-left: 3px solid #5B3CC4; border-radius: 0 10px 10px 0; background: #EFECFB; color: #2A2160; padding: 0.55rem 0.9rem; margin: 0.25rem 0 0.9rem 0; font-size: 0.92rem; line-height: 1.5;}
.ux-brand {font-family: Sora, sans-serif; font-weight: 700; font-size: 1.25rem; letter-spacing: -0.01em; margin-top: 0.2rem;}
.ux-brand-sub {font-size: 0.85rem; opacity: 0.75; margin-bottom: 0.75rem;}
</style>
"""


def apply_global_style() -> None:
    """Styles des blocs HTML maison (les widgets suivent le thème Streamlit)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    """Nom de l'application en tête de la barre latérale."""
    st.sidebar.markdown(
        '<div class="ux-brand">Exit ou fermeture ?</div>'
        '<div class="ux-brand-sub">Profils de startups Crunchbase</div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Encadré « comment lire » placé au-dessus d'un graphique."""
    st.markdown(f'<div class="ux-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def outcome_split(rate: float | None, known: int) -> None:
    """Barre signature : part des sorties réussies et des fermetures dans la sélection."""
    if rate is None or known == 0:
        note("Aucune startup avec une issue connue dans cette sélection.")
        return
    success = 100 * float(rate)
    failure = 100 - success
    label = f"{app_data.fmt_pct(rate)} de sorties réussies et {app_data.fmt_pct(1 - rate)} de fermetures"
    st.markdown(
        f'<div class="ux-split" role="img" aria-label="{html.escape(label)}">'
        f'<div style="flex: {success:.3f}; background: {SUCCESS};"><span>Sorties réussies</span>'
        f"<b>{app_data.fmt_pct(rate)}</b></div>"
        f'<div style="flex: {failure:.3f}; background: {FAILURE};"><span>Fermetures</span>'
        f"<b>{app_data.fmt_pct(1 - rate)}</b></div></div>"
        '<div class="ux-split-caption"><span>IPO ou acquisition</span>'
        f"<span>{app_data.fmt_int(known)} startups dont l’issue est connue</span>"
        "<span>Fermeture</span></div>",
        unsafe_allow_html=True,
    )


def style_figure(fig, height: int | None = None, legend: bool = True):
    """Applique l'identité visuelle à une figure Plotly (chiffres au format français)."""
    fig.update_layout(
        template="plotly_white",
        separators=", ",
        font={"family": BODY_FONT, "color": INK, "size": 13},
        title={"font": {"family": HEADING_FONT, "size": 16, "color": INK}, "x": 0, "xanchor": "left"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 8, "r": 8, "t": 64, "b": 8},
        hoverlabel={"bgcolor": "#FFFFFF", "bordercolor": INK, "font": {"family": BODY_FONT, "color": INK}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "title": {"text": ""}},
        showlegend=legend,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    if height is not None:
        fig.update_layout(height=height)
    return fig


def chart(fig, key: str, height: int | None = None, legend: bool = True) -> None:
    """Affiche une figure Plotly stylée et interactive (survol, zoom, export PNG)."""
    st.plotly_chart(style_figure(fig, height, legend), key=key, theme=None, config=PLOTLY_CONFIG)
