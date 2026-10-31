"""Application Streamlit du projet (point d'entrée).

Lancement depuis la racine du dépôt :

    streamlit run src/app.py

Pages : Dashboard EDA (6 éléments exigés et une vue géographie), Analyse de
profil (pipeline final validé par l'équipe), À propos. Aucun modèle n'est
entraîné dans l'application.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import app_data  # noqa: E402
import dashboard_eda  # noqa: E402
import dashboard_prediction  # noqa: E402
import data_loader  # noqa: E402
import models  # noqa: E402
import ui  # noqa: E402

PAGE_EDA = "Dashboard EDA"
PAGE_PREDICTION = "Analyse de profil"
PAGE_ABOUT = "À propos"
PAGES = (PAGE_EDA, PAGE_PREDICTION, PAGE_ABOUT)
PAGE_ICONS = {
    PAGE_EDA: ":material/monitoring:",
    PAGE_PREDICTION: ":material/psychology:",
    PAGE_ABOUT: ":material/info:",
}
PAGE_CAPTIONS = [
    "Explorer les startups du dataset",
    "Tester un profil avec le modèle final",
    "Le projet, la cible et les limites",
]

st.set_page_config(
    page_title="Exit ou fermeture ? Startups Crunchbase",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Chargement et préparation du dataset…")
def get_frame(path: str, modified: float) -> pd.DataFrame:
    """CSV brut -> DataFrame du dashboard. ``modified`` sert uniquement de clé de cache."""
    return app_data.build_eda_frame(data_loader.load_raw(path))


@st.cache_resource(show_spinner="Chargement du modèle final…")
def get_model(path: str, modified: float) -> models.LoadedModel:
    """Charge et valide le pipeline final ; rechargé si le fichier change."""
    return models.load_model(path)


def dataset_missing_page() -> None:
    st.title("Dataset introuvable")
    st.error("Aucun fichier CSV trouvé dans data/raw/, data/ ou à la racine du projet.", icon=":material/folder_off:")
    with st.container(border=True):
        st.markdown(
            "1. Télécharger le dataset Kaggle « Startup Success/Fail Dataset from Crunchbase ».\n"
            "2. Placer le fichier CSV directement dans `data/raw/` (dossier ignoré par Git).\n"
            "3. Relancer : `streamlit run src/app.py`"
        )


def about_page(frame: pd.DataFrame) -> None:
    st.title("À propos du projet")
    st.markdown(
        "Parmi les startups historiques de Crunchbase ayant connu une acquisition, une introduction "
        "en bourse ou une fermeture, peut-on distinguer leur issue à partir de caractéristiques "
        "disponibles au début de leur activité ?"
    )
    known = app_data.modeling_rows(frame)
    facts = (
        (":material/database: Startups dans le fichier", app_data.fmt_int(len(frame))),
        (":material/flag: Issue connue", app_data.fmt_int(len(known))),
        (":material/trending_up: Taux de succès", app_data.fmt_pct(app_data.base_rate(frame))),
    )
    for column, (label, value) in zip(st.columns(3), facts):
        with column.container(border=True):
            st.metric(label, value)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### :material/target: La cible")
        st.markdown(
            "| Statut Crunchbase | Cible | Lecture |\n"
            "|---|:---:|---|\n"
            "| `ipo` | 1 | sortie réussie (introduction en bourse) |\n"
            "| `acquired` | 1 | sortie réussie (acquisition) |\n"
            "| `closed` | 0 | fermeture |\n"
            "| `operating` | exclu | issue non déterminée |"
        )
    with right:
        st.markdown("##### :material/input: Les entrées du modèle")
        st.markdown(
            "Pays, État ou province, région, ville, secteur principal, année et mois de création. "
            "Le financement est exclu : il n'est connu qu'après la création."
        )
        st.markdown("##### :material/warning: Les limites")
        st.markdown(
            "- Données historiques : le marché actuel peut être différent.\n"
            "- Startups encore actives exclues : biais de sélection.\n"
            "- Aucune garantie de prédiction à horizon fixe.\n"
            "- Un lien statistique n'est pas une cause."
        )
    st.info(app_data.DISCLAIMER, icon=":material/info:")
    st.caption("Source : Kaggle, « Startup Success/Fail Dataset from Crunchbase ».")


def prediction_page(frame: pd.DataFrame) -> None:
    model_file = models.model_path()
    loaded, error = None, None
    if model_file.is_file():
        try:
            loaded = get_model(str(model_file), model_file.stat().st_mtime)
        except Exception as exc:  # noqa: BLE001 - toute erreur de chargement est affichée
            error = f"{type(exc).__name__} : {exc}"
    dashboard_prediction.render(frame, loaded, error, model_file)


def main() -> None:
    ui.apply_global_style()
    ui.sidebar_brand()
    page = st.sidebar.radio(
        "Navigation",
        PAGES,
        key="nav_page",
        captions=PAGE_CAPTIONS,
        format_func=lambda name: f"{PAGE_ICONS[name]} {name}",
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    dataset = app_data.find_dataset()
    if dataset is None:
        dataset_missing_page()
        return
    try:
        frame = get_frame(str(dataset), dataset.stat().st_mtime)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Lecture du dataset impossible ({app_data.relative_to_project(dataset)}) : {exc}")
        return

    if page == PAGE_EDA:
        dashboard_eda.render(frame, source=app_data.relative_to_project(dataset))
    elif page == PAGE_PREDICTION:
        prediction_page(frame)
    else:
        about_page(frame)
    st.sidebar.divider()
    st.sidebar.caption(
        "Outil exploratoire : il compare un profil à des startups passées, sans prédire l'avenir avec certitude."
    )


main()
