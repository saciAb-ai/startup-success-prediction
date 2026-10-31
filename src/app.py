"""Application Streamlit du projet (point d'entrée).

Lancement depuis la racine du dépôt :

    streamlit run src/app.py

Pages : Dashboard EDA (les 6 éléments exigés), Analyse de profil (pipeline
final validé par l'équipe), À propos. L'application n'entraîne aucun modèle.
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

PAGE_EDA = "Dashboard EDA"
PAGE_PREDICTION = "Analyse de profil"
PAGE_ABOUT = "À propos"

st.set_page_config(page_title="Startups : succès ou échec ?", page_icon="📊", layout="wide")


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
    st.error("Aucun fichier CSV trouvé dans data/raw/, data/ ou à la racine du projet.")
    st.markdown(
        "1. Télécharger le dataset Kaggle « Startup Success/Fail Dataset from Crunchbase ».\n"
        "2. Placer le fichier CSV directement dans `data/raw/` (dossier ignoré par Git).\n"
        "3. Relancer : `streamlit run src/app.py`"
    )


def about_page(frame: pd.DataFrame) -> None:
    st.title("À propos du projet")
    st.markdown(
        "**Question étudiée.** Parmi les startups historiques de Crunchbase ayant connu une "
        "acquisition, une introduction en bourse ou une fermeture, peut-on distinguer leur "
        "issue à partir de caractéristiques disponibles au début de leur activité ?"
    )
    st.markdown(
        "| Statut Crunchbase | Cible | Lecture |\n"
        "|---|:---:|---|\n"
        "| `ipo` | 1 | succès observé (introduction en bourse) |\n"
        "| `acquired` | 1 | succès observé (acquisition) |\n"
        "| `closed` | 0 | échec observé (fermeture) |\n"
        "| `operating` | exclu | issue non déterminée |"
    )
    known = app_data.modeling_rows(frame)
    st.markdown(
        f"**Fichier chargé :** {app_data.fmt_int(len(frame))} startups, dont "
        f"{app_data.fmt_int(len(known))} avec une issue connue "
        f"({app_data.fmt_pct(app_data.base_rate(frame))} en classe 1)."
    )
    st.markdown(
        "**Entrées du modèle :** pays, État ou province, région, ville, secteur principal, "
        "année et mois de création. Le financement est exclu : il est postérieur à la création."
    )
    st.markdown(
        "**Limites :** données historiques ; startups encore actives exclues (biais de "
        "sélection) ; aucune garantie de prédiction à horizon fixe ; corrélation ne veut pas "
        "dire causalité."
    )
    st.info(app_data.DISCLAIMER)
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
    st.sidebar.title("Startups : succès ou échec ?")
    page = st.sidebar.radio("Navigation", (PAGE_EDA, PAGE_PREDICTION, PAGE_ABOUT), key="nav_page")
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
    st.sidebar.caption("Outil exploratoire : pas une prédiction certaine de l'avenir d'une startup.")


main()
