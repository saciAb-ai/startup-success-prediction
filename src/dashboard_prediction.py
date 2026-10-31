"""Page « Analyse de profil » : formulaire, score du pipeline final et repères.

Aucun modèle n'est entraîné ici : la page utilise le pipeline final validé par
l'équipe (contrat décrit dans ``models.py`` et ``docs/app_integration.md``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import app_data
import models

UNKNOWN = "— Inconnu / non renseigné —"
OTHER = "Autre (saisie libre)…"
MONTHS = (
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre",
)
LOCATION_FIELDS = (
    # (colonne, libellé, colonnes qui filtrent la liste proposée)
    ("country_code", "Pays (code, ex. USA)", ()),
    ("state_code", "État / province", ("country_code",)),
    ("region", "Région", ("country_code", "state_code")),
    ("city", "Ville", ("country_code", "region")),
)
METADATA_FIELDS = (
    ("trained_at", "Entraîné le"),
    ("trained_by", "Préparé par (code étudiant)"),
    ("sklearn_version", "Version de scikit-learn"),
)
RESULT_KEY = "profile_last_analysis"
CURRENT_YEAR = pd.Timestamp.today().year
READING_GUIDE = """
- La classe compare ce profil aux startups **historiques** de Crunchbase dont l'issue est connue.
- Le score n'est **pas** une probabilité réelle de réussite future : les startups encore actives
  (operating) sont exclues et les données sont anciennes.
- Seules 7 caractéristiques initiales sont utilisées (localisation, secteur, date de création) :
  financement, équipe ou produit ne sont pas pris en compte.
- Les repères « classe 1 » sont descriptifs : ils ne prouvent aucune causalité.
"""


def _choice(container, label: str, options: list[str], key: str, help_text: str | None = None):
    """Valeurs observées + « Inconnu » + saisie libre.

    Retourne (valeur ou None, True si la valeur a été saisie librement).
    """
    selection = container.selectbox(label, [UNKNOWN, *options, OTHER], key=key, help=help_text)
    if selection == UNKNOWN:
        return None, False
    if selection == OTHER:
        typed = container.text_input(f"{label} : saisie libre", key=f"{key}_free").strip()
        return (typed or None), bool(typed)
    return selection, False


def _default_year(frame: pd.DataFrame) -> int:
    """Année proposée par défaut : médiane des startups dont l'issue est connue."""
    years = app_data.modeling_rows(frame)["founding_year"].dropna()
    year = int(years.median()) if len(years) else 2008
    return min(max(year, 1900), CURRENT_YEAR)


def _profile_form(frame: pd.DataFrame) -> tuple[dict, list[str]]:
    """Formulaire des 7 variables initiales ; listes filtrées selon les choix précédents."""
    profile: dict = {}
    free_fields: list[str] = []
    left, right = st.columns(2)

    left.markdown("**Localisation**")
    for column, label, depends_on in LOCATION_FIELDS:
        constraints = {name: profile.get(name) for name in depends_on}
        options = app_data.observed_options(frame, column, constraints)
        profile[column], free = _choice(left, label, options, key=f"profile_{column}")
        if free:
            free_fields.append(label)

    right.markdown("**Activité et date de création**")
    profile["primary_category"], free = _choice(
        right,
        "Secteur principal",
        app_data.observed_options(frame, "primary_category"),
        key="profile_primary_category",
        help_text="Première catégorie de category_list dans Crunchbase.",
    )
    if free:
        free_fields.append("Secteur principal")

    year = None
    if not right.checkbox("Année de création inconnue", key="profile_year_unknown"):
        year = int(
            right.number_input(
                "Année de création",
                min_value=1900,
                max_value=CURRENT_YEAR,
                value=_default_year(frame),
                step=1,
                key="profile_year",
            )
        )
    profile["founding_year"] = year
    month = right.selectbox(
        "Mois de création",
        [UNKNOWN, *MONTHS],
        key="profile_month",
        help="Peu informatif : Crunchbase indique souvent janvier quand seule l'année est connue.",
    )
    profile["founding_month"] = None if month == UNKNOWN else MONTHS.index(month) + 1
    return profile, free_fields


def _model_card(loaded: models.LoadedModel) -> None:
    """Informations sur le modèle chargé (métadonnées facultatives)."""
    with st.expander(f"Modèle utilisé : {loaded.name}"):
        st.markdown(f"Fichier : `{app_data.relative_to_project(loaded.path)}`")
        for key, text in METADATA_FIELDS:
            if loaded.metadata.get(key):
                st.markdown(f"{text} : {loaded.metadata[key]}")
        metrics = loaded.metadata.get("test_metrics")
        numeric = {}
        if isinstance(metrics, dict):
            numeric = {
                name: round(float(value), 3)
                for name, value in metrics.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        if numeric:
            st.dataframe(
                pd.DataFrame({"métrique (jeu de test)": list(numeric), "valeur": list(numeric.values())}),
                hide_index=True,
            )
        else:
            st.caption("Aucune métrique de test fournie dans le fichier de métadonnées.")
    for note in loaded.notes:
        st.warning(note)


def _profile_warnings(frame: pd.DataFrame, profile: dict, free_fields: list[str]) -> list[str]:
    """Signale les profils situés hors du domaine observé pendant l'entraînement."""
    messages = []
    last_year = app_data.max_founding_year(frame)
    year = profile.get("founding_year")
    if year is not None and last_year is not None and year > last_year:
        messages.append(
            f"Année de création postérieure à la dernière année observée ({last_year}) : "
            "le modèle extrapole hors de la période étudiée."
        )
    if free_fields:
        messages.append(
            "Valeur(s) saisie(s) librement (" + ", ".join(free_fields) + ") : si elles n'ont jamais "
            "été vues à l'entraînement, le modèle les traite comme inconnues."
        )
    unknown = sum(value is None for value in profile.values())
    if unknown >= 4:
        messages.append(f"{unknown} champs inconnus sur 7 : le résultat repose surtout sur des valeurs imputées.")
    return messages


def _show_result(frame: pd.DataFrame, loaded: models.LoadedModel, profile: dict, free_fields: list[str]) -> None:
    """Classe, score, repères historiques et avertissements pour un profil."""
    X = models.make_input_frame(profile)
    try:
        result = models.predict_profile(loaded, X)
    except Exception as exc:  # noqa: BLE001 - erreur affichée à l'utilisateur
        st.error(f"Analyse impossible pour ce profil : {type(exc).__name__} : {exc}")
        return

    is_class_1 = result.predicted_class == 1
    if is_class_1:
        st.success("Profil plus proche des startups acquises ou introduites en bourse (classe 1).")
    else:
        st.warning("Profil plus proche des startups fermées (classe 0).")

    base = app_data.base_rate(frame)
    col_score, col_base, col_similar = st.columns(3)
    if result.score_kind == "probability":
        delta = None
        if base is not None:
            delta = f"{(result.score - base) * 100:+.1f} pts vs ensemble".replace(".", ",")
        col_score.metric(
            "Score du modèle (classe 1)",
            app_data.fmt_pct(result.score),
            delta=delta,
            help="Estimation conditionnelle à l'échantillon historique (startups operating exclues) : "
            "ce n'est pas une probabilité réelle de réussite future.",
        )
    elif result.score_kind == "decision":
        col_score.metric(
            "Score de décision",
            f"{result.score:+.2f}".replace(".", ","),
            help="Score non probabiliste (ex. SVM) : positif = classe 1. Ce n'est pas un pourcentage.",
        )
    else:
        col_score.metric("Classe prédite", str(result.predicted_class))
    col_base.metric(
        "Classe 1 dans les données",
        app_data.fmt_pct(base),
        help="Part d'IPO/acquisitions parmi les startups dont l'issue est connue.",
    )

    scope = [
        text
        for text, value in (("même pays", profile["country_code"]), ("même secteur", profile["primary_category"]))
        if value
    ]
    if scope:
        count, rate = app_data.similar_profiles(frame, profile["country_code"], profile["primary_category"])
        col_similar.metric(
            f"Classe 1 ({' + '.join(scope)})",
            app_data.fmt_pct(rate) if count else "—",
            help=f"Repère descriptif calculé sur {count} startups historiques comparables.",
        )
        if 0 < count < 30:
            st.caption(f"Seulement {count} startups comparables : ce repère est peu fiable.")

    if result.score_kind == "probability":
        st.progress(min(max(float(result.score), 0.0), 1.0))
        if (result.score >= 0.5) != is_class_1:
            st.caption(
                "Pour certains modèles (ex. SVM calibré), score et classe peuvent diverger près du "
                "seuil : la classe affichée vient de predict()."
            )

    for message in _profile_warnings(frame, profile, free_fields):
        st.warning(message)
    with st.expander("Données transmises au modèle"):
        st.dataframe(X, hide_index=True)
    with st.expander("Comment lire ce résultat ?"):
        st.markdown(READING_GUIDE)


def _no_model(model_file: Path) -> None:
    st.warning(
        f"Aucun modèle final pour l'instant : `{app_data.relative_to_project(model_file)}` est "
        "introuvable. Le dashboard EDA reste utilisable."
    )
    st.markdown(
        "**Pour activer cette page**, l'équipe modèles dépose, après validation collective :\n"
        "- `models/final_pipeline.joblib` : le Pipeline scikit-learn final entraîné ;\n"
        "- `models/final_pipeline.json` (facultatif) : nom du modèle, version de scikit-learn, "
        "métriques de test.\n\n"
        "Format détaillé : `docs/app_integration.md`."
    )


def _load_error(error: str) -> None:
    st.error(f"Le modèle n'a pas pu être chargé : {error}")
    st.markdown(
        "Pistes : vérifier que `src/preprocessing.py` est présent sur la branche, que la version "
        "de scikit-learn est celle de l'entraînement, et que le fichier respecte le contrat "
        "décrit dans `docs/app_integration.md`."
    )


def render(frame: pd.DataFrame, loaded: models.LoadedModel | None, error: str | None, model_file: Path) -> None:
    """Affiche la page d'analyse de profil."""
    st.title("Analyse de profil")
    st.info(app_data.DISCLAIMER)
    if error:
        _load_error(error)
        return
    if loaded is None:
        _no_model(model_file)
        return

    _model_card(loaded)
    profile, free_fields = _profile_form(frame)
    if st.button("Analyser ce profil", type="primary", key="profile_submit"):
        st.session_state[RESULT_KEY] = (profile, free_fields)
    stored = st.session_state.get(RESULT_KEY)
    if stored is None:
        st.caption("Renseignez le profil puis cliquez sur « Analyser ce profil ».")
        return
    stored_profile, stored_free = stored
    st.subheader("Résultat")
    if stored_profile != profile:
        st.caption("Les champs ont changé depuis la dernière analyse : cliquez à nouveau pour mettre à jour.")
    _show_result(frame, loaded, stored_profile, stored_free)
