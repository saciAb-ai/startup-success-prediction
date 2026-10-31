"""Dashboard EDA de l'application Streamlit : les 6 éléments exigés.

1. Statistiques descriptives          4. Fréquences catégorielles
2. Valeurs manquantes                 5. Corrélations
3. Histogrammes et densités (KDE)     6. Filtres interactifs (barre latérale)

Les calculs viennent de ``app_data`` (testé sans Streamlit) : ce module ne
s'occupe que de l'affichage.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import app_data
import features

NUMERIC_VARIABLES = (
    "founding_year",
    "founding_month",
    "funding_rounds",
    "funding_total_usd",
    app_data.FUNDING_LOG_COLUMN,
)
CATEGORICAL_VARIABLES = (*features.CATEGORICAL_FEATURES, "status")
NOT_MODEL_INPUTS = {"funding_rounds", "funding_total_usd", app_data.FUNDING_LOG_COLUMN}
CORRELATION_COLUMNS = ("founding_year", "founding_month", "funding_rounds", app_data.FUNDING_LOG_COLUMN, "target")
TAB_NAMES = (
    "1. Statistiques descriptives",
    "2. Valeurs manquantes",
    "3. Distributions",
    "4. Fréquences catégorielles",
    "5. Corrélations",
)


def _sidebar_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Élément 6 : filtres interactifs, appliqués à tous les onglets."""
    st.sidebar.header("Filtres du dashboard")
    present = set(frame["issue"])
    issues = st.sidebar.multiselect(
        "Issue",
        [issue for issue in app_data.ISSUE_ORDER if issue in present],
        default=[issue for issue in app_data.MODELING_ISSUES if issue in present],
        key="eda_issues",
        help="Par défaut : le périmètre de la modélisation (IPO/acquisition et fermeture).",
    )
    countries = st.sidebar.multiselect(
        "Pays", app_data.top_values(frame, "country_code"), key="eda_countries", help="Vide = tous les pays."
    )
    categories = st.sidebar.multiselect(
        "Secteur principal",
        app_data.top_values(frame, "primary_category"),
        key="eda_categories",
        help="Vide = tous les secteurs.",
    )
    year_range, include_unknown = None, True
    years = frame["founding_year"].dropna()
    if years.nunique() > 1:
        low, high = int(years.min()), int(years.max())
        year_range = st.sidebar.slider("Année de création", low, high, (low, high), key="eda_years")
        include_unknown = st.sidebar.checkbox("Inclure les années inconnues", value=True, key="eda_unknown_years")

    filtered = app_data.apply_filters(
        frame,
        issues=issues,
        countries=countries,
        categories=categories,
        year_range=year_range,
        include_unknown_year=include_unknown,
    )
    st.sidebar.caption(
        f"{app_data.fmt_int(len(filtered))} startups sur {app_data.fmt_int(len(frame))} après filtrage."
    )
    return filtered


def _descriptive_statistics(data: pd.DataFrame) -> None:
    """Élément 1 : indicateurs clés, répartition des issues, résumés numériques et catégoriels."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Startups affichées", app_data.fmt_int(len(data)))
    col2.metric("Issue connue (cible 0/1)", app_data.fmt_int(len(app_data.modeling_rows(data))))
    col3.metric(
        "Part de la classe 1",
        app_data.fmt_pct(app_data.base_rate(data)),
        help="IPO ou acquisition, parmi les startups dont l'issue est connue.",
    )
    col4.metric("Pays différents", app_data.fmt_int(data["country_code"].nunique()))

    counts = data["issue"].value_counts()
    issues = [issue for issue in app_data.ISSUE_ORDER if issue in counts.index]
    figure = px.bar(
        pd.DataFrame({"issue": issues, "startups": [int(counts[issue]) for issue in issues]}),
        x="issue",
        y="startups",
        color="issue",
        text="startups",
        color_discrete_map=app_data.ISSUE_COLORS,
        title="Répartition des issues",
    )
    figure.update_layout(showlegend=False, xaxis_title="", yaxis_title="Nombre de startups")
    st.plotly_chart(figure, key="eda_issue_chart")

    st.markdown("**Variables numériques**")
    summary = app_data.numeric_summary(data, NUMERIC_VARIABLES)
    if not summary.empty:
        summary.index = [app_data.label(column) for column in summary.index]
        st.dataframe(summary)
    st.caption(
        "Le financement (montant, tours) est décrit ici mais exclu des entrées du modèle : "
        "il est postérieur à la création (risque de fuite temporelle)."
    )
    share = app_data.january_share(data)
    if share is not None:
        st.caption(
            f"{app_data.fmt_pct(share)} des mois de création connus valent janvier : Crunchbase "
            "ne connaît souvent que l'année, le mois est donc peu informatif."
        )

    st.markdown("**Variables catégorielles**")
    categorical = app_data.categorical_summary(data, CATEGORICAL_VARIABLES)
    categorical["variable"] = categorical["variable"].map(app_data.label)
    st.dataframe(categorical, hide_index=True)


def _missing_values(data: pd.DataFrame) -> None:
    """Élément 2 : taux de valeurs manquantes par variable, avec le rôle de chaque variable."""
    table = app_data.missing_table(data, app_data.eda_columns(data))
    chart = table.assign(variable=table["variable"].map(app_data.label))
    figure = px.bar(
        chart,
        x="taux_%",
        y="variable",
        color="rôle",
        orientation="h",
        title="Taux de valeurs manquantes par variable",
        labels={"taux_%": "% de valeurs manquantes", "variable": ""},
    )
    figure.update_yaxes(categoryorder="total ascending")
    figure.update_layout(height=140 + 26 * len(chart), legend_title_text="Rôle")
    st.plotly_chart(figure, key="eda_missing_chart")
    st.dataframe(table, hide_index=True)
    st.caption(
        "Les valeurs manquantes ne sont pas remplies dans le dashboard : elles sont imputées dans "
        "le Pipeline du modèle, appris sur les données d'entraînement seulement (pas de fuite)."
    )


def _distributions(data: pd.DataFrame) -> None:
    """Élément 3 : histogrammes et densités (KDE) par issue."""
    options = [column for column in NUMERIC_VARIABLES if column in data.columns and data[column].notna().any()]
    if not options:
        st.info("Aucune variable numérique renseignée avec ces filtres.")
        return
    col1, col2, col3 = st.columns([2, 1, 1])
    column = col1.selectbox("Variable numérique", options, format_func=app_data.label, key="eda_numeric")
    bins = col2.slider("Nombre de classes", 10, 100, 40, key="eda_bins")
    density = col3.checkbox(
        "Densité + KDE",
        value=True,
        key="eda_density",
        help="Normalise chaque groupe : on compare les formes même si les effectifs diffèrent.",
    )

    subset = data.loc[data[column].notna(), [column, "issue"]].copy()
    subset[column] = subset[column].astype("float64")
    figure = px.histogram(
        subset,
        x=column,
        color="issue",
        nbins=bins,
        barmode="overlay",
        opacity=0.55,
        histnorm="probability density" if density else None,
        color_discrete_map=app_data.ISSUE_COLORS,
        category_orders={"issue": list(app_data.ISSUE_ORDER)},
        labels={column: app_data.label(column), "issue": "Issue"},
    )
    if density:
        for issue, group in subset.groupby("issue"):
            curve = app_data.gaussian_kde(group[column])
            if curve is not None:
                grid, values = curve
                figure.add_trace(
                    go.Scatter(
                        x=grid,
                        y=values,
                        mode="lines",
                        name=f"KDE · {issue}",
                        line={"color": app_data.ISSUE_COLORS.get(issue, "#333333"), "width": 2},
                    )
                )
    figure.update_layout(yaxis_title="Densité" if density else "Nombre de startups", legend_title_text="")
    st.plotly_chart(figure, key="eda_distribution_chart")
    st.dataframe(subset.groupby("issue")[column].describe().round(2))
    if column in NOT_MODEL_INPUTS:
        st.caption("Variable exclue des entrées du modèle (postérieure à la création) : affichée pour la description.")
    if column == "funding_total_usd":
        st.caption("Distribution très asymétrique : préférez log10(financement total USD).")


def _frequencies(data: pd.DataFrame) -> None:
    """Élément 4 : fréquences des modalités par issue, avec taux de succès."""
    col1, col2, col3 = st.columns([2, 1, 1])
    column = col1.selectbox(
        "Variable catégorielle", CATEGORICAL_VARIABLES, format_func=app_data.label, key="eda_categorical"
    )
    top_n = col2.slider("Modalités affichées", 5, 30, 15, key="eda_top_n")
    percent = col3.checkbox(
        "En % par modalité", value=False, key="eda_percent", help="Répartition des issues dans chaque modalité."
    )

    table = app_data.frequency_by_issue(data, column, top_n=top_n)
    issues = [issue for issue in app_data.ISSUE_ORDER if issue in table.columns]
    long = table[issues].reset_index().melt(id_vars=column, var_name="issue", value_name="startups")
    if percent:
        long["valeur"] = 100 * long["startups"] / long.groupby(column)["startups"].transform("sum")
    else:
        long["valeur"] = long["startups"]
    figure = px.bar(
        long,
        x=column,
        y="valeur",
        color="issue",
        color_discrete_map=app_data.ISSUE_COLORS,
        category_orders={column: [str(value) for value in table.index], "issue": issues},
        labels={column: app_data.label(column), "issue": "Issue"},
        title=f"{app_data.label(column)} : effectifs par issue",
    )
    figure.update_layout(
        barmode="stack",
        yaxis_title="% des startups de la modalité" if percent else "Nombre de startups",
        legend_title_text="",
    )
    st.plotly_chart(figure, key="eda_frequency_chart")
    st.dataframe(table)
    st.caption(
        "taux_succès_% : part de la classe 1 parmi les startups dont l'issue est connue. Les "
        "modalités rares sont regroupées dans « Autres » ; les manquants forment « Inconnu (manquant) »."
    )


def _correlations(data: pd.DataFrame) -> None:
    """Élément 5 : corrélations numériques et associations catégorielles."""
    method = st.radio(
        "Méthode pour les variables numériques",
        ("spearman", "pearson"),
        horizontal=True,
        key="eda_corr_method",
        format_func=lambda name: "Spearman (rangs, robuste)" if name == "spearman" else "Pearson (linéaire)",
    )
    corr = app_data.correlation_matrix(data, CORRELATION_COLUMNS, method=method)
    if corr.shape[0] >= 2:
        names = [app_data.label(column) for column in corr.columns]
        figure = px.imshow(
            corr.to_numpy(),
            x=names,
            y=names,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            aspect="auto",
            title=f"Corrélation de {method.capitalize()} (paires complètes)",
        )
        st.plotly_chart(figure, key="eda_corr_chart")
    else:
        st.info("Pas assez de valeurs numériques avec ces filtres.")
    st.caption(
        "Corrélation ne veut pas dire causalité. Le financement est exclu du modèle parce qu'il est "
        "postérieur à la création, pas à cause de sa corrélation avec la cible."
    )

    st.markdown("**Variables catégorielles : V de Cramér** (0 = aucune association, 1 = association parfaite)")
    categorical = data.assign(target=data["target"].astype("string"))
    cramer = app_data.cramers_v_matrix(categorical, [*features.CATEGORICAL_FEATURES, "target"], top_n=20)
    names = [app_data.label(column) for column in cramer.columns]
    figure = px.imshow(
        cramer.to_numpy(),
        x=names,
        y=names,
        zmin=0,
        zmax=1,
        color_continuous_scale="Blues",
        text_auto=".2f",
        aspect="auto",
    )
    st.plotly_chart(figure, key="eda_cramer_chart")
    st.caption(
        "Calculé sur les 20 modalités les plus fréquentes de chaque variable (les autres sont "
        "regroupées). Les variables géographiques sont fortement liées entre elles (redondance)."
    )


def _preview(data: pd.DataFrame) -> None:
    """Aperçu des lignes filtrées."""
    with st.expander("Aperçu des données filtrées (200 premières lignes)"):
        columns = [column for column in ("name", *features.FEATURE_COLUMNS, "status", "issue") if column in data.columns]
        st.dataframe(data.loc[:, columns].head(200), hide_index=True)


def render(frame: pd.DataFrame, source: str) -> None:
    """Affiche le dashboard EDA complet."""
    st.title("Dashboard d'analyse exploratoire (EDA)")
    st.caption(
        f"Source : `{source}` · {app_data.fmt_int(len(frame))} startups dans le fichier brut. "
        "Les filtres de la barre latérale s'appliquent à tous les onglets."
    )
    data = _sidebar_filters(frame)
    if data.empty:
        st.warning("Aucune startup ne correspond aux filtres : élargissez la sélection dans la barre latérale.")
        return

    tabs = st.tabs(list(TAB_NAMES))
    sections = (_descriptive_statistics, _missing_values, _distributions, _frequencies, _correlations)
    for tab, section in zip(tabs, sections):
        with tab:
            section(data)
    _preview(data)
