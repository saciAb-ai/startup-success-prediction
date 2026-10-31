"""Dashboard EDA : les 6 éléments exigés et une vue géographie et tendances.

Éléments exigés : statistiques descriptives (vue d'ensemble), valeurs
manquantes, histogrammes et densités (KDE), fréquences catégorielles,
corrélations et filtres interactifs (barre latérale). L'onglet « Géographie et
tendances » est un complément. Les calculs viennent de app_data et
app_insights, testés sans Streamlit ; ce module ne fait que l'affichage.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import app_data
import app_insights
import features
import ui

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
ROLE_COLORS = {
    "Entrée du modèle": ui.ACCENT,
    "Brute (source d'une entrée)": "#A99BF2",
    "Sert uniquement à construire la cible": ui.INK,
    "Exclue du modèle (postérieure à la création)": "#8A94A6",
    "Identifiant (non utilisé)": "#C3CAD8",
    "Autre": "#DADFE9",
}
CORRELATION_SCALE = [[0.0, "#0F766E"], [0.5, "#F4F4F6"], [1.0, ui.ACCENT]]
ASSOCIATION_SCALE = [[0.0, "#F4F4F6"], [1.0, ui.ACCENT]]
TABS = (
    ":material/dashboard: Vue d'ensemble",
    ":material/rule: Valeurs manquantes",
    ":material/bar_chart: Distributions",
    ":material/category: Catégories",
    ":material/hub: Corrélations",
    ":material/public: Géographie et tendances",
)


# ------------------------------------------------------ filtres (élément 6)
def _filter_defaults(frame: pd.DataFrame):
    present = set(frame["issue"])
    issues = [issue for issue in app_data.MODELING_ISSUES if issue in present]
    years = frame["founding_year"].dropna()
    span = (int(years.min()), int(years.max())) if years.nunique() > 1 else None
    return issues, span


def _reset_filters(issues, span) -> None:
    state = st.session_state
    state["eda_issues"] = list(issues)
    state["eda_countries"] = []
    state["eda_categories"] = []
    if span is not None:
        state["eda_years"] = span
    state["eda_unknown_years"] = True
    state["eda_reset_done"] = True


def _sidebar_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """Élément 6 : filtres interactifs, appliqués à tous les onglets."""
    issues, span = _filter_defaults(frame)
    state = st.session_state
    state.setdefault("eda_issues", issues)
    state.setdefault("eda_countries", [])
    state.setdefault("eda_categories", [])
    if span is not None:
        state.setdefault("eda_years", span)
    state.setdefault("eda_unknown_years", True)

    sidebar = st.sidebar
    sidebar.markdown("#### :material/tune: Filtres")
    present = set(frame["issue"])
    sidebar.multiselect(
        "Issue",
        [issue for issue in app_data.ISSUE_ORDER if issue in present],
        key="eda_issues",
        placeholder="Choisir une issue",
        help="Par défaut : les startups dont l'issue est connue (sortie réussie ou fermeture).",
    )
    sidebar.multiselect(
        "Pays",
        app_data.top_values(frame, "country_code"),
        key="eda_countries",
        placeholder="Tous les pays",
        help="Tape pour chercher, par exemple USA.",
    )
    sidebar.multiselect(
        "Secteur principal",
        app_data.top_values(frame, "primary_category"),
        key="eda_categories",
        placeholder="Tous les secteurs",
        help="Première catégorie Crunchbase de la startup.",
    )
    year_range = None
    if span is not None:
        year_range = sidebar.slider("Année de création", span[0], span[1], key="eda_years")
    include_unknown = sidebar.checkbox("Inclure les années inconnues", key="eda_unknown_years")
    sidebar.button(
        "Réinitialiser les filtres",
        icon=":material/restart_alt:",
        key="eda_reset",
        on_click=_reset_filters,
        args=(issues, span),
    )
    filtered = app_data.apply_filters(
        frame,
        issues=state["eda_issues"],
        countries=state["eda_countries"],
        categories=state["eda_categories"],
        year_range=year_range,
        include_unknown_year=include_unknown,
    )
    sidebar.caption(
        f"{app_data.fmt_int(len(filtered))} startups sur {app_data.fmt_int(len(frame))} après filtrage."
    )
    if state.pop("eda_reset_done", False):
        st.toast("Filtres réinitialisés.", icon=":material/check_circle:")
    return filtered


# ------------------------------------------------------- en-tête et cartes
def _header(data: pd.DataFrame, source: str) -> None:
    st.title("Tableau de bord exploratoire (EDA)")
    st.markdown(
        "Ce qui distingue, dès leur création, les startups qui ont connu une sortie "
        f"(IPO ou acquisition) de celles qui ont fermé. Données : `{source}`."
    )
    ui.outcome_split(app_data.base_rate(data), len(app_data.modeling_rows(data)))


def _kpis(frame: pd.DataFrame, data: pd.DataFrame) -> None:
    rate = app_data.base_rate(data)
    overall = app_data.base_rate(frame)
    delta = None
    if rate is not None and overall is not None and abs(rate - overall) >= 0.0005:
        delta = f"{(rate - overall) * 100:+.1f} pts vs ensemble".replace(".", ",")
    cards = st.columns(4)
    with cards[0].container(border=True):
        st.metric(
            ":material/database: Startups affichées",
            app_data.fmt_int(len(data)),
            help=f"Sur {app_data.fmt_int(len(frame))} startups dans le fichier brut.",
        )
    with cards[1].container(border=True):
        st.metric(
            ":material/flag: Issue connue",
            app_data.fmt_int(len(app_data.modeling_rows(data))),
            help="Startups introduites en bourse, acquises ou fermées.",
        )
    with cards[2].container(border=True):
        st.metric(
            ":material/trending_up: Taux de succès",
            app_data.fmt_pct(rate),
            delta=delta,
            help="Part des IPO et acquisitions parmi les startups dont l'issue est connue.",
        )
    with cards[3].container(border=True):
        st.metric(
            ":material/public: Pays",
            app_data.fmt_int(data["country_code"].nunique()),
            help="Nombre de pays représentés dans la sélection.",
        )


# ----------------------------------------------------------------- onglets
def _overview(data: pd.DataFrame) -> None:
    """Élément 1 : statistiques descriptives."""
    ui.note(
        "Statistiques descriptives : le nombre de startups par issue, puis le résumé de chaque "
        "variable. Survole une barre pour lire la valeur exacte."
    )
    counts = data["issue"].value_counts()
    issues = [issue for issue in app_data.ISSUE_ORDER if issue in counts.index]
    figure = px.bar(
        pd.DataFrame({"issue": issues, "startups": [int(counts[issue]) for issue in issues]}),
        x="startups",
        y="issue",
        orientation="h",
        color="issue",
        text="startups",
        color_discrete_map=app_data.ISSUE_COLORS,
        title="Répartition des issues",
    )
    figure.update_traces(
        texttemplate="%{x:,}",
        textposition="inside",
        insidetextanchor="end",
        hovertemplate="%{y} : %{x:,} startups<extra></extra>",
    )
    figure.update_layout(xaxis_title="Nombre de startups", yaxis={"title": "", "categoryorder": "total ascending"})
    ui.chart(figure, key="eda_issue_chart", height=120 + 52 * len(issues), legend=False)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("##### Variables chiffrées")
        summary = app_data.numeric_summary(data, NUMERIC_VARIABLES)
        if not summary.empty:
            summary.index = [app_data.label(column) for column in summary.index]
            st.dataframe(summary)
        st.caption(
            "Le financement est décrit ici mais exclu des entrées du modèle : "
            "il n'est connu qu'après la création."
        )
        share = app_data.january_share(data)
        if share is not None:
            st.caption(
                f"{app_data.fmt_pct(share)} des mois de création connus valent janvier : "
                "Crunchbase ne connaît souvent que l'année."
            )
    with right:
        st.markdown("##### Variables texte")
        categorical = app_data.categorical_summary(data, CATEGORICAL_VARIABLES)
        categorical["variable"] = categorical["variable"].map(app_data.label)
        st.dataframe(
            categorical.loc[:, ["variable", "modalités", "plus fréquente", "part_%"]],
            hide_index=True,
            column_config={
                "part_%": st.column_config.ProgressColumn(
                    "Part de la plus fréquente (%)", min_value=0, max_value=100, format="%.1f"
                )
            },
        )


def _missing_values(data: pd.DataFrame) -> None:
    """Élément 2 : valeurs manquantes."""
    ui.note(
        "Part des cases vides dans chaque colonne. La couleur donne le rôle de la colonne ; "
        "les vides des entrées du modèle sont complétés automatiquement dans le pipeline, "
        "sur les données d'entraînement seulement."
    )
    table = app_data.missing_table(data, app_data.eda_columns(data))
    readable = table.assign(variable=table["variable"].map(app_data.label))
    figure = px.bar(
        readable,
        x="taux_%",
        y="variable",
        color="rôle",
        orientation="h",
        color_discrete_map=ROLE_COLORS,
        title="Taux de valeurs manquantes",
        labels={"taux_%": "% de cases vides", "variable": "", "rôle": "Rôle"},
    )
    figure.update_yaxes(categoryorder="total ascending")
    figure.update_traces(hovertemplate="%{y} : %{x:.1f} % de cases vides<extra></extra>")
    ui.chart(figure, key="eda_missing_chart", height=160 + 26 * len(readable))
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "taux_%": st.column_config.ProgressColumn("Cases vides (%)", min_value=0, max_value=100, format="%.1f")
        },
    )


def _distributions(data: pd.DataFrame) -> None:
    """Élément 3 : histogrammes et densités (KDE)."""
    ui.note(
        "Compare la forme des distributions : sorties réussies en bleu, fermetures en vermillon. "
        "La boîte du haut résume chaque groupe (médiane et quartiles). Clique sur un nom de la "
        "légende pour masquer un groupe."
    )
    options = [column for column in NUMERIC_VARIABLES if column in data.columns]
    column = (
        st.pills("Variable", options, default=options[0], format_func=app_data.label, key="eda_numeric")
        or options[0]
    )
    left, right = st.columns([2, 3], gap="large")
    with left:
        mode = (
            st.segmented_control("Affichage", ["Densité", "Effectifs"], default="Densité", key="eda_hist_mode")
            or "Densité"
        )
    with right:
        bins = st.slider("Nombre de classes", 10, 100, 40, key="eda_bins")
    subset = data.loc[data[column].notna(), [column, "issue"]].copy()
    if subset.empty:
        st.info("Aucune valeur pour cette variable avec les filtres actuels.")
        return
    subset[column] = subset[column].astype("float64")
    density = mode == "Densité"
    figure = px.histogram(
        subset,
        x=column,
        color="issue",
        nbins=bins,
        barmode="overlay",
        opacity=0.6,
        marginal="box",
        histnorm="probability density" if density else None,
        color_discrete_map=app_data.ISSUE_COLORS,
        category_orders={"issue": list(app_data.ISSUE_ORDER)},
        labels={column: app_data.label(column), "issue": "Issue"},
        title=f"Distribution : {app_data.label(column)}",
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
                        name=f"Densité lissée ({issue})",
                        line={"color": app_data.ISSUE_COLORS.get(issue, ui.INK), "width": 2.5},
                    )
                )
    figure.update_layout(yaxis_title="Densité" if density else "Nombre de startups")
    ui.chart(figure, key="eda_distribution_chart", height=480)
    st.dataframe(subset.groupby("issue")[column].describe().round(2))
    if column in NOT_MODEL_INPUTS:
        st.caption("Variable exclue des entrées du modèle (connue après la création) : affichée pour la description.")
    if column == "funding_total_usd":
        st.caption("Distribution très étirée : préfère la version log10 du financement.")


def _categories(data: pd.DataFrame) -> None:
    """Élément 4 : fréquences catégorielles."""
    ui.note(
        "Effectifs de chaque modalité, découpés par issue. En pourcentages, chaque barre fait 100 % : "
        "compare directement la part de succès. « Autres » regroupe les modalités rares."
    )
    column = (
        st.pills(
            "Variable", CATEGORICAL_VARIABLES, default="country_code", format_func=app_data.label, key="eda_categorical"
        )
        or "country_code"
    )
    left, right = st.columns([2, 3], gap="large")
    with left:
        mode = (
            st.segmented_control("Échelle", ["Effectifs", "Pourcentages"], default="Effectifs", key="eda_freq_mode")
            or "Effectifs"
        )
    with right:
        top_n = st.slider("Modalités affichées", 5, 30, 12, key="eda_top_n")
    table = app_data.frequency_by_issue(data, column, top_n=top_n)
    issues = [issue for issue in app_data.ISSUE_ORDER if issue in table.columns]
    long = table[issues].reset_index().melt(id_vars=column, var_name="issue", value_name="startups")
    percent = mode == "Pourcentages"
    if percent:
        long["valeur"] = 100 * long["startups"] / long.groupby(column)["startups"].transform("sum")
    else:
        long["valeur"] = long["startups"]
    figure = px.bar(
        long,
        x="valeur",
        y=column,
        color="issue",
        orientation="h",
        color_discrete_map=app_data.ISSUE_COLORS,
        category_orders={column: [str(value) for value in table.index][::-1], "issue": issues},
        labels={column: "", "issue": "Issue", "valeur": "% de la modalité" if percent else "Nombre de startups"},
        title=f"{app_data.label(column)} : répartition des issues",
    )
    figure.update_layout(barmode="stack")
    figure.update_traces(
        hovertemplate="%{y}<br>%{fullData.name} : %{x:.1f} %<extra></extra>"
        if percent
        else "%{y}<br>%{fullData.name} : %{x:,} startups<extra></extra>"
    )
    ui.chart(figure, key="eda_frequency_chart", height=150 + 30 * len(table))
    st.dataframe(
        table,
        column_config={
            "taux_succès_%": st.column_config.ProgressColumn(
                "Taux de succès (%)", min_value=0, max_value=100, format="%.1f"
            ),
            "total": st.column_config.NumberColumn("Total"),
            "issue_connue": st.column_config.NumberColumn("Issue connue"),
        },
    )


def _correlations(data: pd.DataFrame) -> None:
    """Élément 5 : corrélations."""
    ui.note(
        "Chaque case mesure le lien entre deux variables. Variables chiffrées : de -1 à +1, "
        "proche de 0 = pas de lien. Variables texte (V de Cramér) : de 0 à 1. "
        "Un lien n'est pas une cause."
    )
    method = (
        st.segmented_control(
            "Méthode",
            ["spearman", "pearson"],
            default="spearman",
            key="eda_corr_method",
            format_func=lambda name: "Spearman (rangs)" if name == "spearman" else "Pearson (linéaire)",
        )
        or "spearman"
    )
    left, right = st.columns(2, gap="large")
    with left:
        corr = app_data.correlation_matrix(data, CORRELATION_COLUMNS, method=method)
        if corr.shape[0] >= 2:
            names = [app_data.label(column) for column in corr.columns]
            figure = px.imshow(
                corr.to_numpy(),
                x=names,
                y=names,
                zmin=-1,
                zmax=1,
                color_continuous_scale=CORRELATION_SCALE,
                text_auto=".2f",
                aspect="auto",
                title="Variables chiffrées",
            )
            ui.chart(figure, key="eda_corr_chart", height=470, legend=False)
        else:
            st.info("Pas assez de valeurs numériques avec ces filtres.")
    with right:
        categorical = data.assign(target=data["target"].astype("string"))
        cramer = app_data.cramers_v_matrix(categorical, [*features.CATEGORICAL_FEATURES, "target"], top_n=20)
        names = [app_data.label(column) for column in cramer.columns]
        figure = px.imshow(
            cramer.to_numpy(),
            x=names,
            y=names,
            zmin=0,
            zmax=1,
            color_continuous_scale=ASSOCIATION_SCALE,
            text_auto=".2f",
            aspect="auto",
            title="Variables texte (V de Cramér)",
        )
        ui.chart(figure, key="eda_cramer_chart", height=470, legend=False)
    st.caption(
        "Le financement est exclu du modèle parce qu'il est connu après la création, pas à cause de "
        "sa corrélation avec la cible. Les variables géographiques sont fortement liées entre elles."
    )


def _geography(data: pd.DataFrame) -> None:
    """Complément : carte des taux de succès et évolution par année de création."""
    ui.note(
        "La carte colore chaque pays selon son taux de succès : bleu au-dessus de la moyenne de la "
        "sélection, vermillon en dessous. Survole un pays pour ses chiffres ; molette pour zoomer."
    )
    min_count = st.slider("Nombre minimum de startups par pays", 5, 100, 20, step=5, key="eda_min_country")
    countries = app_insights.success_by_country(data, min_count=min_count)
    base = app_data.base_rate(data)
    if countries.empty or base is None:
        st.info("Aucun pays n'atteint ce minimum avec les filtres actuels.")
    else:
        figure = px.choropleth(
            countries,
            locations="country_code",
            locationmode="ISO-3",
            color="taux_succès",
            hover_name="country_code",
            hover_data={"startups": True, "taux_succès": ":.1f", "country_code": False},
            color_continuous_scale=ui.DIVERGING_SCALE,
            color_continuous_midpoint=100 * base,
            labels={"taux_succès": "Succès (%)", "startups": "Startups"},
            title="Taux de succès par pays",
        )
        figure.update_geos(
            projection_type="natural earth",
            showframe=False,
            showcoastlines=False,
            showcountries=True,
            countrycolor="#FFFFFF",
            showland=True,
            landcolor="#E6EAF1",
            bgcolor="rgba(0,0,0,0)",
        )
        ui.chart(figure, key="eda_world_map", height=460, legend=False)

    trend = app_insights.success_by_year(data, min_count=20)
    if trend.empty:
        return
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(
        x=trend["founding_year"], y=trend["succès"], name="Sorties réussies", marker_color=ui.SUCCESS, secondary_y=False
    )
    figure.add_bar(
        x=trend["founding_year"], y=trend["échecs"], name="Fermetures", marker_color=ui.FAILURE, secondary_y=False
    )
    figure.add_scatter(
        x=trend["founding_year"],
        y=trend["taux_succès"],
        name="Taux de succès (%)",
        mode="lines+markers",
        line={"color": ui.INK, "width": 2},
        marker={"size": 5},
        connectgaps=False,
        secondary_y=True,
    )
    figure.update_layout(
        barmode="stack", hovermode="x unified", title="Startups par année de création et taux de succès"
    )
    figure.update_yaxes(title_text="Startups", secondary_y=False)
    figure.update_yaxes(title_text="Taux de succès (%)", range=[0, 100], showgrid=False, secondary_y=True)
    figure.update_xaxes(title_text="Année de création", rangeslider={"visible": True, "thickness": 0.08})
    ui.chart(figure, key="eda_trend_chart", height=480)
    st.caption(
        "Glisse les poignées sous le graphique pour choisir une période. Pas de taux pour les années "
        "de moins de 20 startups. Les startups récentes ont eu moins de temps pour connaître une "
        "issue : la fin de la courbe se lit avec prudence."
    )


def _preview(data: pd.DataFrame) -> None:
    columns = [column for column in ("name", *features.FEATURE_COLUMNS, "status", "issue") if column in data.columns]
    with st.expander("Données filtrées", icon=":material/table_view:"):
        st.dataframe(data.loc[:, columns].head(200), hide_index=True)
        st.download_button(
            "Télécharger la sélection (CSV)",
            data=data.loc[:, columns].to_csv(index=False).encode("utf-8"),
            file_name="startups_selection.csv",
            mime="text/csv",
            icon=":material/download:",
            key="eda_download",
        )
        st.caption("Le tableau montre les 200 premières lignes ; le fichier contient toute la sélection.")


def render(frame: pd.DataFrame, source: str) -> None:
    """Affiche le dashboard EDA complet."""
    data = _sidebar_filters(frame)
    _header(data, source)
    if data.empty:
        st.warning(
            "Aucune startup ne correspond aux filtres : élargis la sélection ou clique sur « Réinitialiser les filtres »."
        )
        return
    _kpis(frame, data)
    tabs = st.tabs(list(TABS))
    sections = (_overview, _missing_values, _distributions, _categories, _correlations, _geography)
    for tab, section in zip(tabs, sections):
        with tab:
            section(data)
    _preview(data)
