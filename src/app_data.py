"""Données et calculs du dashboard Streamlit (aucune dépendance à Streamlit).

Projet : Prédiction du succès ou de l'échec des startups à partir de leurs
caractéristiques initiales.

Ce module réutilise les modules déjà présents sur ``main`` au lieu de les recopier :

- ``data_loader`` : lecture du CSV Kaggle brut ;
- ``features``    : variables initiales du modèle (FEATURE_COLUMNS) ;
- ``targets``     : définition officielle de la cible (TARGET_MAPPING) ;
- ``splitting``   : variables interdites en entrée (FORBIDDEN_FEATURES).

Les fonctions sont pures (DataFrame en entrée, résultat en sortie) et sont
testées dans ``tests/test_app_data.py`` sans lancer Streamlit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import data_loader
import features
import splitting
import targets

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_PATH_ENV = "STARTUP_DATA_PATH"  # chemin explicite du CSV (tests, démo)

LABEL_SUCCESS = "Succès (IPO / acquisition)"
LABEL_FAILURE = "Échec (fermeture)"
LABEL_EXCLUDED = "Exclu (operating)"
LABEL_UNKNOWN = "Statut inconnu"
ISSUE_ORDER = (LABEL_SUCCESS, LABEL_FAILURE, LABEL_EXCLUDED, LABEL_UNKNOWN)
MODELING_ISSUES = (LABEL_SUCCESS, LABEL_FAILURE)
ISSUE_COLORS = {  # palette Okabe-Ito : lisible par les personnes daltoniennes
    LABEL_SUCCESS: "#0072B2",
    LABEL_FAILURE: "#D55E00",
    LABEL_EXCLUDED: "#999999",
    LABEL_UNKNOWN: "#CC79A7",
}

MISSING_LABEL = "Inconnu (manquant)"
OTHER_LABEL = "Autres"
IDENTIFIER_COLUMNS = ("permalink", "name", "homepage_url")
FUNDING_LOG_COLUMN = "log10_funding_total_usd"
TECHNICAL_COLUMNS = ("issue", "target", FUNDING_LOG_COLUMN)

COLUMN_LABELS = {
    "country_code": "Pays (code)",
    "state_code": "État / province",
    "region": "Région",
    "city": "Ville",
    "primary_category": "Secteur principal",
    "founding_year": "Année de création",
    "founding_month": "Mois de création",
    "funding_rounds": "Tours de financement",
    "funding_total_usd": "Financement total (USD)",
    FUNDING_LOG_COLUMN: "log10(financement total USD)",
    "status": "Statut Crunchbase",
    "target": "Cible (1 = IPO / acquisition)",
}

DISCLAIMER = (
    "Ce système classe le profil d'une startup par rapport aux issues observées dans un "
    "échantillon historique de Crunchbase. Il constitue une aide à l'analyse, pas une "
    "prédiction certaine de son avenir. Les startups encore actives (operating) sont "
    "exclues de la cible : les scores sont conditionnels à cette sélection."
)


# ------------------------------------------------------------------ affichage
def label(column: str) -> str:
    """Libellé lisible d'une colonne (nom technique s'il n'est pas prévu)."""
    return COLUMN_LABELS.get(column, column)


def fmt_int(value) -> str:
    """Entier au format français : 13 334."""
    return f"{int(value):,}".replace(",", "\u202f")


def fmt_pct(value, digits: int = 1) -> str:
    """Proportion (entre 0 et 1) au format français : 53,2 %. Tiret si absente."""
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.{digits}f}".replace(".", ",") + "\u00a0%"


def relative_to_project(path: str | Path) -> str:
    """Chemin lisible, relatif à la racine du dépôt quand c'est possible."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _as_bool(mask: pd.Series) -> pd.Series:
    """Masque strict : les NA des types nullables pandas deviennent False."""
    return mask.fillna(False).astype(bool)


# ----------------------------------------------------------------- chargement
def find_dataset(base: str | Path = PROJECT_ROOT) -> Path | None:
    """CSV brut à utiliser : STARTUP_DATA_PATH, sinon data_loader.find_raw_file."""
    override = os.environ.get(DATA_PATH_ENV)
    if override:
        path = Path(override)
        return path if path.is_file() else None
    return data_loader.find_raw_file(base)


def build_eda_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Prépare le DataFrame du dashboard à partir du CSV brut.

    - variables du modèle : ``features.build_features`` ;
    - cible officielle : ``targets.TARGET_MAPPING`` (NA pour operating) ;
    - ``issue`` : libellé lisible de la cible ;
    - financement converti en nombres (un texte comme « - » devient manquant).

    Aucune ligne n'est supprimée : le filtre « Issue » du dashboard choisit le
    périmètre (par défaut, celui de la modélisation).
    """
    frame = features.build_features(raw)

    status = frame["status"].astype("string").str.strip().str.lower()
    frame["status"] = status
    target = pd.to_numeric(status.map(targets.TARGET_MAPPING), errors="coerce")
    frame["target"] = target.astype("Int8")

    issue = pd.Series(LABEL_UNKNOWN, index=frame.index, dtype="object")
    issue = issue.mask(_as_bool(target.eq(1)), LABEL_SUCCESS)
    issue = issue.mask(_as_bool(target.eq(0)), LABEL_FAILURE)
    issue = issue.mask(_as_bool(status.eq("operating")), LABEL_EXCLUDED)
    frame["issue"] = issue

    for column in ("funding_total_usd", "funding_rounds"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "funding_total_usd" in frame.columns:
        positive = frame["funding_total_usd"].where(frame["funding_total_usd"] > 0)
        frame[FUNDING_LOG_COLUMN] = np.log10(positive)
    return frame


# --------------------------------------------------------- descriptions (EDA)
def variable_role(column: str) -> str:
    """Rôle d'une colonne dans le projet (affiché à côté des manquants)."""
    if column in features.FEATURE_COLUMNS:
        return "Entrée du modèle"
    if column in getattr(features, "EDA_ONLY_COLUMNS", ()):
        return "Analyse seulement (exclue du modèle)"
    if column == "status":
        return "Sert uniquement à construire la cible"
    if column in splitting.FORBIDDEN_FEATURES or column == FUNDING_LOG_COLUMN:
        return "Exclue du modèle (postérieure à la création)"
    if column in IDENTIFIER_COLUMNS:
        return "Identifiant (non utilisé)"
    if column in ("category_list", "founded_at"):
        return "Brute (source d'une entrée)"
    return "Autre"


def eda_columns(frame: pd.DataFrame) -> list[str]:
    """Colonnes décrites dans le dashboard (sans les colonnes techniques)."""
    return [column for column in frame.columns if column not in TECHNICAL_COLUMNS]


def missing_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Nombre et taux de valeurs manquantes par colonne, du plus au moins touché."""
    columns = list(frame.columns) if columns is None else list(columns)
    counts = frame.loc[:, columns].isna().sum()
    table = pd.DataFrame(
        {
            "variable": [str(column) for column in counts.index],
            "manquants": counts.to_numpy(dtype="int64"),
            "taux_%": np.round(counts.to_numpy(dtype="float64") * 100 / max(len(frame), 1), 2),
        }
    )
    table["rôle"] = table["variable"].map(variable_role)
    return table.sort_values(["manquants", "variable"], ascending=[False, True], ignore_index=True)


_SUMMARY_NAMES = {
    "count": "n",
    "mean": "moyenne",
    "std": "écart-type",
    "min": "min",
    "25%": "Q1",
    "50%": "médiane",
    "75%": "Q3",
    "max": "max",
}


def numeric_summary(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Statistiques descriptives des variables numériques (une ligne par variable)."""
    present = [column for column in columns if column in frame.columns]
    if not present:
        return pd.DataFrame()
    values = frame.loc[:, present].apply(pd.to_numeric, errors="coerce").astype("float64")
    summary = values.describe().T.rename(columns=_SUMMARY_NAMES)
    summary["manquants_%"] = values.isna().mean() * 100
    summary["asymétrie"] = values.skew()
    return summary.round(2)


def categorical_summary(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Résumé des variables catégorielles : renseignement, modalités, modalité dominante."""
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype(str)
        counts = values.value_counts()
        has_values = not counts.empty
        rows.append(
            {
                "variable": column,
                "renseignés": int(values.size),
                "manquants_%": round(100 * (1 - values.size / max(len(frame), 1)), 2),
                "modalités": int(counts.size),
                "plus fréquente": counts.index[0] if has_values else None,
                "part_%": round(100 * counts.iloc[0] / values.size, 2) if has_values else 0.0,
            }
        )
    return pd.DataFrame(rows)


def top_values(frame: pd.DataFrame, column: str, limit: int | None = None) -> list[str]:
    """Modalités observées, de la plus à la moins fréquente."""
    if column not in frame.columns:
        return []
    counts = frame[column].dropna().astype(str).value_counts()
    values = [str(value) for value in counts.index]
    return values if limit is None else values[:limit]


# -------------------------------------------------------- filtre (élément 6)
def apply_filters(
    frame: pd.DataFrame,
    issues: Iterable[str] | None = None,
    countries: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
    year_range: tuple[int, int] | None = None,
    include_unknown_year: bool = True,
) -> pd.DataFrame:
    """Filtres interactifs du dashboard.

    - ``issues`` : None = pas de filtre ; liste vide = aucune ligne ;
    - ``countries`` / ``categories`` : None ou vide = tous ;
    - ``year_range`` : bornes incluses sur founding_year ; les années inconnues
      sont gardées si ``include_unknown_year`` est vrai.
    """
    mask = pd.Series(True, index=frame.index)
    if issues is not None:
        mask &= _as_bool(frame["issue"].isin(list(issues)))
    if countries:
        mask &= _as_bool(frame["country_code"].isin(list(countries)))
    if categories:
        mask &= _as_bool(frame["primary_category"].isin(list(categories)))
    if year_range is not None:
        low, high = year_range
        year = frame["founding_year"]
        keep = year.between(low, high)
        if include_unknown_year:
            keep |= year.isna()
        mask &= keep
    return frame.loc[mask]


# ------------------------------------------------- fréquences (élément 4)
def group_rare(series: pd.Series, top_n: int) -> pd.Series:
    """Garde les top_n modalités et regroupe les autres dans « Autres ».

    Les valeurs manquantes deviennent une modalité explicite.
    """
    values = series.astype("object").where(series.notna(), MISSING_LABEL)
    top = values.value_counts().index[:top_n]
    return values.where(values.isin(top), OTHER_LABEL)


def frequency_by_issue(frame: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    """Effectifs par modalité et par issue, avec le taux de succès de chaque modalité."""
    grouped = group_rare(frame[column], top_n).rename(column)
    table = pd.crosstab(grouped, frame["issue"])
    table = table.reindex(columns=[issue for issue in ISSUE_ORDER if issue in table.columns])
    table["total"] = table.sum(axis=1)

    target = frame["target"].astype("float64")
    known = target.notna().groupby(grouped).sum()
    rate = target.groupby(grouped).mean() * 100
    table["issue_connue"] = known.reindex(table.index).fillna(0).astype(int)
    table["taux_succès_%"] = rate.reindex(table.index).round(1)

    order = table.sort_values("total", ascending=False).index.tolist()
    if OTHER_LABEL in order:  # « Autres » toujours en dernier
        order.remove(OTHER_LABEL)
        order.append(OTHER_LABEL)
    table = table.loc[order]
    table.index.name = column
    table.columns.name = None
    return table


# ------------------------------------------------- distributions (élément 3)
def gaussian_kde(values, n_points: int = 200, max_samples: int = 5000, seed: int = 42):
    """Densité par noyau gaussien (règle de Silverman), en NumPy pur.

    Retourne (grille, densité), ou None s'il y a moins de deux valeurs
    distinctes. Au-delà de max_samples, un sous-échantillon fixe (seed)
    limite le temps de calcul.
    """
    sample = pd.to_numeric(pd.Series(values), errors="coerce").astype("float64").to_numpy()
    sample = sample[np.isfinite(sample)]
    if sample.size < 2 or np.unique(sample).size < 2:
        return None
    if sample.size > max_samples:
        sample = np.random.default_rng(seed).choice(sample, size=max_samples, replace=False)
    std = sample.std(ddof=1)
    q75, q25 = np.percentile(sample, [75, 25])
    spread = min(std, (q75 - q25) / 1.349) if q75 > q25 else std
    bandwidth = 0.9 * spread * sample.size ** (-0.2)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        return None
    grid = np.linspace(sample.min() - 3 * bandwidth, sample.max() + 3 * bandwidth, n_points)
    z = (grid[:, None] - sample[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1) / (sample.size * bandwidth * np.sqrt(2 * np.pi))
    return grid, density


# ------------------------------------------------- corrélations (élément 5)
def correlation_matrix(
    frame: pd.DataFrame, columns: Sequence[str], method: str = "spearman", min_periods: int = 30
) -> pd.DataFrame:
    """Matrice de corrélation sur paires complètes (Pearson ou Spearman)."""
    if method not in {"pearson", "spearman"}:
        raise ValueError("method doit valoir 'pearson' ou 'spearman'.")
    present = [column for column in columns if column in frame.columns]
    values = frame.loc[:, present].apply(pd.to_numeric, errors="coerce").astype("float64")
    values = values.loc[:, values.std() > 0]  # retire les colonnes constantes ou vides
    return values.corr(method=method, min_periods=min_periods)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér (sans correction de biais) entre deux variables catégorielles."""
    table = pd.crosstab(x, y).to_numpy(dtype="float64")
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan")
    n = table.sum()
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())
    return float(min(1.0, np.sqrt(chi2 / (n * (min(table.shape) - 1)))))


def cramers_v_matrix(frame: pd.DataFrame, columns: Sequence[str], top_n: int = 20) -> pd.DataFrame:
    """Matrice des V de Cramér ; modalités rares regroupées pour limiter le biais."""
    present = [column for column in columns if column in frame.columns]
    grouped = {column: group_rare(frame[column], top_n) for column in present}
    matrix = pd.DataFrame(np.eye(len(present)), index=present, columns=present)
    for i, first in enumerate(present):
        for second in present[i + 1 :]:
            value = cramers_v(grouped[first], grouped[second])
            matrix.loc[first, second] = value
            matrix.loc[second, first] = value
    return matrix


# -------------------------------------------- aide à la page de prédiction
def modeling_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Lignes du périmètre de modélisation (issue connue : cible 0 ou 1)."""
    return frame.loc[frame["target"].notna()]


def base_rate(frame: pd.DataFrame) -> float | None:
    """Part de la classe 1 parmi les lignes dont l'issue est connue."""
    known = frame["target"].dropna()
    return float(known.astype("float64").mean()) if len(known) else None


def observed_options(
    frame: pd.DataFrame, column: str, constraints: Mapping[str, str | None] | None = None
) -> list[str]:
    """Valeurs vues dans le périmètre de modélisation, éventuellement filtrées
    (par exemple les États observés pour le pays choisi)."""
    subset = modeling_rows(frame)
    for name, value in (constraints or {}).items():
        if value is not None and name in subset.columns:
            subset = subset.loc[_as_bool(subset[name].eq(value))]
    return top_values(subset, column)


def similar_profiles(
    frame: pd.DataFrame, country: str | None = None, category: str | None = None
) -> tuple[int, float | None]:
    """Effectif et part de classe 1 des startups du même pays et/ou secteur."""
    subset = modeling_rows(frame)
    if country is not None:
        subset = subset.loc[_as_bool(subset["country_code"].eq(country))]
    if category is not None:
        subset = subset.loc[_as_bool(subset["primary_category"].eq(category))]
    return len(subset), base_rate(subset)


def max_founding_year(frame: pd.DataFrame) -> int | None:
    """Dernière année de création observée dans le périmètre de modélisation."""
    years = modeling_rows(frame)["founding_year"].dropna()
    return int(years.max()) if len(years) else None


def january_share(frame: pd.DataFrame) -> float | None:
    """Part des mois de création connus égaux à janvier."""
    months = frame["founding_month"].dropna()
    return float((months == 1).mean()) if len(months) else None
