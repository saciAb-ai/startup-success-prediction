"""Tests statistiques du dashboard EDA (aucune dépendance à Streamlit).

Projet : Prédiction du succès ou de l'échec des startups à partir de leurs
caractéristiques initiales.

Ce module implémente la « Phase 1 — inférence formelle » décrite dans
``docs/handoff_eda.md`` (méthodes et valeurs de référence fournies par ZM,
``notebooks/01_eda.ipynb`` §17) :

- test du chi-deux d'indépendance ``primary_category`` × ``target``, avec
  regroupement des modalités rares (< 30 lignes, sensibilité < 100 lignes),
  effectifs attendus, résidus de Pearson et V de Cramér ;
- tests de Mann-Whitney U (bilatéraux, asymptotiques) sur ``founding_year`` et
  ``funding_total_usd`` entre succès et échecs, avec r de rang-bisérial et
  probabilité de supériorité ;
- correction de Holm sur la famille des trois tests primaires.

Les fonctions de calcul reprennent l'annexe B de la passation. Les cas
dégénérés (sélection filtrée trop petite) lèvent ``InferenceError`` avec un
message en français, que l'interface affiche au lieu d'un résultat trompeur.
Aucune imputation, aucun encodage, aucune standardisation : chaque test
utilise les lignes où la variable testée est observée.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

try:  # scipy est déclaré dans requirements.txt ; message clair s'il manque
    from scipy.stats import chi2_contingency, mannwhitneyu
except ImportError:  # pragma: no cover - dépend de l'environnement
    chi2_contingency = mannwhitneyu = None

OTHER = "Other"  # modalité résiduelle interne (notebook)
OTHER_DISPLAY = "Autres"  # libellé affiché, comme le reste du dashboard
PRIMARY_MIN_ROWS = 30
SENSITIVITY_MIN_ROWS = 100
ALPHA = 0.05
P_FLOOR = 1e-300
SCIPY_MISSING = (
    "Le module scipy est introuvable : installe les dépendances avec "
    "« pip install -r requirements.txt » pour afficher les tests statistiques."
)

MW_VARIABLES = ("founding_year", "funding_total_usd")
MW_LABELS = {
    "founding_year": "Année de création",
    "funding_total_usd": "Financement total (USD)",
}
FAMILY_LABELS = (
    "Chi-deux, secteur principal (< 30 lignes → Autres)",
    "Mann-Whitney, année de création",
    "Mann-Whitney, financement total (USD)",
)


class InferenceError(ValueError):
    """Test impossible ou trompeur sur la sélection courante (message affichable)."""


def scipy_available() -> bool:
    return chi2_contingency is not None and mannwhitneyu is not None


def _require_scipy() -> None:
    if not scipy_available():
        raise InferenceError(SCIPY_MISSING)


# ---------------------------------------------------------------- formatage
def format_p(value) -> str:
    """p-value lisible : « < 1e-300 » en cas de sous-dépassement, jamais 0."""
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if value == 0 or value < P_FLOOR:
        return "< 1e-300"
    return f"{value:.3e}".replace(".", ",")


def _fr(value, digits: int = 2) -> str:
    """Nombre au format français ; séparateur de milliers à partir de 10 000."""
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    grouping = "," if abs(value) >= 10_000 else ""
    if value.is_integer() and digits >= 0:
        text = format(int(value), f"{grouping}d")
    else:
        text = format(value, f"{grouping}.{digits}f")
    return text.replace(",", " ").replace(".", ",")


def fr_fixed(value, digits: int) -> str:
    """Nombre avec exactement ``digits`` décimales (0,2437), au format français."""
    if value is None or pd.isna(value):
        return "—"
    text = format(float(value), f",.{digits}f")
    return text.replace(",", " ").replace(".", ",")


def _yes_no(flag: bool) -> str:
    return "oui" if flag else "non"


# ------------------------------------------------------------------ calculs
def group_rare_categories(series: pd.Series, min_rows: int) -> pd.Series:
    """Regroupe les modalités ayant moins de ``min_rows`` lignes dans « Other »."""
    observed = series.dropna()
    if observed.astype(str).isin([OTHER, OTHER_DISPLAY]).any():
        raise InferenceError(
            f"Une modalité réelle s'appelle déjà « {OTHER} » ou « {OTHER_DISPLAY} » : "
            "le regroupement des modalités rares serait ambigu."
        )
    counts = series.value_counts()
    keep = counts[counts >= min_rows].index
    return series.mask(series.notna() & ~series.isin(keep), OTHER)


def chi_square_report(series: pd.Series, target: pd.Series, min_rows: int) -> dict:
    """Test du chi-deux d'indépendance entre une variable texte et la cible 0/1.

    Les manquants de ``series`` sont exclus (ce n'est pas une modalité). Lève
    ``InferenceError`` si la table n'a pas au moins 2 modalités et les deux
    classes 0 et 1, avant tout appel à ``chi2_contingency``.
    """
    _require_scipy()
    grouped = group_rare_categories(series, min_rows)
    frame = pd.DataFrame({"category": grouped, "target": target}).dropna()
    if frame.empty:
        raise InferenceError("Aucune ligne avec un secteur renseigné dans cette sélection.")
    frame["target"] = frame["target"].astype("int64")
    table = pd.crosstab(frame["category"], frame["target"])
    if set(table.columns) != {0, 1}:
        raise InferenceError("Le chi-deux demande des succès et des échecs avec un secteur renseigné.")
    if table.shape[0] < 2:
        raise InferenceError(
            f"Après regroupement (< {min_rows} lignes → {OTHER_DISPLAY}), il reste moins de deux "
            "secteurs : le chi-deux n'a pas de sens sur cette sélection."
        )
    table = table.loc[:, [0, 1]]
    chi2, p_value, dof, expected = chi2_contingency(table, correction=False)
    n = int(table.to_numpy().sum())
    cramers_v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    residuals = (table.to_numpy() - expected) / np.sqrt(expected)
    residual_series = pd.Series(residuals[:, 1], index=table.index, name="résidu")
    return {
        "min_rows": min_rows,
        "table": table,
        "expected": expected,
        "residuals": residual_series.sort_values(key=np.abs, ascending=False),
        "chi2": float(chi2),
        "p": float(p_value),
        "dof": int(dof),
        "v": float(cramers_v),
        "n": n,
        "categories": int(table.shape[0]),
        "collapsed": int((grouped == OTHER).sum()),
        "min_expected": float(expected.min()),
        "cells_below_5": int((expected < 5).sum()),
        "cells_below_1": int((expected < 1).sum()),
    }


def mann_whitney_report(success_values: pd.Series, failure_values: pd.Series, label: str) -> dict:
    """Test de Mann-Whitney U bilatéral (succès en premier échantillon).

    Lève ``InferenceError`` si un groupe a moins de deux valeurs observées ou si
    toutes les valeurs sont identiques (U et p-value seraient trompeurs).
    """
    _require_scipy()
    success = pd.to_numeric(success_values, errors="coerce").dropna().astype("float64")
    failure = pd.to_numeric(failure_values, errors="coerce").dropna().astype("float64")
    n_success, n_failure = len(success), len(failure)
    if n_success < 2 or n_failure < 2:
        raise InferenceError(
            "Il faut au moins deux valeurs observées chez les succès et chez les échecs "
            f"(ici {n_success} et {n_failure})."
        )
    if pd.concat([success, failure]).nunique() < 2:
        raise InferenceError("Toutes les valeurs observées sont identiques : aucun classement possible.")
    u_statistic, p_value = mannwhitneyu(success, failure, alternative="two-sided", method="asymptotic")
    pairs = n_success * n_failure
    return {
        "test": label,
        "n success": n_success,
        "n failure": n_failure,
        "missing success": int(pd.to_numeric(success_values, errors="coerce").isna().sum()),
        "missing failure": int(pd.to_numeric(failure_values, errors="coerce").isna().sum()),
        "median success": float(success.median()),
        "median failure": float(failure.median()),
        "U statistic": float(u_statistic),
        "p-value": float(p_value),
        "rank-biserial r": float(2 * u_statistic / pairs - 1),
        "P(success > failure)": float(u_statistic / pairs),
    }


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Correction step-down de Holm, résultat dans l'ordre d'entrée.

    Les valeurs manquantes (NaN) sont exclues de la famille et restent NaN :
    ``m`` est le nombre de p-values finies.
    """
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    m = len(finite)
    order = finite[np.argsort(values[finite], kind="stable")]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def split_by_target(scope: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Valeurs de ``column`` chez les succès (cible 1) puis chez les échecs (cible 0)."""
    target = scope["target"]
    return scope.loc[target.eq(1).fillna(False).astype(bool), column], scope.loc[
        target.eq(0).fillna(False).astype(bool), column
    ]


def run_all(scope: pd.DataFrame) -> dict:
    """Calcule les trois tests primaires, la sensibilité et la correction de Holm.

    ``scope`` doit être restreint au périmètre labellisé
    (``app_data.modeling_rows``). Chaque test qui échoue est remplacé par son
    message d'erreur ; Holm porte alors sur les tests disponibles.
    """
    _require_scipy()
    results: dict = {}

    def attempt(key, function, *args):
        try:
            results[key] = function(*args)
        except InferenceError as error:
            results[key] = str(error)

    attempt("chi_primary", chi_square_report, scope["primary_category"], scope["target"], PRIMARY_MIN_ROWS)
    attempt("chi_sensitivity", chi_square_report, scope["primary_category"], scope["target"], SENSITIVITY_MIN_ROWS)
    for column in MW_VARIABLES:
        success, failure = split_by_target(scope, column)
        attempt(f"mw_{column}", mann_whitney_report, success, failure, column)

    family = [
        results["chi_primary"].get("p") if isinstance(results["chi_primary"], dict) else None,
        *(
            results[f"mw_{column}"].get("p-value") if isinstance(results[f"mw_{column}"], dict) else None
            for column in MW_VARIABLES
        ),
    ]
    raw = np.array([np.nan if value is None else value for value in family], dtype=float)
    results["holm_raw"] = raw
    results["holm_adjusted"] = holm_adjust(raw)
    return results


# ------------------------------------------------------- tableaux affichés
CHI_ROWS = (
    "n (lignes utilisées)",
    "catégories après regroupement",
    "lignes regroupées dans « Autres »",
    "statistique du chi-deux",
    "degrés de liberté",
    "p-value",
    "V de Cramér",
    "effectif attendu minimum",
    "cellules avec effectif attendu < 5",
    "cellules avec effectif attendu < 1",
)


def _chi_column(report: dict | str) -> list[str]:
    if not isinstance(report, dict):
        return ["—"] * len(CHI_ROWS)
    return [
        _fr(report["n"]),
        _fr(report["categories"]),
        _fr(report["collapsed"]),
        fr_fixed(report["chi2"], 3),
        _fr(report["dof"]),
        format_p(report["p"]),
        fr_fixed(report["v"], 4),
        fr_fixed(report["min_expected"], 2),
        _fr(report["cells_below_5"]),
        _fr(report["cells_below_1"]),
    ]


def chi_table(primary: dict | str, sensitivity: dict | str) -> pd.DataFrame:
    """Résumé des deux spécifications du chi-deux, côte à côte."""
    return pd.DataFrame(
        {
            f"< {PRIMARY_MIN_ROWS} lignes → Autres (primaire)": _chi_column(primary),
            f"< {SENSITIVITY_MIN_ROWS} lignes → Autres (sensibilité)": _chi_column(sensitivity),
        },
        index=pd.Index(CHI_ROWS, name="Indicateur"),
    )


def residual_table(report: dict, top: int = 10) -> pd.DataFrame:
    """Résidus de Pearson de la colonne succès, triés par valeur absolue."""
    residuals = report["residuals"].head(top)
    table = report["table"]
    names = [OTHER_DISPLAY if name == OTHER else str(name) for name in residuals.index]
    return pd.DataFrame(
        {
            "Secteur": names,
            "Startups": [int(table.loc[name].sum()) for name in residuals.index],
            "Taux de succès (%)": [
                round(100 * table.loc[name, 1] / table.loc[name].sum(), 1) for name in residuals.index
            ],
            "Résidu de Pearson (succès)": [round(float(value), 2) for value in residuals],
        }
    )


MW_COLUMNS = (
    "Test",
    "n succès",
    "n échecs",
    "manquants succès",
    "manquants échecs",
    "médiane succès",
    "médiane échecs",
    "statistique U",
    "p-value",
    "r de rang-bisérial",
    "P(succès > échec)",
)


def mw_table(reports: Sequence[dict | str], labels: Sequence[str]) -> pd.DataFrame:
    """Tableau des tests de Mann-Whitney (une ligne par variable)."""
    rows = []
    for report, name in zip(reports, labels):
        if not isinstance(report, dict):
            rows.append([name, *(["—"] * (len(MW_COLUMNS) - 1))])
            continue
        rows.append(
            [
                name,
                _fr(report["n success"]),
                _fr(report["n failure"]),
                _fr(report["missing success"]),
                _fr(report["missing failure"]),
                _fr(report["median success"], 1),
                _fr(report["median failure"], 1),
                _fr(report["U statistic"], 1),
                format_p(report["p-value"]),
                fr_fixed(report["rank-biserial r"], 4),
                fr_fixed(report["P(success > failure)"], 4),
            ]
        )
    return pd.DataFrame(rows, columns=list(MW_COLUMNS))


def holm_table(raw: Sequence[float], adjusted: Sequence[float], labels: Sequence[str] = FAMILY_LABELS) -> pd.DataFrame:
    """p-values brutes et ajustées (Holm) de la famille des tests primaires."""
    rows = []
    for name, p_raw, p_adj in zip(labels, raw, adjusted):
        available = p_adj is not None and not pd.isna(p_adj)
        rows.append(
            {
                "Test": name,
                "p-value brute": format_p(p_raw),
                "p-value ajustée (Holm)": format_p(p_adj),
                f"significatif au seuil {_fr(ALPHA)}": _yes_no(bool(p_adj < ALPHA)) if available else "—",
            }
        )
    return pd.DataFrame(rows)


def mw_reading(report: dict) -> str:
    """Phrase descriptive (non causale) sur le sens de l'effet d'un Mann-Whitney."""
    r = report["rank-biserial r"]
    share = report["P(success > failure)"]
    phrases = MW_TRENDS.get(report["test"], ("des valeurs plus grandes", "des valeurs plus petites"))
    if abs(r) < 0.05:
        trend = "les succès et les échecs ont des valeurs très proches"
    else:
        trend = "les succès ont tendance à avoir " + (phrases[0] if r > 0 else phrases[1])
    return (
        f"**{MW_LABELS.get(report['test'], report['test'])}** : {trend} "
        f"(médiane {_fr(report['median success'], 1)} contre {_fr(report['median failure'], 1)}). "
        f"Dans {fr_fixed(100 * share, 1)} % des paires succès–échec, c'est le succès qui a la "
        "valeur la plus grande (ex æquo comptés pour moitié)."
    )


MW_TRENDS = {
    "founding_year": ("une année de création plus récente", "une année de création plus ancienne"),
    "funding_total_usd": ("un financement plus élevé", "un financement plus faible"),
}


# ---------------------------------------------- valeurs de référence (§6)
# Ensemble labellisé complet (13 129 startups ; 12 052 avec un secteur),
# notebook 01_eda.ipynb §17. Constantes : jamais recalculées sur les filtres.
REFERENCE_CHI = {
    "primary": {
        "n": 12052,
        "categories": 66,
        "collapsed": 2415,
        "chi2": 715.992,
        "dof": 65,
        "p": 2.215e-110,
        "v": 0.2437,
        "min_expected": 12.99,
        "cells_below_5": 0,
        "cells_below_1": 0,
    },
    "sensitivity": {
        "n": 12052,
        "categories": 27,
        "collapsed": 4261,
        "chi2": 604.159,
        "dof": 26,
        "p": 8.078e-111,
        "v": 0.2239,
        "min_expected": 43.29,
        "cells_below_5": 0,
        "cells_below_1": 0,
    },
}
REFERENCE_MW = {
    "founding_year": {
        "test": "founding_year",
        "n success": 5392,
        "n failure": 4018,
        "missing success": 1576,
        "missing failure": 2143,
        "median success": 2004,
        "median failure": 2009,
        "U statistic": 5795749.5,
        "p-value": 0.0,
        "rank-biserial r": -0.4650,
        "P(success > failure)": 0.2675,
    },
    "funding_total_usd": {
        "test": "funding_total_usd",
        "n success": 6111,
        "n failure": 4851,
        "missing success": 857,
        "missing failure": 1310,
        "median success": 12500000,
        "median failure": 1240000,
        "U statistic": 22697642.5,
        "p-value": 0.0,
        "rank-biserial r": 0.5313,
        "P(success > failure)": 0.7657,
    },
}
REFERENCE_RESIDUALS = (
    ("Biotechnology", 8.17),
    ("Enterprise Software", 5.12),
    (OTHER, -4.83),
    ("Apps", -4.42),
    ("Curated Web", -4.26),
    ("Analytics", 3.77),
    ("Semiconductors", 3.50),
    ("Application Platforms", -3.37),
    ("Software", 3.17),
    ("Security", 3.17),
)
REFERENCE_HOLM_RAW = (2.215e-110, 0.0, 0.0)
REFERENCE_HOLM_ADJUSTED = (2.215e-110, 0.0, 0.0)


def reference_chi_table() -> pd.DataFrame:
    return chi_table(REFERENCE_CHI["primary"], REFERENCE_CHI["sensitivity"])


def reference_mw_table() -> pd.DataFrame:
    return mw_table(
        [REFERENCE_MW[column] for column in MW_VARIABLES], [MW_LABELS[column] for column in MW_VARIABLES]
    )


def reference_holm_table() -> pd.DataFrame:
    return holm_table(REFERENCE_HOLM_RAW, REFERENCE_HOLM_ADJUSTED)
