"""Indicateurs complémentaires du dashboard : taux de succès par année et par pays.

Fonctions pures (sans Streamlit), testées dans tests/test_app_insights.py.
Elles ne portent que sur les startups dont l'issue est connue (cible 0 ou 1).
"""

from __future__ import annotations

import pandas as pd

import app_data

YEAR_COLUMNS = ["founding_year", "startups", "succès", "échecs", "taux_succès"]
COUNTRY_COLUMNS = ["country_code", "startups", "taux_succès"]


def success_by_year(frame: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Effectifs et taux de succès (%) par année de création.

    Le taux vaut NaN pour les années de moins de ``min_count`` startups : un
    pourcentage calculé sur quelques startups n'a pas de sens.
    """
    known = app_data.modeling_rows(frame)
    known = known.loc[known["founding_year"].notna()]
    if known.empty:
        return pd.DataFrame(columns=YEAR_COLUMNS)
    year = known["founding_year"].astype(int).rename("founding_year")
    grouped = known["target"].astype("float64").groupby(year)
    table = pd.DataFrame({"startups": grouped.size(), "succès": grouped.sum().astype(int)})
    table["échecs"] = table["startups"] - table["succès"]
    table["taux_succès"] = (grouped.mean() * 100).where(table["startups"] >= min_count)
    return table.reset_index().loc[:, YEAR_COLUMNS]


def success_by_country(frame: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Effectif et taux de succès (%) par pays (code ISO à 3 lettres).

    Seuls les pays d'au moins ``min_count`` startups sont gardés, du plus au
    moins représenté.
    """
    known = app_data.modeling_rows(frame)
    known = known.loc[known["country_code"].notna()]
    if known.empty:
        return pd.DataFrame(columns=COUNTRY_COLUMNS)
    country = known["country_code"].astype(str).rename("country_code")
    grouped = known["target"].astype("float64").groupby(country)
    table = pd.DataFrame({"startups": grouped.size(), "taux_succès": grouped.mean() * 100})
    table = table.loc[table["startups"] >= min_count]
    return table.sort_values("startups", ascending=False).reset_index().loc[:, COUNTRY_COLUMNS]
