"""Tests de la construction des caracteristiques (``src/features.py``).

Contribution CS30. Le contrat verifie est celui du protocole commun : les six
variables de ``splitting.INITIAL_FEATURES`` sont produites, ``founding_month``
reste une colonne d'EDA, et aucune imputation n'est faite ici. Aucune donnee
Kaggle n'est necessaire.
"""

from __future__ import annotations

import pandas as pd
import pytest

import features
import splitting


def _raw_frame() -> pd.DataFrame:
    """Extrait minimal contenant toutes les colonnes brutes requises."""
    return pd.DataFrame(
        {
            "permalink": ["/a", "/b", "/c", "/d"],
            "status": ["ipo", "closed", "acquired", "operating"],
            "country_code": [" USA ", "", None, "FRA"],
            "state_code": ["CA", None, None, "IDF"],
            "region": ["SF Bay Area", None, None, "Paris"],
            "city": ["San Francisco", None, None, "Paris"],
            "category_list": ["Apps|Games|Mobile", "", None, "|Software|Analytics"],
            "founded_at": ["2010-05-15", "1801-01-01", "not-a-date", "2012-03-01"],
        }
    )


# ---------------------------------------------------------------------------
# Contrat avec splitting.INITIAL_FEATURES
# ---------------------------------------------------------------------------


def test_feature_columns_match_the_shared_protocol() -> None:
    """Les caracteristiques produites doivent egaler le protocole de splitting."""
    assert features.FEATURE_COLUMNS == tuple(splitting.INITIAL_FEATURES)
    assert features.FEATURE_COLUMNS == (
        "country_code",
        "state_code",
        "region",
        "city",
        "primary_category",
        "founding_year",
    )


def test_founding_month_is_computed_but_excluded_from_model_inputs() -> None:
    """``founding_month`` est un artefact de date reserve a l'EDA."""
    assert "founding_month" in features.EDA_ONLY_COLUMNS
    assert "founding_month" not in features.FEATURE_COLUMNS
    assert "founding_month" not in splitting.INITIAL_FEATURES


# ---------------------------------------------------------------------------
# build_features : colonnes produites et immutabilite
# ---------------------------------------------------------------------------


def test_build_features_adds_all_expected_columns() -> None:
    """Toutes les variables du protocole et la colonne EDA sont presentes."""
    result = features.build_features(_raw_frame(), max_year=2026)

    assert set(features.FEATURE_COLUMNS) <= set(result.columns)
    assert set(features.EDA_ONLY_COLUMNS) <= set(result.columns)
    assert len(result) == 4


def test_build_features_does_not_modify_the_input_frame() -> None:
    """La fonction travaille sur une copie du DataFrame brut."""
    frame = _raw_frame()
    original = frame.copy(deep=True)

    features.build_features(frame, max_year=2026)

    pd.testing.assert_frame_equal(frame, original)


def test_build_features_is_deterministic() -> None:
    """Deux appels avec la meme annee de reference donnent le meme resultat."""
    first = features.build_features(_raw_frame(), max_year=2026)
    second = features.build_features(_raw_frame(), max_year=2026)

    pd.testing.assert_frame_equal(first, second)


def test_build_features_keeps_raw_columns_for_the_target_step() -> None:
    """``status`` reste disponible pour ``targets.prepare_target``."""
    result = features.build_features(_raw_frame(), max_year=2026)

    assert "status" in result.columns
    assert "permalink" in result.columns


# ---------------------------------------------------------------------------
# primary_category : premiere categorie non vide
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("Apps|Games|Mobile", "Apps"),
        ("|Software|Analytics", "Software"),
        ("Biotechnology", "Biotechnology"),
        ("  Health Care |Apps", "Health Care"),
        ("||Games", "Games"),
        ("", None),
        ("   ", None),
        ("|||", None),
        (None, None),
    ],
)
def test_extract_primary_category(raw_value, expected) -> None:
    """Le premier jeton non vide est retenu ; une valeur vide reste manquante."""
    result = features.extract_primary_category(pd.Series([raw_value]))

    if expected is None:
        assert pd.isna(result.iloc[0])
    else:
        assert result.iloc[0] == expected


# ---------------------------------------------------------------------------
# split_founded_at : annee/mois et plausibilite
# ---------------------------------------------------------------------------


def test_split_founded_at_masks_implausible_and_unparseable_dates() -> None:
    """Dates avant 1900, futures ou illisibles -> NaN pour l'annee et le mois."""
    dates = pd.Series(
        ["2010-05-15", "1801-01-01", "2914-07-01", "not-a-date", None, "2012-03-01"]
    )

    year, month = features.split_founded_at(dates, max_year=2026)

    assert year.iloc[0] == 2010.0
    assert month.iloc[0] == 5.0
    assert year.iloc[1:5].isna().all()
    assert month.iloc[1:5].isna().all()
    assert year.iloc[5] == 2012.0
    assert month.iloc[5] == 3.0


def test_split_founded_at_uses_the_explicit_reference_year() -> None:
    """``max_year`` rend le controle reproductible d'une annee a l'autre."""
    dates = pd.Series(["2020-01-01"])

    year_ok, _ = features.split_founded_at(dates, max_year=2026)
    year_masked, month_masked = features.split_founded_at(dates, max_year=2019)

    assert year_ok.iloc[0] == 2020.0
    assert pd.isna(year_masked.iloc[0])
    assert pd.isna(month_masked.iloc[0])


def test_split_founded_at_returns_float64_for_imputation() -> None:
    """Le type ``float64`` est requis par ``SimpleImputer`` en aval."""
    year, month = features.split_founded_at(pd.Series(["2010-05-15", None]), max_year=2026)

    assert str(year.dtype) == "float64"
    assert str(month.dtype) == "float64"


# ---------------------------------------------------------------------------
# Nettoyage du texte et absence d'imputation
# ---------------------------------------------------------------------------


def test_build_features_cleans_geographic_text_without_filling() -> None:
    """Espaces retires, chaine vide -> NA : aucune imputation n'est faite ici."""
    result = features.build_features(_raw_frame(), max_year=2026)

    assert result.loc[0, "country_code"] == "USA"
    assert pd.isna(result.loc[1, "country_code"])  # chaine vide
    assert pd.isna(result.loc[2, "country_code"])  # valeur absente
    assert result.loc[3, "country_code"] == "FRA"


def test_build_features_leaves_missing_values_missing() -> None:
    """L'imputation appartient aux pipelines, pas au feature engineering."""
    result = features.build_features(_raw_frame(), max_year=2026)

    assert pd.isna(result.loc[1, "founding_year"])  # 1801
    assert pd.isna(result.loc[2, "founding_year"])  # illisible
    assert pd.isna(result.loc[1, "primary_category"])  # chaine vide
    assert pd.isna(result.loc[2, "primary_category"])  # valeur absente
    assert not result.astype("string").apply(
        lambda column: column.str.contains("Inconnu", regex=False, na=False).any()
    ).any()


def test_build_features_never_creates_target_or_derived_funding_columns() -> None:
    """Aucune colonne cible ni variable de financement n'est fabriquee ici."""
    result = features.build_features(_raw_frame(), max_year=2026)

    assert "target" not in result.columns
    assert not ({"time_to_first_funding_days", "funding_duration_days"} & set(result.columns))


# ---------------------------------------------------------------------------
# Validation des entrees
# ---------------------------------------------------------------------------


def test_build_features_rejects_missing_raw_columns() -> None:
    """Une colonne brute absente doit etre signalee explicitement."""
    frame = _raw_frame().drop(columns="founded_at")

    with pytest.raises(ValueError, match="founded_at"):
        features.build_features(frame, max_year=2026)


def test_build_features_rejects_empty_frame() -> None:
    """Un DataFrame vide ne peut pas produire de caracteristiques."""
    empty = pd.DataFrame({column: pd.Series(dtype="object") for column in features.REQUIRED_RAW_COLUMNS})

    with pytest.raises(ValueError, match="vide"):
        features.build_features(empty, max_year=2026)


def test_build_features_rejects_non_dataframe_input() -> None:
    """Le contrat d'entree est explicite."""
    with pytest.raises(TypeError):
        features.build_features({"category_list": []}, max_year=2026)  # type: ignore[arg-type]
