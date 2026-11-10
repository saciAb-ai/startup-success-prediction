"""Tests de la construction de la cible (``src/targets.py``).

Contribution CS30. Ces tests verifient le contrat de la section 3 de
``docs/methodology.md`` : ``ipo`` et ``acquired`` -> classe 1, ``closed`` ->
classe 0, ``operating`` exclu. Aucune donnee Kaggle n'est necessaire.
"""

from __future__ import annotations

import pandas as pd
import pytest

import features
import targets


def _status_frame() -> pd.DataFrame:
    """Petit jeu brut couvrant les quatre statuts, la casse et les espaces."""
    return pd.DataFrame(
        {
            "permalink": ["/a", "/b", "/c", "/d", "/e"],
            "status": [" IPO ", "Acquired", "Closed", "Operating", "closed"],
            "country_code": ["USA", "FRA", "DZA", "USA", "GBR"],
        }
    )


# ---------------------------------------------------------------------------
# Table de correspondance officielle
# ---------------------------------------------------------------------------


def test_target_mapping_is_the_official_one() -> None:
    """La table de la methodologie ne doit pas deriver silencieusement."""
    assert targets.TARGET_MAPPING == {"ipo": 1, "acquired": 1, "closed": 0}


def test_operating_is_known_but_not_mapped() -> None:
    """``operating`` est un statut valide, mais il ne recoit aucune classe."""
    assert "operating" in targets.KNOWN_STATUSES
    assert "operating" not in targets.TARGET_MAPPING


def test_status_and_target_are_not_model_inputs() -> None:
    """La cible ne doit jamais se retrouver parmi les variables d'entree."""
    assert "status" not in features.FEATURE_COLUMNS
    assert "target" not in features.FEATURE_COLUMNS
    assert not (set(features.FEATURE_COLUMNS) & features.FORBIDDEN_FEATURES)


# ---------------------------------------------------------------------------
# prepare_target : filtrage, normalisation et cible binaire
# ---------------------------------------------------------------------------


def test_prepare_target_maps_success_failure_and_excludes_operating() -> None:
    """IPO et acquisition -> 1, fermeture -> 0, ``operating`` -> exclu."""
    result = targets.prepare_target(_status_frame())

    assert len(result) == 4  # operating exclu
    assert result["status"].tolist() == ["ipo", "acquired", "closed", "closed"]
    assert result["target"].tolist() == [1, 1, 0, 0]
    assert "operating" not in result["status"].tolist()


def test_prepare_target_normalizes_case_and_surrounding_spaces() -> None:
    """``" IPO "`` et ``"Acquired"`` doivent etre reconnus sans retraitement."""
    result = targets.prepare_target(_status_frame())

    assert result.loc[0, "status"] == "ipo"
    assert result.loc[1, "status"] == "acquired"
    assert result.loc[0, "target"] == 1


def test_prepare_target_returns_int8_binary_target() -> None:
    """``target`` est un entier compact, sans valeur manquante."""
    result = targets.prepare_target(_status_frame())

    assert str(result["target"].dtype) == "int8"
    assert result["target"].notna().all()
    assert set(result["target"].unique()) == {0, 1}


def test_prepare_target_preserves_rows_order_and_other_columns() -> None:
    """Le filtrage conserve l'ordre d'origine et les colonnes metier."""
    result = targets.prepare_target(_status_frame())

    assert result["permalink"].tolist() == ["/a", "/b", "/c", "/e"]
    assert result["country_code"].tolist() == ["USA", "FRA", "DZA", "GBR"]


def test_prepare_target_does_not_modify_the_input_frame() -> None:
    """La fonction travaille sur une copie : le brut reste intact."""
    frame = _status_frame()
    original = frame.copy(deep=True)

    targets.prepare_target(frame)

    pd.testing.assert_frame_equal(frame, original)


def test_prepare_target_rejects_unknown_status() -> None:
    """Un statut hors du vocabulaire officiel doit echouer explicitement."""
    frame = _status_frame()
    frame.loc[1, "status"] = "unknown-status"

    with pytest.raises(ValueError, match="Statuts inconnus"):
        targets.prepare_target(frame)


def test_prepare_target_rejects_missing_status_value() -> None:
    """Un statut manquant ne doit pas etre classe par defaut."""
    frame = _status_frame()
    frame.loc[2, "status"] = None

    with pytest.raises(ValueError, match="Statuts inconnus"):
        targets.prepare_target(frame)


def test_prepare_target_rejects_frame_without_status_column() -> None:
    """La colonne ``status`` est obligatoire."""
    frame = _status_frame().drop(columns="status")

    with pytest.raises(ValueError, match="status"):
        targets.prepare_target(frame)


def test_prepare_target_rejects_empty_frame() -> None:
    """Un DataFrame vide n'a pas de cible exploitable."""
    empty = pd.DataFrame({"status": pd.Series(dtype="object")})

    with pytest.raises(ValueError, match="vide"):
        targets.prepare_target(empty)


def test_prepare_target_rejects_frame_with_only_operating_rows() -> None:
    """Sans aucune issue connue, il n'existe pas de classe a predire."""
    frame = pd.DataFrame({"status": ["operating", "Operating"]})

    with pytest.raises(ValueError, match="Aucune startup"):
        targets.prepare_target(frame)


def test_prepare_target_rejects_non_dataframe_input() -> None:
    """Le contrat d'entree est explicite."""
    with pytest.raises(TypeError):
        targets.prepare_target(["ipo", "closed"])  # type: ignore[arg-type]
