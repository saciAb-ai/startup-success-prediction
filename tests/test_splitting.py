"""Tests du decoupage train/test (``src/splitting.py``).

Contribution CS30. Le protocole exige un partage 80/20 stratifie avec
``random_state=42`` et l'exclusion de toute variable posterieure a la creation.
Les tests synthetiques verifient le comportement ; un test supplementaire
verifie les effectifs canoniques ``10 503 / 2 626`` lorsqu'un CSV est present.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pandas as pd
import pytest

import features
import splitting
import targets
from conftest import make_raw_frame

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "startup_data.csv"


def _prepared_frame(n_rows: int = 400, seed: int = 7) -> pd.DataFrame:
    """Dataset synthetique avec les six variables et une cible binaire."""
    raw = make_raw_frame(n_rows=n_rows, seed=seed)
    return targets.prepare_target(features.build_features(raw, max_year=2026))


# ---------------------------------------------------------------------------
# Protocole par defaut
# ---------------------------------------------------------------------------


def test_default_protocol_is_80_20_with_seed_42() -> None:
    """Les valeurs par defaut sont celles imposees par la methodologie."""
    parameters = inspect.signature(splitting.split_dataset).parameters

    assert parameters["test_size"].default == 0.20
    assert parameters["random_state"].default == 42
    assert tuple(parameters["features"].default) == splitting.INITIAL_FEATURES


def test_protocol_contains_the_six_features_and_nothing_forbidden() -> None:
    """Aucune variable cible ou de financement ne peut entrer dans X."""
    assert len(splitting.INITIAL_FEATURES) == 6
    assert not (set(splitting.INITIAL_FEATURES) & splitting.FORBIDDEN_FEATURES)
    assert {"status", "target"} <= splitting.FORBIDDEN_FEATURES
    assert features.FORBIDDEN_FEATURES <= splitting.FORBIDDEN_FEATURES


# ---------------------------------------------------------------------------
# Tailles, stratification et reproductibilite
# ---------------------------------------------------------------------------


def test_split_preserves_rows_and_uses_the_requested_test_size() -> None:
    """Le decoupage ne perd aucune ligne et respecte la proportion demandee."""
    frame = _prepared_frame()
    X_train, X_test, y_train, y_test = splitting.split_dataset(frame)

    expected_test = math.ceil(len(frame) * 0.20)
    assert len(X_test) == expected_test
    assert len(X_train) == len(frame) - expected_test
    assert len(X_train) + len(X_test) == len(frame)
    assert len(y_train) == len(X_train) and len(y_test) == len(X_test)


def test_split_is_stratified_on_the_target() -> None:
    """La proportion de classe 1 doit rester stable entre train et test."""
    frame = _prepared_frame()
    _, _, y_train, y_test = splitting.split_dataset(frame)

    overall = frame["target"].mean()
    assert y_train.mean() == pytest.approx(overall, abs=0.01)
    assert y_test.mean() == pytest.approx(overall, abs=0.01)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}


def test_split_is_reproducible_with_the_same_seed() -> None:
    """Deux appels avec ``random_state=42`` donnent exactement le meme partage."""
    frame = _prepared_frame()
    first = splitting.split_dataset(frame, random_state=42)
    second = splitting.split_dataset(frame, random_state=42)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_series_equal(first[2], second[2])
    pd.testing.assert_series_equal(first[3], second[3])


def test_different_seed_changes_the_partition() -> None:
    """La graine controle bien le tirage : une autre graine change le partage."""
    frame = _prepared_frame()
    first = splitting.split_dataset(frame, random_state=42)
    second = splitting.split_dataset(frame, random_state=43)

    assert set(first[0].index) != set(second[0].index)


def test_train_and_test_rows_are_disjoint() -> None:
    """Aucune observation ne peut appartenir aux deux jeux a la fois."""
    frame = _prepared_frame()
    X_train, X_test, _, _ = splitting.split_dataset(frame)

    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train.index.union(X_test.index)) == len(frame)


# ---------------------------------------------------------------------------
# Absence de fuite : X ne contient que les six variables, valeurs intactes
# ---------------------------------------------------------------------------


def test_split_returns_only_the_selected_features() -> None:
    """X contient exactement les variables du protocole, dans le meme ordre."""
    frame = _prepared_frame()
    X_train, X_test, _, _ = splitting.split_dataset(frame)

    assert list(X_train.columns) == list(splitting.INITIAL_FEATURES)
    assert list(X_test.columns) == list(splitting.INITIAL_FEATURES)
    assert "status" not in X_train.columns
    assert "target" not in X_train.columns


def test_split_does_not_transform_or_impute_the_test_set() -> None:
    """Le decoupage ne modifie pas les valeurs : le test reste hors pipeline."""
    frame = _prepared_frame()
    X_train, X_test, _, _ = splitting.split_dataset(frame)

    expected = frame.loc[X_test.index, list(splitting.INITIAL_FEATURES)]
    pd.testing.assert_frame_equal(X_test, expected)
    assert X_test.isna().sum().sum() == frame.loc[X_test.index, list(splitting.INITIAL_FEATURES)].isna().sum().sum()


@pytest.mark.parametrize("forbidden", sorted(splitting.FORBIDDEN_FEATURES))
def test_split_rejects_every_forbidden_feature(forbidden: str) -> None:
    """Toute variable posterieure a la creation est refusee, nommement."""
    frame = _prepared_frame()
    frame[forbidden] = 0
    requested = list(splitting.INITIAL_FEATURES) + [forbidden]

    with pytest.raises(ValueError, match="interdites"):
        splitting.split_dataset(frame, features=requested)


def test_split_does_not_modify_the_input_frame() -> None:
    """La fonction ne doit pas alterer le dataset prepare."""
    frame = _prepared_frame()
    original = frame.copy(deep=True)

    splitting.split_dataset(frame)

    pd.testing.assert_frame_equal(frame, original)


# ---------------------------------------------------------------------------
# Validation des entrees
# ---------------------------------------------------------------------------


def test_split_rejects_frame_without_target() -> None:
    """``prepare_target`` doit avoir ete appele avant le decoupage."""
    frame = _prepared_frame().drop(columns="target")

    with pytest.raises(ValueError, match="target"):
        splitting.split_dataset(frame)


@pytest.mark.parametrize("invalid_value", [2, -1, None])
def test_split_rejects_non_binary_or_missing_target(invalid_value) -> None:
    """La cible doit etre strictement binaire et sans valeur manquante."""
    frame = _prepared_frame()
    frame.loc[frame.index[0], "target"] = invalid_value

    with pytest.raises(ValueError, match="target"):
        splitting.split_dataset(frame)


def test_split_rejects_single_class_dataset() -> None:
    """Il faut au moins deux exemples dans chaque classe."""
    frame = _prepared_frame()
    frame["target"] = 1

    with pytest.raises(ValueError, match="classes"):
        splitting.split_dataset(frame)


def test_split_rejects_duplicate_features() -> None:
    """Une variable repetee fausserait la matrice d'entree."""
    frame = _prepared_frame()

    with pytest.raises(ValueError, match="doublons"):
        splitting.split_dataset(frame, features=["city", "city"])


def test_split_rejects_empty_feature_list() -> None:
    """Un modele sans variable d'entree n'a pas de sens."""
    frame = _prepared_frame()

    with pytest.raises(ValueError, match="vide"):
        splitting.split_dataset(frame, features=[])


def test_split_rejects_missing_feature_columns() -> None:
    """Une variable du protocole non produite doit etre signalee."""
    frame = _prepared_frame()

    with pytest.raises(ValueError, match="absentes"):
        splitting.split_dataset(frame, features=["unknown_feature"])


@pytest.mark.parametrize("invalid_size", [0, 1, -0.1, 1.5])
def test_split_rejects_invalid_test_size(invalid_size: float) -> None:
    """``test_size`` doit rester strictement entre 0 et 1."""
    frame = _prepared_frame()

    with pytest.raises(ValueError, match="test_size"):
        splitting.split_dataset(frame, test_size=invalid_size)


def test_split_rejects_non_dataframe_input() -> None:
    """Le contrat d'entree est explicite."""
    with pytest.raises(TypeError):
        splitting.split_dataset([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Effectifs canoniques du dataset de l'equipe
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DATA_PATH.is_file(),
    reason="CSV canonique absent : test reserve au depot complet de l'equipe.",
)
def test_canonical_dataset_split_is_10503_2626_with_seed_42() -> None:
    """Sur ``data/raw/startup_data.csv``, le partage doit etre 10 503 / 2 626."""
    import data_loader

    raw = data_loader.load_raw(DATA_PATH)
    prepared = targets.prepare_target(features.build_features(raw, max_year=2026))
    X_train, X_test, y_train, y_test = splitting.split_dataset(prepared)

    assert len(prepared) == 13_129
    assert (len(X_train), len(X_test)) == (10_503, 2_626)
    assert y_train.value_counts().sort_index().to_dict() == {0: 4_929, 1: 5_574}
    assert y_test.value_counts().sort_index().to_dict() == {0: 1_232, 1: 1_394}
    assert list(X_train.columns) == list(splitting.INITIAL_FEATURES)

    # Reproducibilite sur les donnees reelles.
    second = splitting.split_dataset(prepared, random_state=42)
    pd.testing.assert_frame_equal(X_test, second[1])
    pd.testing.assert_series_equal(y_test, second[3])
