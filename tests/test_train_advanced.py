"""Tests du module des modeles avances (Random Forest et SVM).

Aucune donnee Kaggle n'est necessaire : les donnees synthetiques viennent de
``tests/conftest.py``. Les tests verifient l'alignement sur le protocole commun
(6 variables, ``min_frequency=20``, preprocesseur de l'equipe) et le
fonctionnement de bout en bout des deux ``Pipeline``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluation  # noqa: E402
import preprocessing  # noqa: E402
import splitting  # noqa: E402
import train_advanced as ta  # noqa: E402
from conftest import make_training_data  # noqa: E402


# ---------------------------------------------------------------------------
# Protocole
# ---------------------------------------------------------------------------


def test_protocol_uses_the_six_team_features() -> None:
    assert ta.FEATURE_COLUMNS == tuple(splitting.INITIAL_FEATURES)
    assert len(ta.FEATURE_COLUMNS) == 6
    assert "founding_month" not in ta.FEATURE_COLUMNS
    assert "founding_month" in ta.features.EDA_ONLY_COLUMNS


def test_min_frequency_matches_the_team_default() -> None:
    assert ta.MIN_FREQUENCY == preprocessing.DEFAULT_MIN_FREQUENCY == 20


def test_check_protocol_accepts_aligned_modules() -> None:
    ta.check_protocol()  # ne doit pas lever


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


def _numeric_transformer(pipeline) -> object:
    """Retourne le sous-pipeline numerique sans exiger un ``fit`` prealable."""
    preprocessor = pipeline.named_steps["preprocessor"]
    return {
        name: transformer for name, transformer, _ in preprocessor.transformers
    }["num"]


def test_random_forest_pipeline_has_no_scaler() -> None:
    pipeline = ta.build_random_forest_pipeline()
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    assert isinstance(pipeline.named_steps["model"], RandomForestClassifier)
    assert "scaler" not in _numeric_transformer(pipeline).named_steps


def test_svm_pipeline_scales_numeric_features() -> None:
    pipeline = ta.build_svm_pipeline()
    assert list(pipeline.named_steps) == ["preprocessor", "model"]
    assert isinstance(pipeline.named_steps["model"], SVC)
    assert "scaler" in _numeric_transformer(pipeline).named_steps


# ---------------------------------------------------------------------------
# Grilles et validation croisee
# ---------------------------------------------------------------------------


def test_grids_use_the_model_step_prefix() -> None:
    forest = ta.get_random_forest_param_grid()
    assert forest
    assert all(key.startswith("model__") for key in forest)

    svm = ta.get_svm_param_grid()
    assert svm
    assert all(
        key.startswith("model__") for grid in svm for key in grid
    )


def test_cross_validation_is_stratified_five_folds() -> None:
    cv = ta.build_cross_validation()
    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True

    with pytest.raises(ValueError):
        ta.build_cross_validation(n_splits=3)


def test_prepare_experiment_returns_two_unfitted_searches() -> None:
    experiment = ta.prepare_experiment(rf_n_jobs=1, svm_n_jobs=1)
    assert set(experiment) == {"random_forest", "svm"}
    for search in experiment.values():
        assert isinstance(search, GridSearchCV)
        assert list(search.estimator.named_steps) == ["preprocessor", "model"]
        with pytest.raises(AttributeError):
            _ = search.best_params_


# ---------------------------------------------------------------------------
# Fonctionnement de bout en bout
# ---------------------------------------------------------------------------


def test_random_forest_search_fits_and_evaluates_on_synthetic_data() -> None:
    X, y = make_training_data(n_rows=400, seed=7)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    search = ta.build_random_forest_search(
        param_grid={"model__n_estimators": [10]},
        cv=cv,
        n_jobs=1,
    )
    search.fit(X, y)

    metrics = evaluation.evaluate_classifier(search.best_estimator_, X, y)
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert metrics["roc_auc"] is not None


def test_svm_search_fits_and_evaluates_on_synthetic_data() -> None:
    X, y = make_training_data(n_rows=400, seed=7)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    search = ta.build_svm_search(
        param_grid=[{"model__kernel": ["linear"], "model__C": [1.0]}],
        cv=cv,
        n_jobs=1,
    )
    search.fit(X, y)

    metrics = evaluation.evaluate_classifier(search.best_estimator_, X, y)
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    # SVC sans probability : ROC-AUC calcule via decision_function.
    assert metrics["roc_auc"] is not None


def test_load_dataset_rejects_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with pytest.raises(FileNotFoundError):
        ta.load_dataset(missing)
