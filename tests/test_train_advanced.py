"""Tests du module des modeles avances (Random Forest et SVM).

Aucune donnee Kaggle n'est necessaire : les donnees synthetiques viennent de
``tests/conftest.py``. Les tests verifient l'alignement sur le protocole commun
(6 variables, ``min_frequency=20``, preprocesseur de l'equipe) et le
fonctionnement de bout en bout des deux ``Pipeline``.
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Analyses complementaires : importance des variables
# ---------------------------------------------------------------------------


def test_build_pipeline_from_params_strips_the_model_prefix() -> None:
    pipeline = ta.build_pipeline_from_params(
        "random_forest", {"model__n_estimators": 50, "model__max_depth": 7}
    )
    model = pipeline.named_steps["model"]
    assert model.n_estimators == 50
    assert model.max_depth == 7
    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_build_pipeline_from_params_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        ta.build_pipeline_from_params("knn", {})


def test_load_best_params_returns_empty_when_file_is_absent(tmp_path: Path) -> None:
    assert ta.load_best_params(tmp_path / "absent.json") == {}


def test_load_best_params_reads_the_results_file(tmp_path: Path) -> None:
    payload = {
        "models": {
            "random_forest": {"best_params": {"model__n_estimators": 200}},
            "svm": {"best_params": {"model__C": 1.0}},
        }
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    params = ta.load_best_params(path)
    assert params["random_forest"] == {"model__n_estimators": 200}
    assert params["svm"] == {"model__C": 1.0}


def test_source_feature_maps_one_hot_columns_back_to_the_six_variables() -> None:
    assert ta._source_feature("num__founding_year") == "founding_year"
    assert ta._source_feature("cat__country_code_USA") == "country_code"
    assert ta._source_feature("cat__state_code_Inconnu") == "state_code"
    assert ta._source_feature("cat__primary_category_Software_Apps") == (
        "primary_category"
    )


def _fit_small_forest() -> "ta.Pipeline":
    X, y = make_training_data(n_rows=400, seed=11)
    pipeline = ta.build_random_forest_pipeline(n_estimators=10)
    pipeline.fit(X, y)
    return pipeline


def test_extract_feature_importances_groups_back_to_six_variables() -> None:
    pipeline = _fit_small_forest()
    importance = ta.extract_feature_importances(pipeline, top_n=5)

    assert importance["n_columns"] == len(
        preprocessing.feature_names_out(pipeline.named_steps["preprocessor"])
    )
    assert len(importance["top_columns"]) == 5
    assert importance["impurity_total"] == pytest.approx(1.0)

    grouped = {
        row["source_feature"]: row["importance"]
        for row in importance["grouped_by_source"]
    }
    assert set(grouped) == set(ta.FEATURE_COLUMNS)
    assert sum(grouped.values()) == pytest.approx(1.0, abs=1e-4)
    assert importance["top_columns"][0]["importance"] >= (
        importance["top_columns"][-1]["importance"]
    )


def test_extract_feature_importances_rejects_a_model_without_importances() -> None:
    X, y = make_training_data(n_rows=300, seed=3)
    pipeline = ta.build_svm_pipeline(kernel="linear")
    pipeline.fit(X, y)
    with pytest.raises(TypeError):
        ta.extract_feature_importances(pipeline)


# ---------------------------------------------------------------------------
# Analyses complementaires : robustesse
# ---------------------------------------------------------------------------


def test_compute_robustness_returns_one_row_per_seed() -> None:
    X, y = make_training_data(n_rows=400, seed=5)
    result = ta.compute_robustness(
        "random_forest",
        X,
        y,
        params={"model__n_estimators": 10},
        seeds=(0, 1),
        n_jobs=1,
    )

    assert result["model"] == "random_forest"
    assert [row["seed"] for row in result["seeds"]] == [0, 1]
    for row in result["seeds"]:
        assert len(row["folds"]) == ta.CV_FOLDS
        assert row["min"] <= row["mean"] <= row["max"]
    assert result["overall_min"] <= result["overall_mean"] <= result["overall_max"]
    assert result["overall_min"] >= 0.0
    assert result["overall_max"] <= 1.0


def test_compute_robustness_rejects_an_empty_seed_list() -> None:
    X, y = make_training_data(n_rows=300, seed=2)
    with pytest.raises(ValueError):
        ta.compute_robustness("random_forest", X, y, seeds=())


# ---------------------------------------------------------------------------
# Restitution
# ---------------------------------------------------------------------------


def test_markdown_formatters_expose_the_main_sections() -> None:
    base = {
        "protocol": {
            "features": list(ta.FEATURE_COLUMNS),
            "excluded": list(ta.features.EDA_ONLY_COLUMNS),
            "min_frequency": ta.MIN_FREQUENCY,
            "scoring": ta.SCORING,
            "cv_folds": ta.CV_FOLDS,
            "seeds": [0, 1],
        },
        "dataset": {
            "raw_rows": 10,
            "kept_rows": 8,
            "train_rows": 6,
            "test_rows": 2,
            "positive_share": 0.5,
        },
        "importance": {
            "n_columns": 3,
            "impurity_total": 1.0,
            "grouped_by_source": [
                {"source_feature": "city", "importance": 1.0}
            ],
            "top_columns": [
                {
                    "column": "cat__city_Paris",
                    "source_feature": "city",
                    "importance": 1.0,
                }
            ],
        },
    }
    importance_md = ta.format_importance_markdown(base)
    assert "Importance des variables" in importance_md
    assert "cat__city_Paris" in importance_md

    robust = {
        "protocol": base["protocol"],
        "dataset": base["dataset"],
        "models": {
            "random_forest": {
                "seeds": [
                    {
                        "seed": 0,
                        "mean": 0.7,
                        "std": 0.01,
                        "min": 0.68,
                        "max": 0.72,
                        "folds": [0.68, 0.72],
                    }
                ],
                "overall_mean": 0.7,
                "overall_std": 0.0,
                "overall_min": 0.7,
                "overall_max": 0.7,
                "fit_seconds": 1.2,
            }
        },
    }
    robustness_md = ta.format_robustness_markdown(robust)
    assert "Robustesse" in robustness_md
    assert "random_forest" in robustness_md
