"""Tests du contrat de chargement du modèle (pipelines factices, sans Kaggle)."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

import features
import models


def _save(tmp_path, pipeline, metadata=None):
    """Enregistre un pipeline (et ses métadonnées) comme le ferait l'équipe modèles."""
    path = tmp_path / "final_pipeline.joblib"
    joblib.dump(pipeline, path)
    if metadata is not None:
        path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _toy_data(n_rows: int = 40, extra_column: str | None = None):
    """Petit jeu X/y synthétique pour fabriquer des modèles volontairement non conformes."""
    rng = np.random.default_rng(0)
    data = {column: rng.choice(["a", "b", "c"], n_rows) for column in features.CATEGORICAL_FEATURES}
    data.update({column: rng.integers(1, 12, n_rows).astype(float) for column in features.NUMERIC_FEATURES})
    if extra_column:
        data[extra_column] = 1.0
    return pd.DataFrame(data), pd.Series(np.arange(n_rows) % 2)


def test_input_frame_columns_types_and_missing_values():
    X = models.make_input_frame({"country_code": " USA ", "city": "", "founding_year": "2010"})
    assert list(X.columns) == list(features.FEATURE_COLUMNS)
    assert X.loc[0, "country_code"] == "USA"
    assert pd.isna(X.loc[0, "city"]) and pd.isna(X.loc[0, "primary_category"])
    assert X.loc[0, "founding_year"] == 2010.0
    assert "founding_month" not in X.columns or pd.isna(X.loc[0, "founding_month"])
    assert str(X["founding_year"].dtype) == "float64"


def test_input_frame_rejects_unknown_variables():
    with pytest.raises(ValueError):
        models.make_input_frame({"funding_total_usd": 1_000_000})


def test_missing_model_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        models.load_model(tmp_path / "absent.joblib")


def test_probability_model(tmp_path, dummy_pipeline):
    path = _save(tmp_path, dummy_pipeline, {"model_name": "Pipeline de test", "sklearn_version": "0.0.0"})
    loaded = models.load_model(path)
    assert loaded.name == "Pipeline de test"
    assert any("scikit-learn" in note for note in loaded.notes)

    X = models.make_input_frame(
        {"country_code": "USA", "primary_category": "Software", "founding_year": 2010}
    )
    result = models.predict_profile(loaded, X)
    assert result.score_kind == "probability"
    assert 0.0 <= result.score <= 1.0
    assert result.predicted_class == int(dummy_pipeline.predict(X)[0])


def test_decision_function_model(tmp_path, svc_pipeline):
    loaded = models.load_model(_save(tmp_path, svc_pipeline))
    X = models.make_input_frame({"country_code": "FRA", "founding_year": 2005})
    result = models.predict_profile(loaded, X)
    assert result.score_kind == "decision"
    assert result.predicted_class == int(result.score > 0)


def test_rejects_classes_other_than_0_and_1(tmp_path):
    X, y = _toy_data()
    pipeline = Pipeline([("model", DummyClassifier())]).fit(X, y + 1)
    with pytest.raises(models.ModelContractError):
        models.load_model(_save(tmp_path, pipeline))


def test_rejects_unexpected_columns(tmp_path):
    X, y = _toy_data(extra_column="funding_total_usd")
    pipeline = Pipeline([("model", DummyClassifier())]).fit(X, y)
    with pytest.raises(models.ModelContractError):
        models.load_model(_save(tmp_path, pipeline))


def test_rejects_invalid_metadata(tmp_path, dummy_pipeline):
    path = _save(tmp_path, dummy_pipeline)
    path.with_suffix(".json").write_text("{pas du json", encoding="utf-8")
    with pytest.raises(models.ModelContractError):
        models.load_model(path)


def test_eda_only_columns_never_reach_the_model():
    """Les colonnes d'analyse seulement (ex. founding_month) sont refusées en entrée."""
    for column in getattr(features, "EDA_ONLY_COLUMNS", ()):
        with pytest.raises(ValueError):
            models.make_input_frame({column: 1})
