"""Tests de l'export du modele final (sans donnees Kaggle).

Les donnees synthetiques viennent de ``tests/conftest.py``. Les tests verifient
les metadonnees, l'ecriture du couple ``.joblib`` / ``.json`` et le rechargement
reel par l'API de l'application (``src/models.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import export_final_model as efm  # noqa: E402
import models  # noqa: E402
import train_advanced as ta  # noqa: E402
from conftest import make_training_data  # noqa: E402


DATASET = {
    "raw_rows": 10,
    "kept_rows": 8,
    "train_rows": 6,
    "test_rows": 2,
    "positive_share": 0.5,
}


def _fitted_forest():
    X, y = make_training_data(n_rows=300, seed=4)
    pipeline = ta.build_random_forest_pipeline(n_estimators=10)
    pipeline.fit(X, y)
    return pipeline


def _metadata(model_name: str = "random_forest") -> dict:
    return efm.build_metadata(
        model_name,
        {"model__n_estimators": 200},
        {"f1_macro": 0.7, "confusion_matrix": [[1, 2], [3, 4]]},
        DATASET,
        trained_by="test",
    )


def test_build_metadata_contains_the_contract_keys() -> None:
    metadata = _metadata()

    assert metadata["model_key"] == "random_forest"
    assert metadata["model_name"] == "Random Forest"
    assert metadata["features"] == list(ta.FEATURE_COLUMNS)
    assert metadata["best_params"] == {"n_estimators": 200}
    assert metadata["test_metrics"] == {"f1_macro": 0.7}
    assert metadata["sklearn_version"]
    assert metadata["trained_by"] == "test"
    assert metadata["dataset"]["train_rows"] == 6
    assert metadata["exported_at"]


def test_build_metadata_rejects_an_unknown_model() -> None:
    with pytest.raises(ValueError):
        efm.build_metadata("knn", {}, {}, DATASET)


def test_export_pipeline_writes_both_files_and_reloads(tmp_path: Path) -> None:
    output = tmp_path / "final_pipeline.joblib"
    written = efm.export_pipeline(_fitted_forest(), output, _metadata())

    assert written == output
    assert output.is_file()
    metadata_file = tmp_path / "final_pipeline.json"
    assert metadata_file.is_file()
    assert json.loads(metadata_file.read_text(encoding="utf-8"))["model_name"] == (
        "Random Forest"
    )

    loaded = models.load_model(output)
    assert loaded.name == "Random Forest"
    prediction = models.predict_profile(
        loaded,
        models.make_input_frame({"city": "Paris", "founding_year": 2015}),
    )
    assert prediction.predicted_class in (0, 1)
    assert prediction.score_kind == "probability"


def test_export_pipeline_keeps_the_svm_decision_contract(tmp_path: Path) -> None:
    X, y = make_training_data(n_rows=300, seed=5)
    pipeline = ta.build_svm_pipeline(kernel="linear")
    pipeline.fit(X, y)

    output = efm.export_pipeline(
        pipeline,
        tmp_path / "svm.joblib",
        _metadata("svm"),
    )
    loaded = models.load_model(output)
    prediction = models.predict_profile(
        loaded,
        models.make_input_frame({"city": "Paris", "founding_year": 2015}),
    )

    assert loaded.name == "SVM"
    assert prediction.score_kind == "decision"


def test_export_pipeline_rejects_an_untrained_pipeline(tmp_path: Path) -> None:
    with pytest.raises(models.ModelContractError):
        efm.export_pipeline(
            ta.build_random_forest_pipeline(),
            tmp_path / "untrained.joblib",
            {},
        )


def test_load_recorded_results_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        efm.load_recorded_results(tmp_path / "absent.json")


def test_load_recorded_results_reads_the_models_section(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps({"models": {"svm": {"best_params": {}, "test": {}}}}),
        encoding="utf-8",
    )

    assert set(efm.load_recorded_results(path)) == {"svm"}
