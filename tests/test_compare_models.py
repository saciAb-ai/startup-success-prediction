"""Tests du tableau comparatif des six modeles (sans donnees Kaggle).

Les fixtures reproduisent le schema des deux fichiers de resultats reels, sur
des valeurs synthetiques. Aucun entrainement, aucune lecture du jeu de test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import compare_models as cm  # noqa: E402


FEATURES = [
    "country_code",
    "state_code",
    "region",
    "city",
    "primary_category",
    "founding_year",
]


def _test_block(value: float) -> dict:
    return {
        "accuracy": value,
        "balanced_accuracy": value,
        "precision_macro": value,
        "recall_macro": value,
        "f1_macro": value,
        "roc_auc": value,
        "precision_fermeture": value,
        "recall_fermeture": value,
        "f1_fermeture": value,
        "confusion_matrix": [[10, 5], [4, 11]],
    }


def _classical_payload() -> dict:
    return {
        "protocol": {
            "data_path": "data/raw/startup_data.csv",
            "features": FEATURES,
            "excluded": ["founding_month"],
            "scoring": "f1_macro",
            "cv_folds": 5,
            "random_state": 42,
        },
        "dataset": {
            "raw_rows": 65403,
            "kept_rows": 13129,
            "train_rows": 10503,
            "test_rows": 2626,
            "positive_share_train": 0.5307,
        },
        "versions": {
            "python": "3.13.9",
            "numpy": "2.1.3",
            "pandas": "2.2.3",
            "scikit-learn": "1.6.1",
            "joblib": "1.4.2",
        },
        "models": {
            "zero_r": {"best_params": {}, "test": _test_block(0.34)},
            "knn": {"best_params": {"n_neighbors": 201}, "cv_score": 0.68,
                    "cv_std": 0.01, "test": _test_block(0.68)},
            "decision_tree": {"best_params": {"max_depth": 20}, "cv_score": 0.69,
                              "cv_std": 0.01, "test": _test_block(0.69)},
            "naive_bayes": {"best_params": {"var_smoothing": 1e-05},
                            "cv_score": 0.65, "cv_std": 0.01,
                            "test": _test_block(0.64)},
        },
    }


def _advanced_payload(advanced_versions: dict | None = None) -> dict:
    payload = {
        "protocol": {
            "data_path": "data\\raw\\startup_data.csv",
            "features": FEATURES,
            "excluded": ["founding_month"],
            "scoring": "f1_macro",
            "cv_folds": 5,
            "random_state": 42,
        },
        "dataset": {
            "raw_rows": 65403,
            "kept_rows": 13129,
            "train_rows": 10503,
            "test_rows": 2626,
            "positive_share": 0.5307,
        },
        "models": {
            "random_forest": {"best_params": {"model__n_estimators": 200},
                              "best_cv_score": 0.70,
                              "test": _test_block(0.70)},
            "svm": {"best_params": {"model__C": 1.0}, "best_cv_score": 0.70,
                    "test": _test_block(0.70)},
        },
    }
    if advanced_versions is not None:
        payload["versions"] = advanced_versions
    return payload


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_source_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cm.load_source(tmp_path / "absent.json")


def test_load_source_rejects_a_payload_without_models(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.json", {"models": {}})
    with pytest.raises(ValueError):
        cm.load_source(path)


def test_build_comparison_contains_six_models_in_protocol_order(tmp_path: Path) -> None:
    classical = _write(tmp_path / "classical.json", _classical_payload())
    advanced = _write(tmp_path / "advanced.json", _advanced_payload())

    comparison = cm.build_comparison(classical, advanced)

    assert list(comparison["models"]) == list(cm.MODEL_ORDER)
    assert len(comparison["models"]) == 6
    assert comparison["comparison_status"] == cm.COMPARISON_STATUS
    assert comparison["validation_required"] == cm.VALIDATION_REQUIRED
    assert comparison["protocol_ok"] is True


def test_protocol_checks_flags_a_dataset_mismatch(tmp_path: Path) -> None:
    classical_payload = _classical_payload()
    classical_payload["dataset"]["kept_rows"] = 13334
    classical = _write(tmp_path / "classical.json", classical_payload)
    advanced = _write(tmp_path / "advanced.json", _advanced_payload())

    comparison = cm.build_comparison(classical, advanced)

    assert comparison["protocol_ok"] is False
    failed = [check["name"] for check in comparison["protocol_checks"] if not check["ok"]]
    assert "effectif_kept_rows" in failed


def test_protocol_checks_flags_a_scoring_mismatch(tmp_path: Path) -> None:
    classical_payload = _classical_payload()
    classical_payload["protocol"]["scoring"] = "balanced_accuracy"
    classical = _write(tmp_path / "classical.json", classical_payload)
    advanced = _write(tmp_path / "advanced.json", _advanced_payload())

    comparison = cm.build_comparison(classical, advanced)

    assert comparison["protocol_ok"] is False
    failed = [check["name"] for check in comparison["protocol_checks"] if not check["ok"]]
    assert "critere_cv" in failed


def test_versions_are_not_comparable_without_advanced_versions(tmp_path: Path) -> None:
    classical = _write(tmp_path / "classical.json", _classical_payload())
    advanced = _write(tmp_path / "advanced.json", _advanced_payload())

    comparison = cm.build_comparison(classical, advanced)

    assert comparison["versions"]["versions_comparable"] is False
    assert comparison["versions"]["classical"]["scikit-learn"] == "1.6.1"


def test_format_markdown_states_it_is_not_validated(tmp_path: Path) -> None:
    classical = _write(tmp_path / "classical.json", _classical_payload())
    advanced = _write(tmp_path / "advanced.json", _advanced_payload())
    comparison = cm.build_comparison(classical, advanced)

    markdown = cm.format_markdown(comparison)

    assert "comparaison informative, non validee" in markdown
    assert "Le modele final n'est pas designe" in markdown
    for display_name in cm.MODEL_NAMES.values():
        assert display_name in markdown


def test_write_reports_creates_json_and_markdown(tmp_path: Path) -> None:
    classical = _write(tmp_path / "classical.json", _classical_payload())
    advanced = _write(tmp_path / "advanced.json", _advanced_payload())
    comparison = cm.build_comparison(classical, advanced)

    json_path, md_path = cm.write_reports(
        comparison, tmp_path / "out" / "comparison.json", tmp_path / "out" / "comparison.md"
    )

    assert json_path.is_file()
    assert md_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["comparison_status"] == cm.COMPARISON_STATUS
    assert len(payload["models"]) == 6


def test_collect_models_rejects_a_missing_model() -> None:
    classical = _classical_payload()
    del classical["models"]["naive_bayes"]
    advanced = _advanced_payload()

    with pytest.raises(ValueError):
        cm.collect_models([classical, advanced])


def test_collect_models_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        cm.collect_models([_classical_payload(), _classical_payload()])
