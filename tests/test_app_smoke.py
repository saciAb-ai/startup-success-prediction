"""Tests de fumée de l'application Streamlit (AppTest), sur données synthétiques.

Ils exécutent réellement src/app.py : chaque page doit s'afficher sans
exception, avec et sans modèle final.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "app.py"
PAGE_PREDICTION = "Analyse de profil"


@pytest.fixture
def dataset_csv(tmp_path, raw_frame, monkeypatch):
    """Faux CSV Kaggle, transmis à l'application par STARTUP_DATA_PATH."""
    path = tmp_path / "startups.csv"
    raw_frame.to_csv(path, index=False)
    monkeypatch.setenv("STARTUP_DATA_PATH", str(path))
    return path


def _run_app(page: str | None = None) -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=120)
    app.run()
    if page is not None:
        app.sidebar.radio[0].set_value(page)
        app.run()
    return app


def test_eda_dashboard(dataset_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("STARTUP_MODEL_PATH", str(tmp_path / "absent.joblib"))
    app = _run_app()
    assert not app.exception
    assert any("EDA" in title.value for title in app.title)


def test_profile_page_without_model(dataset_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("STARTUP_MODEL_PATH", str(tmp_path / "absent.joblib"))
    app = _run_app(PAGE_PREDICTION)
    assert not app.exception
    assert any("Aucun modèle final" in warning.value for warning in app.warning)


def test_profile_page_with_model(dataset_csv, tmp_path, monkeypatch, dummy_pipeline):
    model_file = tmp_path / "final_pipeline.joblib"
    joblib.dump(dummy_pipeline, model_file)
    model_file.with_suffix(".json").write_text(json.dumps({"model_name": "Pipeline de test"}), encoding="utf-8")
    monkeypatch.setenv("STARTUP_MODEL_PATH", str(model_file))

    app = _run_app(PAGE_PREDICTION)
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) >= 2
