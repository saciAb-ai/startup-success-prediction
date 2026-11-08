"""Fixtures partagées des tests de l'application (données synthétiques).

Aucune donnée Kaggle n'est nécessaire : les tests fabriquent un faux extrait
avec les mêmes colonnes que le CSV brut.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import features  # noqa: E402
import targets  # noqa: E402

GEOGRAPHY = (
    ("USA", "CA", "SF Bay Area", "San Francisco"),
    ("USA", "NY", "New York City", "New York"),
    ("FRA", None, "Paris", "Paris"),
    ("GBR", "H9", "London", "London"),
    ("DZA", None, "Algiers", "Algiers"),
)
CATEGORIES = ("Software|Apps", "Biotechnology", "|Games|Mobile", "Health Care", None)
STATUSES = ("acquired", "closed", "ipo", "operating")


def make_raw_frame(n_rows: int = 240, seed: int = 0) -> pd.DataFrame:
    """Faux extrait du CSV Kaggle : mêmes colonnes, valeurs synthétiques."""
    rng = np.random.default_rng(seed)
    geography = [GEOGRAPHY[i] for i in rng.integers(0, len(GEOGRAPHY), n_rows)]
    founded = [
        f"{year}-{month:02d}-01"
        for year, month in zip(rng.integers(1995, 2014, n_rows), rng.integers(1, 13, n_rows))
    ]
    founded[0] = None  # date absente
    founded[1] = "1801-01-01"  # date aberrante -> manquante après features.py
    funding = rng.integers(10_000, 50_000_000, n_rows).astype(str).astype(object)
    funding[::7] = "-"  # financement inconnu
    return pd.DataFrame(
        {
            "permalink": [f"/organization/startup-{i}" for i in range(n_rows)],
            "name": [f"Startup {i}" for i in range(n_rows)],
            "homepage_url": [None] * n_rows,
            "category_list": [CATEGORIES[i] for i in rng.integers(0, len(CATEGORIES), n_rows)],
            "funding_total_usd": funding,
            "status": rng.choice(STATUSES, size=n_rows, p=(0.3, 0.3, 0.1, 0.3)),
            "country_code": [geo[0] for geo in geography],
            "state_code": [geo[1] for geo in geography],
            "region": [geo[2] for geo in geography],
            "city": [geo[3] for geo in geography],
            "funding_rounds": rng.integers(1, 6, n_rows),
            "founded_at": founded,
            "first_funding_at": ["2012-01-01"] * n_rows,
            "last_funding_at": ["2013-06-01"] * n_rows,
        }
    )


def make_training_data(n_rows: int = 300, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """X (colonnes FEATURE_COLUMNS) et y, au format attendu par le pipeline final."""
    frame = targets.prepare_target(features.build_features(make_raw_frame(n_rows, seed)))
    X = frame.loc[:, list(features.FEATURE_COLUMNS)].copy()
    for column in features.CATEGORICAL_FEATURES:
        # Le prétraitement commun gère pd.NA ; ce pipeline de test, plus simple, attend np.nan.
        X[column] = X[column].astype("object").where(X[column].notna(), np.nan)
    return X, frame["target"].astype(int)


def make_test_pipeline(estimator):
    """Pipeline FACTICE entraîné sur données synthétiques (tests uniquement)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Inconnu")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(
        [
            ("cat", categorical, list(features.CATEGORICAL_FEATURES)),
            ("num", numeric, list(features.NUMERIC_FEATURES)),
        ]
    )
    X, y = make_training_data()
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)]).fit(X, y)


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture(scope="session")
def dummy_pipeline():
    from sklearn.linear_model import LogisticRegression

    return make_test_pipeline(LogisticRegression(max_iter=1000))


@pytest.fixture(scope="session")
def svc_pipeline():
    from sklearn.svm import SVC

    return make_test_pipeline(SVC(kernel="linear"))
