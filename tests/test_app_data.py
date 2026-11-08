"""Tests de la couche données du dashboard (sans Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import app_data
import features


@pytest.fixture
def eda_frame(raw_frame):
    return app_data.build_eda_frame(raw_frame)


def test_target_and_issue_follow_official_mapping(raw_frame, eda_frame):
    status = raw_frame["status"].to_numpy()
    assert len(eda_frame) == len(raw_frame)  # aucune ligne supprimée
    assert set(features.FEATURE_COLUMNS) <= set(eda_frame.columns)
    assert eda_frame.loc[status == "closed", "target"].eq(0).all()
    assert eda_frame.loc[np.isin(status, ["ipo", "acquired"]), "target"].eq(1).all()
    operating = status == "operating"
    assert eda_frame.loc[operating, "target"].isna().all()
    assert (eda_frame.loc[operating, "issue"] == app_data.LABEL_EXCLUDED).all()


def test_funding_text_becomes_missing(raw_frame, eda_frame):
    dash = (raw_frame["funding_total_usd"] == "-").to_numpy()
    assert eda_frame.loc[dash, "funding_total_usd"].isna().all()
    assert eda_frame.loc[~dash, app_data.FUNDING_LOG_COLUMN].notna().all()


def test_missing_table_counts_and_order():
    frame = pd.DataFrame({"a": [1.0, None, None, 4.0], "b": ["x", "y", None, "z"]})
    table = app_data.missing_table(frame).set_index("variable")
    assert table.loc["a", "manquants"] == 2
    assert table.loc["a", "taux_%"] == pytest.approx(50.0)
    assert table.loc["b", "manquants"] == 1
    assert table.index[0] == "a"


def test_filters(eda_frame):
    modeling = app_data.apply_filters(eda_frame, issues=app_data.MODELING_ISSUES)
    assert set(modeling["issue"]) <= set(app_data.MODELING_ISSUES)
    assert app_data.apply_filters(eda_frame, issues=[]).empty
    usa = app_data.apply_filters(eda_frame, countries=["USA"])
    assert set(usa["country_code"].dropna()) == {"USA"}
    strict = app_data.apply_filters(eda_frame, year_range=(2000, 2005), include_unknown_year=False)
    assert strict["founding_year"].between(2000, 2005).all()
    lenient = app_data.apply_filters(eda_frame, year_range=(2000, 2005), include_unknown_year=True)
    assert len(lenient) == len(strict) + int(eda_frame["founding_year"].isna().sum())


def test_frequency_by_issue(eda_frame):
    table = app_data.frequency_by_issue(eda_frame, "country_code", top_n=2)
    assert table["total"].sum() == len(eda_frame)
    assert table.index[-1] == app_data.OTHER_LABEL
    usa = eda_frame.loc[eda_frame["country_code"].eq("USA").fillna(False).astype(bool)]
    expected = usa["target"].astype("float64").mean() * 100
    assert table.loc["USA", "taux_succès_%"] == pytest.approx(expected, abs=0.06)


def test_gaussian_kde_is_a_density():
    values = np.random.default_rng(0).normal(size=2000)
    grid, density = app_data.gaussian_kde(values)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    assert trapezoid(density, grid) == pytest.approx(1.0, abs=0.02)
    assert app_data.gaussian_kde([3, 3, 3]) is None


def test_correlation_matrix_is_valid(eda_frame):
    columns = ["founding_year", "funding_rounds", "target"]
    corr = app_data.correlation_matrix(eda_frame, columns, method="pearson")
    assert list(corr.columns) == columns
    values = corr.to_numpy(dtype="float64")
    assert np.allclose(values, values.T, equal_nan=True)
    assert np.nanmax(np.abs(values)) <= 1 + 1e-9


def test_cramers_v_bounds():
    x = pd.Series(list("aabbcc") * 20)
    assert app_data.cramers_v(x, x) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.choice(list("abc"), 3000))
    b = pd.Series(rng.choice(list("xyz"), 3000))
    assert app_data.cramers_v(a, b) < 0.1


def test_prediction_context_helpers(eda_frame):
    options = app_data.observed_options(eda_frame, "state_code", {"country_code": "USA"})
    assert set(options) <= {"CA", "NY"}
    n, rate = app_data.similar_profiles(eda_frame, country="USA")
    usa_rows = app_data.modeling_rows(eda_frame)["country_code"].eq("USA").fillna(False)
    assert n == int(usa_rows.sum())
    assert rate is None or 0.0 <= rate <= 1.0
