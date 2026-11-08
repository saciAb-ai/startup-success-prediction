"""Tests des indicateurs par année et par pays (données synthétiques)."""

from __future__ import annotations

import pytest

import app_data
import app_insights


@pytest.fixture
def eda_frame(raw_frame):
    return app_data.build_eda_frame(raw_frame)


def test_success_by_year(eda_frame):
    table = app_insights.success_by_year(eda_frame, min_count=1)
    known = app_data.modeling_rows(eda_frame)
    assert table["startups"].sum() == int(known["founding_year"].notna().sum())
    assert (table["succès"] + table["échecs"] == table["startups"]).all()
    assert table["taux_succès"].between(0, 100).all()


def test_success_by_year_hides_small_years(eda_frame):
    table = app_insights.success_by_year(eda_frame, min_count=10_000)
    assert table["taux_succès"].isna().all()


def test_success_by_country(eda_frame):
    table = app_insights.success_by_country(eda_frame, min_count=1)
    known = app_data.modeling_rows(eda_frame)
    assert set(table["country_code"]) == set(known["country_code"].dropna())
    assert table["startups"].is_monotonic_decreasing
    usa = known.loc[known["country_code"].eq("USA").fillna(False).astype(bool)]
    expected = usa["target"].astype("float64").mean() * 100
    assert table.set_index("country_code").loc["USA", "taux_succès"] == pytest.approx(expected)
