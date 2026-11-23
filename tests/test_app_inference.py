"""Tests de l'onglet « Tests statistiques » (src/app_inference.py).

- formules vérifiées sur de petites tables calculables à la main ;
- cas limites des sélections filtrées (aucun test trompeur, aucune exception) ;
- reproduction des valeurs de référence de docs/handoff_eda.md (§6, annexe C)
  sur le vrai CSV de data/raw, s'il est présent.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2 as chi2_distribution

import app_data
import app_inference as ai
import data_loader


# ----------------------------------------------------------------- outils
def _scope(categories, targets_, years=None, funding=None) -> pd.DataFrame:
    n = len(targets_)
    return pd.DataFrame(
        {
            "primary_category": pd.Series(categories, dtype="string"),
            "target": pd.Series(targets_, dtype="Int8"),
            "founding_year": pd.Series(years if years is not None else [2000.0] * n, dtype="float64"),
            "funding_total_usd": pd.Series(funding if funding is not None else [1.0] * n, dtype="float64"),
        }
    )


# ------------------------------------------------------------- formatage
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "< 1e-300"),
        (1e-310, "< 1e-300"),
        (None, "—"),
        (float("nan"), "—"),
        (2.2149809e-110, "2,215e-110"),
        (0.0123, "1,230e-02"),
    ],
)
def test_format_p_never_shows_zero(value, expected):
    assert ai.format_p(value) == expected


# ------------------------------------------------------------------- Holm
def test_holm_adjust_known_values():
    # tri : 0,01×3 = 0,03 ; 0,03×2 = 0,06 ; 0,04×1 = 0,04 -> max courant 0,06
    adjusted = ai.holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_adjust_caps_at_one_and_keeps_order():
    adjusted = ai.holm_adjust([0.6, 0.5, 0.9])
    np.testing.assert_allclose(adjusted, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(ai.holm_adjust([0.2, 0.001]), [0.2, 0.002])


def test_holm_adjust_excludes_nan_from_family():
    adjusted = ai.holm_adjust([0.01, np.nan, 0.04])
    assert math.isnan(adjusted[1])
    np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.04])


def test_holm_adjust_underflow_stays_zero():
    np.testing.assert_allclose(ai.holm_adjust([2.215e-110, 0.0, 0.0]), [2.215e-110, 0.0, 0.0])


# ------------------------------------------------------------ regroupement
def test_group_rare_categories_threshold_and_missing():
    series = pd.Series(["A"] * 30 + ["B"] * 29 + ["C"] * 100 + [None] * 5, dtype="string")
    grouped = ai.group_rare_categories(series, 30)
    assert grouped.value_counts().to_dict() == {"C": 100, "A": 30, "Other": 29}
    assert grouped.isna().sum() == 5  # un manquant n'est pas une modalité
    grouped_100 = ai.group_rare_categories(series, 100)
    assert grouped_100.value_counts().to_dict() == {"C": 100, "Other": 59}


def test_group_rare_categories_refuses_real_other_category():
    with pytest.raises(ai.InferenceError):
        ai.group_rare_categories(pd.Series(["Other", "A"], dtype="string"), 1)


# --------------------------------------------------------------- chi-deux
def test_chi_square_report_hand_computed_table():
    # A : 30 succès / 10 échecs ; B : 10 succès / 30 échecs ; attendus = 20 partout
    categories = ["A"] * 40 + ["B"] * 40
    target = [1] * 30 + [0] * 10 + [1] * 10 + [0] * 30
    report = ai.chi_square_report(pd.Series(categories), pd.Series(target), min_rows=1)
    assert report["n"] == 80
    assert report["dof"] == 1
    assert report["chi2"] == pytest.approx(20.0)  # 4 × 10² / 20
    assert report["p"] == pytest.approx(chi2_distribution.sf(20.0, 1))
    assert report["v"] == pytest.approx(0.5)  # sqrt(20 / 80)
    assert report["min_expected"] == pytest.approx(20.0)
    assert report["residuals"]["A"] == pytest.approx(10 / math.sqrt(20))
    assert report["residuals"]["B"] == pytest.approx(-10 / math.sqrt(20))
    assert report["cells_below_5"] == 0 and report["collapsed"] == 0


def test_chi_square_report_excludes_missing_and_counts_other():
    categories = ["A"] * 40 + ["B"] * 40 + ["R1", "R2"] + [None] * 3
    target = [1, 0] * 20 + [0, 1] * 20 + [1, 0] + [1, 0, 1]
    report = ai.chi_square_report(pd.Series(categories, dtype="string"), pd.Series(target), min_rows=30)
    assert report["n"] == 82  # les 3 secteurs manquants sont exclus
    assert report["collapsed"] == 2
    assert "Other" in report["table"].index
    assert report["cells_below_5"] == 2  # « Other » n'a que 2 lignes -> avertissement


@pytest.mark.parametrize(
    ("categories", "target"),
    [
        (["A"] * 10, [1, 0] * 5),  # une seule modalité (table 1 × 2)
        (["A"] * 5 + ["B"] * 5, [1] * 10),  # une seule classe
        ([None] * 6, [1, 0] * 3),  # aucun secteur renseigné
        (["A"] * 5 + ["B"] * 5, [1, 0] * 5),  # tout regroupé dans « Other » (seuil 30)
    ],
)
def test_chi_square_report_degenerate_tables(categories, target):
    with pytest.raises(ai.InferenceError):
        ai.chi_square_report(pd.Series(categories, dtype="string"), pd.Series(target), min_rows=30)


# ----------------------------------------------------------- Mann-Whitney
def test_mann_whitney_report_complete_separation():
    success = pd.Series([3.0, 4.0, 5.0, np.nan])
    failure = pd.Series([1.0, 2.0])
    report = ai.mann_whitney_report(success, failure, "x")
    assert report["U statistic"] == 6.0  # 3 × 2 paires, toutes gagnées par les succès
    assert report["P(success > failure)"] == pytest.approx(1.0)
    assert report["rank-biserial r"] == pytest.approx(1.0)
    assert (report["n success"], report["n failure"]) == (3, 2)
    assert (report["missing success"], report["missing failure"]) == (1, 0)
    assert (report["median success"], report["median failure"]) == (4.0, 1.5)


def test_mann_whitney_report_ties_count_half():
    report = ai.mann_whitney_report(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]), "x")
    assert report["P(success > failure)"] == pytest.approx(0.5)
    assert report["rank-biserial r"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("success", "failure"),
    [
        ([], [1.0, 2.0]),  # groupe vide
        ([1.0], [1.0, 2.0]),  # une seule observation
        ([np.nan, np.nan], [1.0, 2.0]),  # que des manquants
        ([3.0, 3.0], [3.0, 3.0, 3.0]),  # valeurs toutes identiques
    ],
)
def test_mann_whitney_report_degenerate_groups(success, failure):
    with pytest.raises(ai.InferenceError):
        ai.mann_whitney_report(pd.Series(success, dtype="float64"), pd.Series(failure, dtype="float64"), "x")


# ---------------------------------------------------------------- run_all
def test_run_all_degenerate_selection_returns_messages():
    scope = _scope(["A"] * 6, [1, 1, 1, 0, 0, 0], years=[2000.0] * 6, funding=[np.nan] * 6)
    results = ai.run_all(scope)
    for key in ("chi_primary", "chi_sensitivity", "mw_founding_year", "mw_funding_total_usd"):
        assert isinstance(results[key], str)
    assert np.isnan(results["holm_adjusted"]).all()
    table = ai.holm_table(results["holm_raw"], results["holm_adjusted"])
    assert set(table["p-value ajustée (Holm)"]) == {"—"}


def test_run_all_partial_family_adjusts_available_tests():
    rng = np.random.default_rng(0)
    targets_ = [1] * 50 + [0] * 50
    scope = _scope(["A"] * 100, targets_, years=list(rng.integers(1990, 2014, 100)), funding=[np.nan] * 100)
    results = ai.run_all(scope)
    assert isinstance(results["chi_primary"], str)  # un seul secteur
    assert isinstance(results["mw_founding_year"], dict)
    assert results["holm_adjusted"][1] == pytest.approx(min(1.0, results["holm_raw"][1]))


def test_tables_use_french_labels_and_formats():
    scope = _scope(["A"] * 40 + ["B"] * 40, [1] * 30 + [0] * 10 + [1] * 10 + [0] * 30,
                   years=list(range(1990, 2070)), funding=[float(v) for v in range(80)])
    results = ai.run_all(scope)
    reports = [results[f"mw_{column}"] for column in ai.MW_VARIABLES]
    table = ai.mw_table(reports, list(ai.MW_LABELS.values()))
    assert list(table.columns) == list(ai.MW_COLUMNS)
    assert "P(succès > échec)" in table.columns
    holm = ai.holm_table(results["holm_raw"], results["holm_adjusted"])
    assert "significatif au seuil 0,05" in holm.columns
    assert "0" not in set(holm["p-value brute"])
    assert "statistique du chi-deux" in ai.chi_table(results["chi_primary"], results["chi_sensitivity"]).index


def test_reference_tables_match_handoff():
    chi = ai.reference_chi_table()
    assert chi.loc["p-value"].tolist() == ["2,215e-110", "8,078e-111"]
    assert chi.loc["V de Cramér"].tolist() == ["0,2437", "0,2239"]
    mw = ai.reference_mw_table()
    assert mw["p-value"].tolist() == ["< 1e-300", "< 1e-300"]
    assert mw["statistique U"].tolist() == ["5 795 749,5", "22 697 642,5"]
    assert ai.reference_holm_table()["significatif au seuil 0,05"].tolist() == ["oui"] * 3


# --------------------------------------------------- max_year (§2.5 passation)
def test_build_eda_frame_uses_fixed_max_year(raw_frame):
    raw = raw_frame.copy()
    raw.loc[2, "founded_at"] = "2030-05-01"
    assert app_data.EDA_MAX_YEAR == 2026
    assert pd.isna(app_data.build_eda_frame(raw).loc[2, "founding_year"])
    assert app_data.build_eda_frame(raw, max_year=2035).loc[2, "founding_year"] == 2030


# ------------------------------------- reproduction sur le vrai jeu de données
@pytest.fixture(scope="module")
def real_results():
    path = data_loader.find_raw_file(app_data.PROJECT_ROOT)
    if path is None:
        pytest.skip("CSV réel absent de data/raw")
    scope = app_data.modeling_rows(app_data.build_eda_frame(data_loader.load_raw(path)))
    if len(scope) != 13129:
        pytest.skip(f"CSV différent de celui de la passation ({len(scope)} lignes labellisées)")
    return ai.run_all(scope)


@pytest.mark.parametrize("key", ["primary", "sensitivity"])
def test_real_chi_square_matches_reference(real_results, key):
    report = real_results[f"chi_{key}"]
    reference = ai.REFERENCE_CHI[key]
    for name in ("n", "categories", "collapsed", "dof", "cells_below_5", "cells_below_1"):
        assert report[name] == reference[name], name
    assert round(report["chi2"], 3) == reference["chi2"]
    assert float(f"{report['p']:.3e}") == reference["p"]
    assert round(report["v"], 4) == reference["v"]
    assert round(report["min_expected"], 2) == reference["min_expected"]


def test_real_residuals_match_reference(real_results):
    residuals = real_results["chi_primary"]["residuals"].head(10)
    assert [(name, round(float(value), 2)) for name, value in residuals.items()] == list(ai.REFERENCE_RESIDUALS)


@pytest.mark.parametrize("column", ai.MW_VARIABLES)
def test_real_mann_whitney_matches_reference(real_results, column):
    report = real_results[f"mw_{column}"]
    reference = ai.REFERENCE_MW[column]
    for name in ("n success", "n failure", "missing success", "missing failure", "median success", "median failure"):
        assert report[name] == reference[name], name
    assert report["U statistic"] == reference["U statistic"]
    assert ai.format_p(report["p-value"]) == "< 1e-300"
    assert round(report["rank-biserial r"], 4) == reference["rank-biserial r"]
    assert round(report["P(success > failure)"], 4) == reference["P(success > failure)"]


def test_real_holm_keeps_all_tests_significant(real_results):
    adjusted = real_results["holm_adjusted"]
    assert float(f"{adjusted[0]:.3e}") == 2.215e-110
    assert [ai.format_p(value) for value in adjusted[1:]] == ["< 1e-300", "< 1e-300"]
    assert (adjusted < ai.ALPHA).all()
