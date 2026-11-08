"""Tests des aides d'affichage (format français des nombres)."""

from __future__ import annotations

import pandas as pd

import ui


def test_fr_number():
    assert ui.fr_number(2004.784, 2) == "2004,78"
    assert ui.fr_number(13334) == "13\u202f334"
    assert ui.fr_number(2007.0) == "2007"
    assert ui.fr_number(float("nan")) == "—"
    assert ui.fr_number("USA") == "USA"


def test_french_numbers_formats_only_numbers():
    frame = pd.DataFrame({"a": [1.5, 12345.25], "b": ["x", "y"]})
    shown = ui.french_numbers(frame)
    assert shown.loc[0, "a"] == "1,50"
    assert shown.loc[1, "a"] == "12\u202f345,25"
    assert shown.loc[0, "b"] == "x"
