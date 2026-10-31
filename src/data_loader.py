"""Chargement et audit du dataset brut Crunchbase.

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales.

Ce module ne transforme pas les donnees : il charge le CSV brut et produit un
rapport de controle (dimensions, types, valeurs manquantes, doublons, statuts,
plage de dates). La cible est construite par ``src/targets.py`` (SA) et les
caracteristiques par ``src/features.py`` (role Data).

Usage :
    python src/data_loader.py data/raw/startup_data.csv
    python src/data_loader.py            # cherche automatiquement le CSV
    python src/data_loader.py --demo     # test sans dataset Kaggle
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Colonnes minimales attendues dans le CSV brut pour que la suite du projet
# (cible + caracteristiques) puisse fonctionner.
REQUIRED_RAW_COLUMNS = ("status", "category_list", "founded_at")

# Emplacements ou chercher le CSV brut si aucun chemin n'est fourni.
SEARCH_PATTERNS = ("data/raw/*.csv", "data/*.csv", "*.csv")

# Bornes de controle des dates de creation (annees aberrantes).
MIN_PLAUSIBLE_YEAR = 1900


def load_raw(path: str | Path) -> pd.DataFrame:
    """Charge le CSV brut et normalise uniquement les noms de colonnes.

    Parametres
    ----------
    path : str | Path
        Chemin du CSV telecharge depuis Kaggle.

    Retour
    ------
    pandas.DataFrame
        Donnees brutes telles quelles, colonnes sans espaces superflus.

    Notes
    -----
    Aucune ligne, aucun statut et aucune valeur n'est modifie ici. Le filtrage
    des statuts (ipo/acquired/closed) est la responsabilite de
    ``targets.prepare_target``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    # utf-8-sig retire le BOM eventuel en tete de la premiere colonne.
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        raise ValueError(f"Le fichier ne contient aucune ligne : {path}")

    missing = sorted(set(REQUIRED_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            f"Colonnes obligatoires absentes : {missing}. "
            f"Colonnes presentes : {list(df.columns)}"
        )
    return df


def find_raw_file(base: str | Path = ".") -> Path | None:
    """Retourne le plus gros CSV trouve dans les emplacements attendus.

    Le plus gros fichier est retenu pour eviter de selectionner un extrait de
    test alors que le dataset complet est present.
    """
    base = Path(base)
    candidates: list[Path] = []
    for pattern in SEARCH_PATTERNS:
        candidates.extend(path for path in base.glob(pattern) if path.is_file())
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def audit(df: pd.DataFrame, current_year: int | None = None) -> dict:
    """Calcule un rapport de controle du DataFrame brut.

    Parametres
    ----------
    df : pandas.DataFrame
        Donnees brutes issues de ``load_raw``.
    current_year : int | None
        Annee de reference pour detecter les dates futures (defaut : annee
        courante du systeme).

    Retour
    ------
    dict
        Dimensions, types, doublons, valeurs manquantes par colonne, effectifs
        par statut et controle de la colonne ``founded_at``.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'argument df doit etre un pandas.DataFrame.")
    if df.empty:
        raise ValueError("Le DataFrame est vide.")

    reference_year = pd.Timestamp.today().year if current_year is None else int(current_year)

    missing_counts = df.isna().sum()
    missing = {
        column: int(count)
        for column, count in missing_counts.sort_values(ascending=False).items()
        if int(count) > 0
    }

    status = df["status"].astype("string").str.strip().str.lower()
    status_counts = {
        ("<manquant>" if pd.isna(value) else str(value)): int(count)
        for value, count in status.value_counts(dropna=False).items()
    }

    founded = pd.to_datetime(df["founded_at"], errors="coerce", format="mixed")
    years = founded.dt.year
    valid_years = years.dropna()
    too_old = int((valid_years < MIN_PLAUSIBLE_YEAR).sum())
    in_future = int((valid_years > reference_year).sum())

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "n_duplicates": int(df.duplicated().sum()),
        "missing": missing,
        "missing_cells": int(missing_counts.sum()),
        "status_counts": status_counts,
        "founded_at": {
            "n_unparsed": int(founded.isna().sum()),
            "min_year": None if valid_years.empty else int(valid_years.min()),
            "max_year": None if valid_years.empty else int(valid_years.max()),
            "n_before_1900": too_old,
            "n_in_future": in_future,
        },
    }


def print_report(report: dict) -> None:
    """Affiche le rapport produit par ``audit`` de facon lisible."""
    print(f"Lignes : {report['n_rows']}")
    print(f"Colonnes : {report['n_columns']}")
    print(f"Doublons complets : {report['n_duplicates']}")
    print(f"Cellules manquantes : {report['missing_cells']}")

    print("\nStatuts :")
    for status, count in sorted(report["status_counts"].items(), key=lambda item: -item[1]):
        print(f"  {status:<12} {count:>7}")

    print("\nValeurs manquantes par colonne :")
    if not report["missing"]:
        print("  aucune")
    for column, count in report["missing"].items():
        share = 100 * count / report["n_rows"]
        print(f"  {column:<24} {count:>7}  ({share:5.1f} %)")

    dates = report["founded_at"]
    print("\nColonne founded_at :")
    print(f"  dates illisibles : {dates['n_unparsed']}")
    print(f"  annee minimale   : {dates['min_year']}")
    print(f"  annee maximale   : {dates['max_year']}")
    print(f"  avant 1900       : {dates['n_before_1900']}")
    print(f"  dans le futur    : {dates['n_in_future']}")


def _demo() -> None:
    """Test du module sans dataset Kaggle (donnees synthetiques)."""
    example = pd.DataFrame(
        {
            "status": [" ipo", "acquired", "closed", "operating", None],
            "category_list": ["Software", "Health", "Software", None, "Retail"],
            "founded_at": ["2010-05-01", "2012-03-01", "1801-01-01", "2015-07-01", "2035-01-01"],
            "country_code": ["USA", "FRA", "USA", "DZA", "USA"],
            "city": ["NY", None, "LA", None, "NY"],
        }
    )
    report = audit(example, current_year=2026)

    assert report["n_rows"] == 5
    assert report["n_duplicates"] == 0
    assert report["status_counts"]["<manquant>"] == 1
    assert report["missing"]["city"] == 2
    assert report["founded_at"]["n_unparsed"] == 0
    assert report["founded_at"]["n_before_1900"] == 1
    assert report["founded_at"]["n_in_future"] == 1

    print_report(report)
    print("\nTEST OK : audit des dimensions, manquants, statuts et dates aberrantes.")


def main(argv: list[str] | None = None) -> int:
    """Point d'entree en ligne de commande."""
    args = list(sys.argv[1:] if argv is None else argv)

    if "--demo" in args:
        _demo()
        return 0

    path = Path(args[0]) if args else find_raw_file()
    if path is None:
        print("Aucun CSV trouve dans data/raw/, data/ ou la racine du projet.")
        print("Telecharger le dataset Kaggle (voir data/README_DATA.md) puis relancer :")
        print("  python src/data_loader.py data/raw/startup_data.csv")
        return 1

    print(f"Fichier : {path}")
    print_report(audit(load_raw(path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
