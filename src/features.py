"""Construction des caracteristiques initiales des startups.

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales.

Ce module transforme les colonnes brutes (``category_list``, ``founded_at`` et
geographie) en caracteristiques attendues par ``src/splitting.py`` :

    country_code, state_code, region, city,
    primary_category, founding_year, founding_month

Regles importantes
------------------
- Aucune imputation ici : les valeurs manquantes restent manquantes pour que
  les ``Pipeline`` scikit-learn les apprennent sur les seuls jeux
  d'entrainement (pas de fuite de donnees).
- ``status`` et ``target`` ne sont jamais utilises ni crees : la cible
  appartient a ``src/targets.py`` (role SA).
- Les dates aberrantes (annee < 1900 ou > annee de reference) sont converties
  en valeurs manquantes : ce sont des saisies invalides dans Crunchbase.

Limite connue sur ``founded_at``
--------------------------------
Environ 41 % des dates lisibles sont au 1er janvier et 56 % des mois valent
janvier : Crunchbase ne connait souvent que l'annee. ``founding_month`` est donc
peu informatif ; le mois n'est pas mis a ``NaN`` pour autant (choix a valider
avec SA), mais il ne doit pas etre interprete comme un vrai mois de creation.
"""

from __future__ import annotations

import sys

import pandas as pd

# Caracteristiques categorieles (texte) attendues par splitting.py.
CATEGORICAL_FEATURES = (
    "country_code",
    "state_code",
    "region",
    "city",
    "primary_category",
)

# Caracteristiques numeriques derivees de founded_at.
NUMERIC_FEATURES = ("founding_year", "founding_month")

# Contrat commun avec splitting.INITIAL_FEATURES (verifie dans les tests).
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Colonnes brutes necessaires pour construire FEATURE_COLUMNS.
REQUIRED_RAW_COLUMNS = (
    "country_code",
    "state_code",
    "region",
    "city",
    "category_list",
    "founded_at",
)

# Variables qui ne doivent jamais devenir des entrees de modele.
FORBIDDEN_FEATURES = {"status", "target"}

# Bornes de plausibilite des dates de creation.
MIN_PLAUSIBLE_YEAR = 1900

assert not (set(FEATURE_COLUMNS) & FORBIDDEN_FEATURES), "Fuite de cible dans les entrees."


def _clean_text(series: pd.Series) -> pd.Series:
    """Convertit en texte, retire les espaces et transforme le vide en NA.

    Une chaine vide ou remplie d'espaces ne doit pas devenir une modalite
    categorielle : elle doit rester manquante pour etre imputee dans le
    Pipeline.
    """
    text = series.astype("string").str.strip()
    return text.mask(text == "")


def extract_primary_category(category_list: pd.Series) -> pd.Series:
    """Retourne la premiere categorie non vide de ``category_list``.

    Exemple : ``"Apps|Games|Mobile"`` donne ``"Apps"``.
    ``"|Software"`` donne ``"Software"`` ; une valeur vide donne ``NA``.
    """
    text = _clean_text(category_list)
    # Ignore les separateurs en tete pour que le premier jeton soit non vide.
    text = text.str.replace(r"^\|+", "", regex=True)
    first = text.str.split("|", regex=False).str[0]
    return _clean_text(first)


def split_founded_at(
    founded_at: pd.Series,
    max_year: int | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Separe ``founded_at`` en annee et mois, avec controle de plausibilite.

    Parametres
    ----------
    founded_at : pandas.Series
        Dates au format texte (souvent ``AAAA-MM-JJ``).
    max_year : int | None
        Annee maximale acceptee. Par defaut l'annee courante. La fixer
        explicitement (par exemple ``2026``) rend les resultats reproductibles
        d'une annee a l'autre.

    Retour
    ------
    (founding_year, founding_month) : tuple[pandas.Series, pandas.Series]
        Deux series ``float64`` ; ``NaN`` signifie date inconnue ou aberrante.
        Le type ``float64`` est choisi pour rester compatible avec
        ``SimpleImputer``.
    """
    reference_year = pd.Timestamp.today().year if max_year is None else int(max_year)
    dates = pd.to_datetime(_clean_text(founded_at), errors="coerce", format="mixed")

    year = dates.dt.year.astype("Int64")
    month = dates.dt.month.astype("Int64")

    implausible = year.isna() | (year < MIN_PLAUSIBLE_YEAR) | (year > reference_year)
    year = year.mask(implausible).astype("float64")
    month = month.mask(implausible).astype("float64")
    return year, month


def build_features(df: pd.DataFrame, max_year: int | None = None) -> pd.DataFrame:
    """Ajoute les caracteristiques initiales au DataFrame brut.

    Parametres
    ----------
    df : pandas.DataFrame
        Donnees brutes issues de ``data_loader.load_raw`` (ou de ``read_csv``).
    max_year : int | None
        Annee maximale plausible, voir ``split_founded_at``.

    Retour
    ------
    pandas.DataFrame
        Copie du DataFrame d'origine avec les colonnes de ``FEATURE_COLUMNS``
        ajoutees. Les autres colonnes (dont ``status``) sont conservees telles
        quelles pour que ``targets.prepare_target`` puisse s'executer ensuite ;
        elles ne sont pas des entrees de modele.

    Notes
    -----
    Le DataFrame d'origine n'est pas modifie.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("L'argument df doit etre un pandas.DataFrame.")
    if df.empty:
        raise ValueError("Le DataFrame est vide.")

    missing = sorted(set(REQUIRED_RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(
            f"Colonnes brutes absentes : {missing}. "
            "Charger d'abord le CSV avec data_loader.load_raw."
        )

    result = df.copy()

    # Geographie : texte nettoye, modalites vides remplacees par NA.
    for column in ("country_code", "state_code", "region", "city"):
        result[column] = _clean_text(df[column])

    # Secteur : premiere categorie de category_list.
    result["primary_category"] = extract_primary_category(df["category_list"])

    # Date de creation : annee et mois, dates aberrantes exclues.
    founding_year, founding_month = split_founded_at(df["founded_at"], max_year=max_year)
    result["founding_year"] = founding_year
    result["founding_month"] = founding_month

    return result


def _check_contract_with_splitting() -> str:
    """Verifie que FEATURE_COLUMNS correspond a splitting.INITIAL_FEATURES."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import splitting  # import local (meme dossier src/)

    expected = tuple(splitting.INITIAL_FEATURES)
    if set(expected) != set(FEATURE_COLUMNS):
        raise AssertionError(
            "Contrat incoherent avec splitting.INITIAL_FEATURES : "
            f"attendu {expected}, obtenu {FEATURE_COLUMNS}"
        )
    return f"Contrat OK avec splitting.INITIAL_FEATURES ({len(expected)} variables)."


def _demo() -> None:
    """Test du module sans dataset Kaggle (donnees synthetiques)."""
    example = pd.DataFrame(
        {
            "category_list": [
                "Apps|Games|Mobile",
                "|Software|Analytics",
                "Biotechnology",
                "",
                None,
            ],
            "founded_at": [
                "2010-05-15",
                "2012-03-01",
                "1801-01-01",
                "2914-07-01",
                None,
            ],
            "country_code": ["USA", " FRA ", "", None, "DZA"],
            "state_code": ["CA", "IDF", None, "16", None],
            "region": ["SF Bay", "Paris", None, None, "Algiers"],
            "city": ["San Francisco", "Paris", None, "16", None],
        }
    )
    original = example.copy(deep=True)

    result = build_features(example, max_year=2026)

    # Le DataFrame d'origine ne doit pas etre modifie.
    pd.testing.assert_frame_equal(example, original)

    # primary_category : premier jeton non vide, vide -> NA.
    assert result.loc[0, "primary_category"] == "Apps"
    assert result.loc[1, "primary_category"] == "Software"
    assert result.loc[2, "primary_category"] == "Biotechnology"
    assert pd.isna(result.loc[3, "primary_category"])
    assert pd.isna(result.loc[4, "primary_category"])

    # Annee/mois : dates valides conservees, aberrantes -> NaN.
    assert result.loc[0, "founding_year"] == 2010 and result.loc[0, "founding_month"] == 5
    assert pd.isna(result.loc[2, "founding_year"])  # 1801
    assert pd.isna(result.loc[3, "founding_year"])  # 2914
    assert pd.isna(result.loc[4, "founding_year"])  # date absente

    # Types numeriques compatibles avec SimpleImputer.
    assert str(result["founding_year"].dtype) == "float64"
    assert str(result["founding_month"].dtype) == "float64"

    # Texte : espaces retires, chaine vide -> NA.
    assert result.loc[1, "country_code"] == "FRA"
    assert pd.isna(result.loc[2, "country_code"])

    # Aucune colonne interdite n'est creee.
    assert not (set(result.columns) & FORBIDDEN_FEATURES)

    # Toutes les caracteristiques attendues sont presentes.
    missing = sorted(set(FEATURE_COLUMNS) - set(result.columns))
    assert not missing

    print("TEST OK : primary_category, year/month, nettoyage du texte, pas de fuite.")
    print(_check_contract_with_splitting())
    print("\nExtrait :")
    print(result.loc[:, FEATURE_COLUMNS].to_string())


if __name__ == "__main__":
    _demo()
