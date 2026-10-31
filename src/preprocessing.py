"""Pretraitement commun (imputation, encodage, standardisation) en Pipeline.

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales.

Pourquoi des Pipeline ?
------------------------
Tout traitement qui apprend des parametres (mediane, liste des modalites,
moyenne, ecart-type) doit etre appris sur les seules donnees d'entrainement.
Place dans un ``Pipeline``, il est re-appris a chaque fold de validation
croisee : aucune information du jeu de test ne fuit dans l'entrainement.

Contenu du preprocesseur
------------------------
- Variables categorieles (pays, etat, region, ville, secteur) :
  ``SimpleImputer`` avec la valeur "Inconnu" puis ``OneHotEncoder``.
  ``handle_unknown="ignore"`` evite un plantage sur une ville jamais vue
  (application Streamlit) et ``min_frequency`` regroupe les modalites rares.
- Variables numeriques (annee et mois de creation) :
  ``SimpleImputer`` par la mediane puis, si ``scale=True``, ``StandardScaler``.

Quel reglage pour quel modele ?
-------------------------------
- ``scale=True``  : KNN, SVM (sensibles a l'echelle des variables).
- ``scale=False`` : Zero-R, Decision Tree, Random Forest, Naive Bayes.

Usage
-----
    from preprocessing import build_pipeline
    model = build_pipeline(RandomForestClassifier(), scale=False)
    model.fit(X_train, y_train)          # tout est appris ici, train uniquement
    model.predict(X_test)
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Contrat commun avec features.FEATURE_COLUMNS (verifie dans les tests).
CATEGORICAL_FEATURES = (
    "country_code",
    "state_code",
    "region",
    "city",
    "primary_category",
)
NUMERIC_FEATURES = ("founding_year", "founding_month")
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Modalite explicite pour les valeurs manquantes : elle distingue "information
# absente" d'une vraie modalite et evite de choisir arbitrairement la plus
# frequente, qui pourrait masquer un effet propre au manque d'information.
MISSING_CATEGORY = "Inconnu"

# Seuil de regroupement des modalites rares. Avec 5 111 villes et 728 secteurs
# differents, un encodage complet produit 2 820 colonnes sur le train ;
# min_frequency=20 le ramene a 307 colonnes sans perdre les modalites utiles.
DEFAULT_MIN_FREQUENCY = 20


class _CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Prepare les variables categorieles pour ``SimpleImputer``.

    Le dtype pandas ``string`` nullable utilise ``pd.NA``, que
    ``SimpleImputer`` ne sait pas tester (``pd.NA != pd.NA`` est ambigu).
    Cette etape repasse donc en ``object`` et remplace les manquants par
    ``np.nan``, que scikit-learn reconnait comme valeur manquante.

    Elle expose ``get_feature_names_out`` pour que les noms des colonnes
    produites par le ``ColumnTransformer`` restent disponibles.
    """

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(pd.DataFrame(X).columns, dtype=object)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).astype("object")
        return frame.where(pd.notna(frame), np.nan).to_numpy(dtype=object)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def build_preprocessor(
    scale: bool = False,
    min_frequency: int | None = DEFAULT_MIN_FREQUENCY,
) -> ColumnTransformer:
    """Construit le preprocesseur commun (non entraine).

    Parametres
    ----------
    scale : bool
        Ajoute une standardisation des variables numeriques (True pour KNN et
        SVM, False pour les arbres et Naive Bayes).
    min_frequency : int | None
        Frequence minimale d'une modalite categorielle ; en dessous, la
        modalite est regroupee dans ``infrequent_sklearn``. ``None`` desactive
        le regroupement.

    Retour
    ------
    sklearn.compose.ColumnTransformer
        A ajuster (``fit``/``fit_transform``) sur X_train uniquement.
    """
    if min_frequency is not None and int(min_frequency) < 1:
        raise ValueError("min_frequency doit etre un entier >= 1 ou None.")

    categorical_steps = [
        ("clean", _CategoricalCleaner()),
        ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=min_frequency,
                sparse_output=False,
                dtype="float64",
            ),
        ),
    ]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("cat", Pipeline(categorical_steps), list(CATEGORICAL_FEATURES)),
            ("num", Pipeline(numeric_steps), list(NUMERIC_FEATURES)),
        ],
        verbose_feature_names_out=True,
    )


def build_pipeline(
    model,
    scale: bool = False,
    min_frequency: int | None = DEFAULT_MIN_FREQUENCY,
) -> Pipeline:
    """Assemble preprocesseur + modele en un seul ``Pipeline``.

    Le Pipeline garantit que l'imputation et l'encodage sont ajustes sur les
    seuls donnees d'entrainement, y compris a chaque fold de ``GridSearchCV``.

    Parametres
    ----------
    model : estimateur scikit-learn
        Classifieur avec ``fit`` et ``predict`` (ex. ``DummyClassifier``).
    scale : bool
        Voir ``build_preprocessor``.
    min_frequency : int | None
        Voir ``build_preprocessor``.

    Retour
    ------
    sklearn.pipeline.Pipeline
        Etapes nommees ``preprocessor`` et ``model``.
    """
    if model is None:
        raise ValueError("Un estimateur scikit-learn est requis (parametre model).")
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(scale=scale, min_frequency=min_frequency)),
            ("model", model),
        ]
    )


def feature_names_out(preprocessor: ColumnTransformer) -> list[str]:
    """Retourne les noms des colonnes produites par un preprocesseur ajuste.

    Utile pour l'importance des variables (Random Forest) et la lecture des
    coefficients (SVM lineaire).
    """
    return [str(name) for name in preprocessor.get_feature_names_out()]


def _check_contract_with_features() -> str:
    """Verifie que les listes de variables correspondent a src/features.py.

    Le controle est ignore si ``features.py`` n'est pas present (par exemple
    sur une branche qui ne contient pas encore l'etape precedente).
    """
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import features  # import local (meme dossier src/)
    except ImportError:
        return "Contrat non verifie : src/features.py absent de cette branche."

    expected = tuple(features.FEATURE_COLUMNS)
    if set(expected) != set(FEATURE_COLUMNS):
        raise AssertionError(
            "Contrat incoherent avec features.FEATURE_COLUMNS : "
            f"attendu {expected}, obtenu {FEATURE_COLUMNS}"
        )
    return f"Contrat OK avec features.FEATURE_COLUMNS ({len(expected)} variables)."


def _demo() -> None:
    """Test du module sans dataset Kaggle (donnees synthetiques)."""
    import numpy as np
    import pandas as pd
    from sklearn.dummy import DummyClassifier

    X = pd.DataFrame(
        {
            "country_code": ["USA", "FRA", None, "USA", "DZA", "USA"],
            "state_code": ["CA", None, "IDF", "CA", None, "NY"],
            "region": ["SF Bay", None, "Paris", "NY", "Algiers", "NY"],
            "city": ["SF", "Paris", None, "NY", "Algiers", "NY"],
            "primary_category": ["Software", "Health", None, "Software", "Retail", "Software"],
            "founding_year": [2010.0, 2012.0, np.nan, 2008.0, 2011.0, np.nan],
            "founding_month": [5.0, 3.0, np.nan, 9.0, 7.0, np.nan],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1], name="target")

    # 1) Sans standardisation : les valeurs numeriques restent dans leur echelle.
    plain = build_preprocessor(scale=False, min_frequency=None)
    X_plain = plain.fit_transform(X)
    assert X_plain.ndim == 2
    assert not np.isnan(X_plain).any(), "aucune valeur manquante ne doit rester"

    names = feature_names_out(plain)
    year_values = X_plain[:, names.index("num__founding_year")]
    month_values = X_plain[:, names.index("num__founding_month")]
    # Les deux annees manquantes sont imputees par la mediane observee : 2010.5.
    assert (year_values == 2010.5).sum() == 2
    assert (month_values == 6.0).sum() == 2  # mediane de [3, 5, 7, 9]
    # Sans standardisation, l'echelle d'origine est conservee.
    assert year_values.min() == 2008.0 and year_values.max() == 2012.0

    # 2) Avec standardisation : moyenne ~ 0 et ecart-type ~ 1 sur le train.
    scaled = build_preprocessor(scale=True, min_frequency=None)
    X_scaled = scaled.fit_transform(X)
    year_scaled = X_scaled[:, feature_names_out(scaled).index("num__founding_year")]
    assert abs(year_scaled.mean()) < 1e-9
    assert abs(year_scaled.std() - 1.0) < 1e-9

    # 3) Une modalite jamais vue ne fait pas echouer la transformation.
    unseen = X.iloc[[0]].copy()
    unseen.loc[0, "city"] = "Atlantis"
    unseen.loc[0, "primary_category"] = "Espace"
    assert plain.transform(unseen).shape[0] == 1

    # 4) Pipeline complet : entrainement et prediction sans erreur.
    pipeline = build_pipeline(DummyClassifier(strategy="prior"), scale=False)
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    assert set(np.unique(predictions)) <= {0, 1}

    # 5) Les parametres sont bien appris sur le train fourni.
    imputer = pipeline.named_steps["preprocessor"].named_transformers_["num"].named_steps["imputer"]
    assert imputer.statistics_[0] == 2010.5  # mediane de [2010, 2012, 2008]

    print("TEST OK : imputation, encodage, standardisation, modalite inconnue, Pipeline.")
    print(_check_contract_with_features())
    print(f"Colonnes produites (sans regroupement) : {len(names)}")
    print(f"Exemples de noms : {names[:3]} ... {names[-2:]}")


if __name__ == "__main__":
    _demo()
