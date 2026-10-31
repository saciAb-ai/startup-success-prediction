"""Separation train/test commune et reproductible.

Projet : Prediction du succes ou de l'echec des startups a partir de leurs caracteristiques initiales.

Important : cette fonction attend les caracteristiques creees par l'equipe Data.
Elle ne nettoie pas les donnees et ne doit pas apprendre de transformations.
Imputation, encodage et standardisation doivent etre faits dans les Pipeline
scikit-learn des modeles, sur les seuls jeux d'entrainement.
"""

from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

# Liste initiale a valider avec la personne en charge du preprocessing.
# primary_category est derive de category_list ; founding_year de founded_at.
INITIAL_FEATURES = (
    "country_code",
    "state_code",
    "region",
    "city",
    "primary_category",
    "founding_year",
)

# Variables interdites parmi les entrees de ce protocole initial.
FORBIDDEN_FEATURES = {
    "status",
    "target",
    "funding_total_usd",
    "funding_rounds",
    "first_funding_at",
    "last_funding_at",
    "time_to_first_funding_days",
    "funding_duration_days",
    "avg_funding_per_round",
}


def split_dataset(
    df: pd.DataFrame,
    features: Sequence[str] = INITIAL_FEATURES,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Retourne X_train, X_test, y_train, y_test avec stratification.

    Parametres
    ----------
    df : pandas.DataFrame
        Dataset deja filtre, avec une colonne target binaire (0 ou 1).
    features : sequence de str
        Caracteristiques initiales selectionnees en accord avec l'equipe Data.
    test_size : float
        Proportion du jeu de test, par defaut 20 %.
    random_state : int
        Graine fixee pour reproduire le partage.

    Notes
    -----
    Le jeu de test doit rester intact jusqu'a l'evaluation finale. Les choix
    d'hyperparametres se font uniquement sur X_train/y_train, via CV >= 5 folds.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit etre un pandas.DataFrame.")
    if not 0 < test_size < 1:
        raise ValueError("test_size doit etre strictement entre 0 et 1.")
    if "target" not in df.columns:
        raise ValueError("La colonne 'target' est absente : appeler prepare_target d'abord.")
    features = list(features)
    if not features:
        raise ValueError("La liste de caracteristiques est vide.")
    if len(features) != len(set(features)):
        raise ValueError("La liste de caracteristiques contient des doublons.")
    forbidden = sorted(set(features) & FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(f"Caracteristiques interdites (fuite temporelle/cible) : {forbidden}")
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(
            f"Caracteristiques absentes : {missing}. "
            "Faire creer ces colonnes par l'etape de feature engineering."
        )
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    if df["target"].isna().any() or not df["target"].isin([0, 1]).all():
        raise ValueError("'target' doit contenir uniquement 0 et 1, sans valeur manquante.")
    counts = df["target"].value_counts()
    if set(counts.index) != {0, 1} or counts.min() < 2:
        raise ValueError("Les deux classes (0 et 1) doivent compter au moins 2 exemples chacune.")

    X = df.loc[:, features].copy()
    y = df["target"].astype(int).copy()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
