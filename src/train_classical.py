"""Entrainement et evaluation du baseline Zero-R (modeles classiques).

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales.

Zero-R s'appuie sur ``DummyClassifier(strategy="most_frequent")`` : le modele
ignore les caracteristiques et predit toujours la classe majoritaire observee
dans le jeu d'entrainement. Il fournit le plancher de reference auquel les
autres modeles classiques (KNN, arbre de decision, Naive Bayes) seront
compares.

Ce module ne reimplemente rien : chargement, caracteristiques, cible,
decoupage, pretraitement et metriques viennent des modules partages
(``data_loader``, ``features``, ``targets``, ``splitting``, ``preprocessing``,
``evaluation``).

Usage :
    python src/train_classical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Le projet utilise des imports "plats" (import features, import splitting...).
# On garantit donc que src/ est dans sys.path, quel que soit le repertoire
# depuis lequel le script est lance.
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from sklearn.dummy import DummyClassifier

import data_loader
import features
import preprocessing
import splitting
import targets
from evaluation import evaluate_classifier

PROJECT_ROOT = SRC_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "startup_data.csv"

# Valeurs figees pour la reproductibilite (voir docs/methodology.md).
MAX_YEAR = 2026
RANDOM_STATE = 42

# Libelles lisibles pour les metriques renvoyees par evaluate_classifier.
METRIC_LABELS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"),
    ("f1_macro", "F1 (macro)"),
    ("precision_fermeture", "Precision classe 0 (fermeture)"),
    ("recall_fermeture", "Recall classe 0 (fermeture)"),
    ("f1_fermeture", "F1 classe 0 (fermeture)"),
)


def build_zero_r():
    """Retourne le Pipeline partage (pretraitement + Zero-R).

    ``scale=False`` : Zero-R ne calcule aucune distance ni aucun produit
    scalaire, la standardisation est donc inutile (voir
    docs/handoff_preprocessing.md).
    """
    return preprocessing.build_pipeline(
        DummyClassifier(strategy="most_frequent"),
        scale=False,
    )


def load_prepared_data(path: str | Path = DATASET_PATH):
    """Charge le CSV brut et applique la chaine partagee jusqu'a la cible.

    Etapes (aucune n'est reimplementee) :
    ``data_loader.load_raw`` -> ``features.build_features`` ->
    ``targets.prepare_target``.
    """
    raw = data_loader.load_raw(path)
    engineered = features.build_features(raw, max_year=MAX_YEAR)
    return targets.prepare_target(engineered)


def _check_protocol(X) -> None:
    """Verifie que X respecte exactement le protocole de splitting.py.

    Le controle refuse toute variable manquante, ajoutee ou interdite
    (fuite de cible ou d'information future).
    """
    expected = list(splitting.INITIAL_FEATURES)
    columns = list(X.columns)
    if columns != expected:
        raise ValueError(
            f"Variables d'entree inattendues : {columns} (attendu : {expected})."
        )
    forbidden = sorted(set(columns) & splitting.FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(f"Variables interdites presentes dans X : {forbidden}.")


def run_zero_r(data) -> dict:
    """Entraine Zero-R sur le train, predit et evalue sur le test.

    Le modele est ajuste uniquement sur ``X_train``/``y_train``. Le jeu de
    test reste intact jusqu'a l'evaluation finale : aucune fuite de donnees.
    Aucune recherche d'hyperparametres n'est necessaire pour Zero-R.
    """
    X_train, X_test, y_train, y_test = splitting.split_dataset(
        data, random_state=RANDOM_STATE
    )
    _check_protocol(X_train)
    _check_protocol(X_test)

    model = build_zero_r()
    model.fit(X_train, y_train)  # apprentissage sur le train uniquement

    y_pred = np.asarray(model.predict(X_test))
    metrics = evaluate_classifier(model, X_test, y_test)

    # Verification du comportement attendu de Zero-R : une seule classe
    # predite, et c'est la classe majoritaire apprise sur le train.
    majority_train = int(y_train.value_counts().idxmax())
    predicted_classes = sorted(np.unique(y_pred).tolist())
    if predicted_classes != [majority_train]:
        raise AssertionError(
            "Zero-R doit predire uniquement la classe majoritaire du train : "
            f"majoritaire={majority_train}, predite={predicted_classes}."
        )

    return {
        "model": model,
        "features": list(X_train.columns),
        "n_total": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_counts": {
            int(label): int(count)
            for label, count in y_train.value_counts().sort_index().items()
        },
        "test_counts": {
            int(label): int(count)
            for label, count in y_test.value_counts().sort_index().items()
        },
        "majority_train": majority_train,
        "predicted_classes": predicted_classes,
        "metrics": metrics,
    }


def _format_metric(value) -> str:
    if value is None:
        return "indisponible"
    return f"{value:.4f}"


def print_report(results: dict) -> None:
    """Affiche les effectifs, le modele et les metriques du jeu de test."""
    train_counts = results["train_counts"]
    test_counts = results["test_counts"]
    n_train = results["n_train"]
    n_test = results["n_test"]

    print("=== Chaine partagee ===")
    print(f"observations apres filtrage : {results['n_total']}")
    print(f"train / test                : {n_train} / {n_test}")
    for label in (0, 1):
        share = 100 * train_counts.get(label, 0) / n_train if n_train else 0.0
        print(f"classe {label} (train)          : {train_counts.get(label, 0)} ({share:.1f} %)")
    for label in (0, 1):
        print(f"classe {label} (test)           : {test_counts.get(label, 0)}")

    print("\n=== Modele ===")
    print("Zero-R : DummyClassifier(strategy='most_frequent')")
    print(f"variables d'entree ({len(results['features'])}) : {results['features']}")
    print(f"classe majoritaire apprise (train) : {results['majority_train']}")
    print(f"classes predites sur le test       : {results['predicted_classes']}")

    metrics = results["metrics"]
    print(f"\n=== Metriques (jeu de test, {n_test} observations) ===")
    for key, label in METRIC_LABELS:
        print(f"{label:<32} {_format_metric(metrics.get(key))}")
    print(f"{'ROC-AUC':<32} {_format_metric(metrics.get('roc_auc'))}")

    matrix = metrics["confusion_matrix"]
    print("\nMatrice de confusion (lignes = reel, colonnes = predit) :")
    print("            pred 0   pred 1")
    print(f"reel 0      {matrix[0][0]:>6}   {matrix[0][1]:>6}")
    print(f"reel 1      {matrix[1][0]:>6}   {matrix[1][1]:>6}")


def main() -> int:
    """Point d'entree en ligne de commande."""
    if not DATASET_PATH.is_file():
        print(f"Dataset introuvable : {DATASET_PATH}")
        print("Telecharger le CSV Kaggle dans data/raw/startup_data.csv.")
        return 1

    print(f"Dataset : {DATASET_PATH}")
    data = load_prepared_data()
    results = run_zero_r(data)
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
