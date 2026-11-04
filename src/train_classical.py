"""Entrainement et evaluation du baseline Zero-R (modeles classiques).

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales.

Zero-R s'appuie sur ``DummyClassifier(strategy="most_frequent")`` : le modele
ignore les caracteristiques et predit toujours la classe majoritaire observee
dans le jeu d'entrainement. Il fournit le plancher de reference auquel les
autres modeles classiques (KNN, arbre de decision, Naive Bayes) seront
compares.

KNN suit le meme protocole partage : pipeline ``scale=True`` (les distances
dependent de l'echelle), recherche d'hyperparametres par validation croisee
5 folds sur le train uniquement, puis une seule evaluation sur le test.

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

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

# KNN : grille d'hyperparametres exploree par validation croisee sur le train.
# ``k`` impair pour limiter les ex aequo entre les deux classes.
KNN_PARAM_GRID = {
    "model__n_neighbors": [5, 11, 25, 51, 101, 201, 301, 401],
    "model__weights": ["uniform", "distance"],
}
KNN_CV_FOLDS = 5
KNN_SCORING = "balanced_accuracy"


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


def build_knn():
    """Retourne le Pipeline partage (pretraitement + KNN).

    ``scale=True`` : KNN classe par proximite, donc la standardisation de
    ``founding_year`` est necessaire (voir docs/handoff_preprocessing.md).
    Le pretraitement reste dans le Pipeline : il est re-appris a chaque fold
    de validation croisee et sur le train complet, jamais sur le test.
    """
    return preprocessing.build_pipeline(
        KNeighborsClassifier(metric="minkowski", p=2),
        scale=True,
    )


def tune_knn(X_train, y_train) -> GridSearchCV:
    """Optimise KNN par validation croisee 5 folds sur le train uniquement.

    Le score suivi est la balanced accuracy, adaptee au desequilibre des
    classes (53,2 % de classe 1) et coherent avec le rapport final. Le jeu de
    test n'est jamais vu pendant cette recherche.
    """
    search = GridSearchCV(
        estimator=build_knn(),
        param_grid=KNN_PARAM_GRID,
        scoring=KNN_SCORING,
        cv=StratifiedKFold(
            n_splits=KNN_CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


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


def run_knn(data) -> dict:
    """Entraine KNN sur le train, predit et evalue sur le test.

    Sequence : decoupage partage (identique a Zero-R) -> verification du
    protocole -> recherche d'hyperparametres par CV 5 folds sur le train ->
    ajustement final sur tout le train -> evaluation unique sur le test.
    Le test n'intervient ni dans la recherche ni dans l'ajustement.
    """
    X_train, X_test, y_train, y_test = splitting.split_dataset(
        data, random_state=RANDOM_STATE
    )
    _check_protocol(X_train)
    _check_protocol(X_test)

    search = tune_knn(X_train, y_train)
    model = search.best_estimator_

    y_pred = np.asarray(model.predict(X_test))
    metrics = evaluate_classifier(model, X_test, y_test)

    best_index = int(search.best_index_)
    cv_std = float(search.cv_results_["std_test_score"][best_index])
    best_params = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }

    return {
        "model": model,
        "search": search,
        "best_params": best_params,
        "cv_score": float(search.best_score_),
        "cv_std": cv_std,
        "features": list(X_train.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "predicted_counts": {
            int(label): int(count)
            for label, count in zip(*np.unique(y_pred, return_counts=True))
        },
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


def print_knn_report(results: dict) -> None:
    """Affiche le reglage retenu, les predictions et les metriques KNN."""
    params = results["best_params"]
    n_test = results["n_test"]

    print("\n=== Modele ===")
    print("KNN : KNeighborsClassifier (distance euclidienne, pipeline scale=True)")
    print(f"variables d'entree ({len(results['features'])}) : {results['features']}")
    print(
        "hyperparametres retenus (CV "
        f"{KNN_CV_FOLDS} folds, {KNN_SCORING}) : "
        f"n_neighbors={params.get('n_neighbors')}, weights={params.get('weights')}"
    )
    print(
        f"balanced accuracy en CV (train)    : "
        f"{results['cv_score']:.4f} +/- {results['cv_std']:.4f}"
    )
    print(f"predictions sur le test             : {results['predicted_counts']}")

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


def print_comparison(zero_results: dict, knn_results: dict) -> None:
    """Compare les deux modeles sur le meme jeu de test."""
    print("\n=== Comparaison Zero-R / KNN (jeu de test) ===")
    print(f"{'Modele':<8}{'Accuracy':>11}{'Bal. acc.':>12}{'Macro F1':>11}{'ROC-AUC':>10}")
    for name, results in (("Zero-R", zero_results), ("KNN", knn_results)):
        metrics = results["metrics"]
        print(
            f"{name:<8}"
            f"{metrics['accuracy']:>11.4f}"
            f"{metrics['balanced_accuracy']:>12.4f}"
            f"{metrics['f1_macro']:>11.4f}"
            f"{_format_metric(metrics['roc_auc']):>10}"
        )
    gain = knn_results["metrics"]["balanced_accuracy"] - zero_results["metrics"]["balanced_accuracy"]
    print(f"\nGain de balanced accuracy par rapport au plancher : {gain:+.4f}")


def main() -> int:
    """Point d'entree en ligne de commande."""
    if not DATASET_PATH.is_file():
        print(f"Dataset introuvable : {DATASET_PATH}")
        print("Telecharger le CSV Kaggle dans data/raw/startup_data.csv.")
        return 1

    print(f"Dataset : {DATASET_PATH}")
    data = load_prepared_data()

    zero_results = run_zero_r(data)
    print_report(zero_results)

    knn_results = run_knn(data)
    print_knn_report(knn_results)

    print_comparison(zero_results, knn_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
