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

L'arbre de decision suit le meme protocole avec ``scale=False`` (il est
insensible a l'echelle) et une grille elargie (critere, profondeur,
regularisation, ponderation des classes).

Naive Bayes gaussien complete la famille classique avec ``scale=False`` (la
standardisation est sans effet sur un modele gaussien par variable). Le
critere de validation croisee est ``f1_macro`` pour tous les modeles, aligne
sur ``train_advanced.py`` (Random Forest / SVM) a la demande de SA ; la
balanced accuracy reste calculee et publiee en complement.

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
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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

# Protocole de validation croisee commun a tous les modeles classiques.
# Critere unique aligne sur ``train_advanced.py`` (Random Forest / SVM) a la
# demande de SA : ``f1_macro``. La balanced accuracy est conservee en second
# critere, uniquement pour le rapport (elle ne sert pas a choisir le modele).
CV_FOLDS = 5
CV_SCORING = "f1_macro"
CV_SECONDARY_SCORING = "balanced_accuracy"

# KNN : grille d'hyperparametres exploree par validation croisee sur le train.
# ``k`` impair pour limiter les ex aequo entre les deux classes.
KNN_PARAM_GRID = {
    "model__n_neighbors": [5, 11, 25, 51, 101, 201, 301, 401],
    "model__weights": ["uniform", "distance"],
}

# Arbre de decision : grille 3 x 5 x 4 x 4 x 2 = 480 combinaisons, 5 folds.
# Les profondeurs bornees et les seuils de feuille limitent le surapprentissage
# des arbres profonds sur 306 colonnes one-hot. ``log_loss`` est conserve pour
# repondre au protocole, meme si scikit-learn le traite comme ``entropy``.
DECISION_TREE_PARAM_GRID = {
    "model__criterion": ["gini", "entropy", "log_loss"],
    "model__max_depth": [None, 5, 10, 20, 30],
    "model__min_samples_split": [2, 5, 10, 20],
    "model__min_samples_leaf": [1, 2, 5, 10],
    "model__class_weight": [None, "balanced"],
}

# Naive Bayes gaussien : seul hyperparametre reellement disponible, la variance
# de lissage ``var_smoothing`` (elargissement numerique des variances). La
# grille reste volontairement courte : le modele est quasi sans reglage.
NAIVE_BAYES_PARAM_GRID = {
    "model__var_smoothing": [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
}


def _cv_splitter() -> StratifiedKFold:
    """Retourne le decoupage CV commun : 5 folds stratifies, graine 42."""
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def _secondary_cv(model, X_train, y_train) -> tuple[float, float]:
    """Balanced accuracy en CV du reglage retenu (information, pas selection).

    Le choix des hyperparametres reste fonde sur ``CV_SCORING`` ; cette mesure
    sert uniquement a garder la continuite avec les rapports precedents.
    """
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=_cv_splitter(),
        scoring=CV_SECONDARY_SCORING,
    )
    return float(scores.mean()), float(scores.std())


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

    Le score suivi est ``f1_macro`` : critere unique de l'equipe, aligne sur
    ``train_advanced.py`` (Random Forest / SVM) a la demande de SA. La balanced
    accuracy est seulement rapportee en complement. Le jeu de test n'est jamais
    vu pendant cette recherche.
    """
    search = GridSearchCV(
        estimator=build_knn(),
        param_grid=KNN_PARAM_GRID,
        scoring=CV_SCORING,
        cv=_cv_splitter(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def build_decision_tree():
    """Retourne le Pipeline partage (pretraitement + arbre de decision).

    ``scale=False`` : un arbre separe sur des seuils, il est insensible a
    l'echelle des variables (voir docs/handoff_preprocessing.md).
    ``random_state`` est fixe pour rendre l'arbre reproductible.
    """
    return preprocessing.build_pipeline(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        scale=False,
    )


def tune_decision_tree(X_train, y_train) -> GridSearchCV:
    """Optimise l'arbre par validation croisee 5 folds sur le train seul.

    Meme protocole que KNN : ``StratifiedKFold`` melange avec la graine
    commune, score suivi = ``f1_macro`` (critere unique de l'equipe). Le test
    n'est jamais transmis a cette fonction.
    """
    search = GridSearchCV(
        estimator=build_decision_tree(),
        param_grid=DECISION_TREE_PARAM_GRID,
        scoring=CV_SCORING,
        cv=_cv_splitter(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def build_naive_bayes():
    """Retourne le Pipeline partage (pretraitement + Naive Bayes gaussien).

    ``scale=False`` : la standardisation est une transformation affine, sans
    effet sur l'ajustement d'une gaussienne par variable ; elle n'apporterait
    donc rien (voir docs/handoff_preprocessing.md). ``GaussianNB`` est
    deterministe : aucun ``random_state`` n'est necessaire.
    """
    return preprocessing.build_pipeline(GaussianNB(), scale=False)


def tune_naive_bayes(X_train, y_train) -> GridSearchCV:
    """Optimise ``var_smoothing`` par CV 5 folds sur le train uniquement.

    Meme protocole et meme critere (``f1_macro``) que KNN et l'arbre ; le test
    n'est jamais transmis a cette fonction.
    """
    search = GridSearchCV(
        estimator=build_naive_bayes(),
        param_grid=NAIVE_BAYES_PARAM_GRID,
        scoring=CV_SCORING,
        cv=_cv_splitter(),
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
    bal_mean, bal_std = _secondary_cv(model, X_train, y_train)

    return {
        "model": model,
        "search": search,
        "best_params": best_params,
        "cv_score": float(search.best_score_),
        "cv_std": cv_std,
        "cv_balanced_accuracy": bal_mean,
        "cv_balanced_std": bal_std,
        "features": list(X_train.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "predicted_counts": {
            int(label): int(count)
            for label, count in zip(*np.unique(y_pred, return_counts=True))
        },
        "metrics": metrics,
    }


def run_decision_tree(data) -> dict:
    """Entraine l'arbre sur le train, predit et evalue sur le test.

    Sequence identique a KNN : decoupage partage -> verification du protocole
    -> CV 5 folds sur le train -> reentrainement du meilleur pipeline sur tout
    le train -> evaluation unique sur le test. La balanced accuracy
    d'entrainement est aussi mesuree (train seul) pour analyser le
    surapprentissage, sans jamais toucher au test pendant l'optimisation.
    """
    X_train, X_test, y_train, y_test = splitting.split_dataset(
        data, random_state=RANDOM_STATE
    )
    _check_protocol(X_train)
    _check_protocol(X_test)

    search = tune_decision_tree(X_train, y_train)
    model = search.best_estimator_

    y_pred = np.asarray(model.predict(X_test))
    metrics = evaluate_classifier(model, X_test, y_test)

    tree = model.named_steps["model"]
    best_index = int(search.best_index_)
    best_params = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    bal_mean, bal_std = _secondary_cv(model, X_train, y_train)

    return {
        "model": model,
        "search": search,
        "best_params": best_params,
        "cv_score": float(search.best_score_),
        "cv_std": float(search.cv_results_["std_test_score"][best_index]),
        "cv_balanced_accuracy": bal_mean,
        "cv_balanced_std": bal_std,
        "train_score": float(balanced_accuracy_score(y_train, model.predict(X_train))),
        "depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "features": list(X_train.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "predicted_counts": {
            int(label): int(count)
            for label, count in zip(*np.unique(y_pred, return_counts=True))
        },
        "metrics": metrics,
    }


def run_naive_bayes(data) -> dict:
    """Entraine Naive Bayes sur le train, predit et evalue sur le test.

    Sequence identique aux autres modeles : decoupage partage -> verification
    du protocole -> CV 5 folds sur le train -> ajustement final sur tout le
    train -> evaluation unique sur le test. Le test n'est ni vu pendant la
    recherche ni utilise pour choisir ``var_smoothing``.
    """
    X_train, X_test, y_train, y_test = splitting.split_dataset(
        data, random_state=RANDOM_STATE
    )
    _check_protocol(X_train)
    _check_protocol(X_test)

    search = tune_naive_bayes(X_train, y_train)
    model = search.best_estimator_

    y_pred = np.asarray(model.predict(X_test))
    metrics = evaluate_classifier(model, X_test, y_test)

    best_index = int(search.best_index_)
    best_params = {
        key.replace("model__", ""): value
        for key, value in search.best_params_.items()
    }
    bal_mean, bal_std = _secondary_cv(model, X_train, y_train)

    return {
        "model": model,
        "search": search,
        "best_params": best_params,
        "cv_score": float(search.best_score_),
        "cv_std": float(search.cv_results_["std_test_score"][best_index]),
        "cv_balanced_accuracy": bal_mean,
        "cv_balanced_std": bal_std,
        "train_score": float(balanced_accuracy_score(y_train, model.predict(X_train))),
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
        f"{CV_FOLDS} folds, {CV_SCORING}) : "
        f"n_neighbors={params.get('n_neighbors')}, weights={params.get('weights')}"
    )
    print(
        f"{CV_SCORING} en CV (train)              : "
        f"{results['cv_score']:.4f} +/- {results['cv_std']:.4f}"
    )
    print(
        f"{CV_SECONDARY_SCORING} en CV (train)      : "
        f"{results['cv_balanced_accuracy']:.4f} +/- {results['cv_balanced_std']:.4f}"
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


def print_decision_tree_report(results: dict) -> None:
    """Affiche le reglage retenu, l'arbre et les metriques du Decision Tree."""
    params = results["best_params"]
    n_test = results["n_test"]

    print("\n=== Modele ===")
    print("Decision Tree : DecisionTreeClassifier (pipeline scale=False)")
    print(f"variables d'entree ({len(results['features'])}) : {results['features']}")
    print(
        "hyperparametres retenus (CV "
        f"{CV_FOLDS} folds, {CV_SCORING}) :"
    )
    for key in (
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "class_weight",
    ):
        print(f"  {key:<18} : {params.get(key)}")
    print(f"profondeur / feuilles de l'arbre  : {results['depth']} / {results['n_leaves']}")
    print(
        f"{CV_SCORING} en CV (train)              : "
        f"{results['cv_score']:.4f} +/- {results['cv_std']:.4f}"
    )
    print(
        f"{CV_SECONDARY_SCORING} en CV (train)      : "
        f"{results['cv_balanced_accuracy']:.4f} +/- {results['cv_balanced_std']:.4f}"
    )
    print(f"balanced accuracy sur le train     : {results['train_score']:.4f}")
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


def print_naive_bayes_report(results: dict) -> None:
    """Affiche le reglage retenu et les metriques de Naive Bayes gaussien."""
    params = results["best_params"]
    n_test = results["n_test"]

    print("\n=== Modele ===")
    print("Naive Bayes : GaussianNB (pipeline scale=False)")
    print(f"variables d'entree ({len(results['features'])}) : {results['features']}")
    print(
        "hyperparametres retenus (CV "
        f"{CV_FOLDS} folds, {CV_SCORING}) : "
        f"var_smoothing={params.get('var_smoothing')}"
    )
    print(
        f"{CV_SCORING} en CV (train)              : "
        f"{results['cv_score']:.4f} +/- {results['cv_std']:.4f}"
    )
    print(
        f"{CV_SECONDARY_SCORING} en CV (train)      : "
        f"{results['cv_balanced_accuracy']:.4f} +/- {results['cv_balanced_std']:.4f}"
    )
    print(f"balanced accuracy sur le train     : {results['train_score']:.4f}")
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


def print_comparison(models) -> None:
    """Compare tous les modeles sur le meme jeu de test.

    ``models`` est une sequence de couples ``(nom, resultats)``. Le premier
    element sert de plancher de reference (Zero-R).
    """
    print("\n=== Comparaison des modeles (jeu de test) ===")
    header = (
        f"{'Modele':<15}{'Accuracy':>10}{'Bal. acc.':>11}"
        f"{'Prec. macro':>13}{'Recall macro':>14}{'F1 macro':>10}{'ROC-AUC':>10}"
    )
    print(header)
    for name, results in models:
        metrics = results["metrics"]
        print(
            f"{name:<15}"
            f"{metrics['accuracy']:>10.4f}"
            f"{metrics['balanced_accuracy']:>11.4f}"
            f"{metrics['precision_macro']:>13.4f}"
            f"{metrics['recall_macro']:>14.4f}"
            f"{metrics['f1_macro']:>10.4f}"
            f"{_format_metric(metrics['roc_auc']):>10}"
        )

    floor = models[0][0]
    floor_metrics = models[0][1]["metrics"]
    for name, results in models[1:]:
        metrics = results["metrics"]
        print(
            f"\n{name} vs {floor} : "
            f"accuracy {metrics['accuracy'] - floor_metrics['accuracy']:+.4f} | "
            f"balanced accuracy {metrics['balanced_accuracy'] - floor_metrics['balanced_accuracy']:+.4f} | "
            f"F1 macro {metrics['f1_macro'] - floor_metrics['f1_macro']:+.4f} | "
            f"ROC-AUC {metrics['roc_auc'] - floor_metrics['roc_auc']:+.4f}"
        )


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

    tree_results = run_decision_tree(data)
    print_decision_tree_report(tree_results)

    nb_results = run_naive_bayes(data)
    print_naive_bayes_report(nb_results)

    print_comparison(
        [
            ("Zero-R", zero_results),
            ("KNN", knn_results),
            ("Decision Tree", tree_results),
            ("Naive Bayes", nb_results),
        ]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
