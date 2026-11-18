"""Entrainement et comparaison de deux modeles ensemble : XGBoost et AdaBoost.

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales.

Ce module complete les modeles classiques de ``train_classical.py`` avec deux
methodes d'ensemble. Il ne reimplemente aucun module partage : chargement,
caracteristiques, cible, decoupage, pretraitement et metriques proviennent de
``data_loader``, ``features``, ``targets``, ``splitting``, ``preprocessing``
et ``evaluation``.

Protocole strictement identique a celui de l'equipe :
- CSV de reference explicite ``data/raw/startup_data.csv`` (sans ISR) ;
- 6 variables initiales, ``founding_month`` exclu (EDA uniquement) ;
- split stratifie 80/20, graine 42 ;
- ``StratifiedKFold(5, shuffle=True, random_state=42)`` ;
- critere de selection unique ``f1_macro`` (aligned sur train_advanced.py) ;
- recherche d'hyperparametres sur le train uniquement, test utilise une seule
  fois pour l'evaluation finale ;
- ``scale=False`` pour les deux modeles (arbres : insensibles a l'echelle).

Aucune identite personnelle n'est ecrite dans les rapports : seul le code
etudiant ``CS23`` apparait.

Usage ::

    python src/train_ensemble.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Le projet utilise des imports "plats" (import features, import splitting...).
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import data_loader  # noqa: E402
import features  # noqa: E402
import preprocessing  # noqa: E402
import splitting  # noqa: E402
import targets  # noqa: E402
from evaluation import evaluate_classifier  # noqa: E402

PROJECT_ROOT = SRC_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "startup_data.csv"
RESULTS_PATH = PROJECT_ROOT / "reports" / "cs23_ensemble_results.json"
MARKDOWN_PATH = PROJECT_ROOT / "reports" / "cs23_ensemble_report.md"
CLASSICAL_RESULTS_PATH = PROJECT_ROOT / "reports" / "cs23_classical_results.json"

# Valeurs figees pour la reproductibilite.
MAX_YEAR = 2026
RANDOM_STATE = 42
CV_FOLDS = 5
CV_SCORING = "f1_macro"
CV_SECONDARY_SCORING = "balanced_accuracy"

# XGBoost : profondeur et nombre d'arbres sont les leviers principaux ;
# ``subsample`` et ``colsample_bytree`` ajoutent une regularisation stochastique.
XGBOOST_PARAM_GRID = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0],
}

# AdaBoost : profondeur de l'arbre faible, nombre d'estimateurs et taux
# d'apprentissage. 27 combinaisons, CV 5 folds.
ADABOOST_PARAM_GRID = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.05, 0.1, 0.5],
    "model__estimator__max_depth": [1, 2, 3],
}

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


def _cv_splitter() -> StratifiedKFold:
    """Retourne le decoupage CV commun : 5 folds stratifies, graine 42."""
    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def _secondary_cv(model, X_train, y_train) -> tuple[float, float]:
    """Balanced accuracy en CV du reglage retenu (information, pas selection)."""
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=_cv_splitter(),
        scoring=CV_SECONDARY_SCORING,
    )
    return float(scores.mean()), float(scores.std())


def load_prepared_data(path: str | Path = DATASET_PATH):
    """Charge le CSV de reference et applique la chaine partagee jusqu'a la cible."""
    raw = data_loader.load_raw(path)
    engineered = features.build_features(raw, max_year=MAX_YEAR)
    return targets.prepare_target(engineered)


def _check_protocol(X) -> None:
    """Verifie que X respecte exactement les 6 variables autorisees."""
    expected = list(splitting.INITIAL_FEATURES)
    if list(X.columns) != expected:
        raise ValueError(
            f"Variables d'entree inattendues : {list(X.columns)} (attendu : {expected})."
        )
    forbidden = sorted(set(expected) & splitting.FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(f"Variables interdites : {forbidden}")


def build_xgboost():
    """Retourne le Pipeline partage (pretraitement + XGBClassifier).

    ``xgboost`` est importe paresseusement : le message d'erreur reste explicite
    si la bibliotheque n'est pas installee dans l'environnement.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - depend de l'environnement
        raise RuntimeError(
            "XGBoost n'est pas installe. Installer la dependance avec "
            "`python -m pip install xgboost` avant de lancer ce script."
        ) from exc

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,  # le parallelisme est porte par GridSearchCV
        random_state=RANDOM_STATE,
    )
    return preprocessing.build_pipeline(model, scale=False)


def build_adaboost():
    """Retourne le Pipeline partage (pretraitement + AdaBoost sur arbres)."""
    base = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model = AdaBoostClassifier(
        estimator=base,
        random_state=RANDOM_STATE,
    )
    return preprocessing.build_pipeline(model, scale=False)


def tune_xgboost(X_train, y_train) -> GridSearchCV:
    """Recherche d'hyperparametres XGBoost par CV sur le train uniquement."""
    search = GridSearchCV(
        estimator=build_xgboost(),
        param_grid=XGBOOST_PARAM_GRID,
        scoring=CV_SCORING,
        cv=_cv_splitter(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_adaboost(X_train, y_train) -> GridSearchCV:
    """Recherche d'hyperparametres AdaBoost par CV sur le train uniquement."""
    search = GridSearchCV(
        estimator=build_adaboost(),
        param_grid=ADABOOST_PARAM_GRID,
        scoring=CV_SCORING,
        cv=_cv_splitter(),
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def _run_model(data, tune_fn, label: str) -> dict:
    """Sequence commune : split -> CV sur train -> evaluation unique sur test."""
    X_train, X_test, y_train, y_test = splitting.split_dataset(
        data, random_state=RANDOM_STATE
    )
    _check_protocol(X_train)
    _check_protocol(X_test)

    started = time.perf_counter()
    search = tune_fn(X_train, y_train)
    tune_seconds = time.perf_counter() - started
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
        "label": label,
        "model": model,
        "search": search,
        "best_params": best_params,
        "cv_score": float(search.best_score_),
        "cv_std": float(search.cv_results_["std_test_score"][best_index]),
        "cv_balanced_accuracy": bal_mean,
        "cv_balanced_std": bal_std,
        "n_configurations": int(len(search.cv_results_["params"])),
        "tune_seconds": float(tune_seconds),
        "train_score": float(balanced_accuracy_score(y_train, model.predict(X_train))),
        "features": list(X_train.columns),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "predicted_counts": {
            int(label_): int(count)
            for label_, count in zip(*np.unique(y_pred, return_counts=True))
        },
        "metrics": metrics,
    }


def run_xgboost(data) -> dict:
    """Entraine et evalue XGBoost selon le protocole commun."""
    return _run_model(data, tune_xgboost, "XGBoost")


def run_adaboost(data) -> dict:
    """Entraine et evalue AdaBoost selon le protocole commun."""
    return _run_model(data, tune_adaboost, "AdaBoost")


def _format_metric(value) -> str:
    if value is None:
        return "indisponible"
    return f"{value:.4f}"


def print_model_report(results: dict) -> None:
    """Affiche le reglage retenu et les metriques d'un modele ensemble."""
    print("\n=== Modele ===")
    print(f"{results['label']} (pipeline scale=False)")
    print(f"variables d'entree ({len(results['features'])}) : {results['features']}")
    print(
        f"hyperparametres retenus (CV {CV_FOLDS} folds, {CV_SCORING}) : "
        f"{results['best_params']}"
    )
    print(
        f"{CV_SCORING} en CV (train)           : "
        f"{results['cv_score']:.4f} +/- {results['cv_std']:.4f}"
    )
    print(
        f"{CV_SECONDARY_SCORING} en CV (train)   : "
        f"{results['cv_balanced_accuracy']:.4f} +/- {results['cv_balanced_std']:.4f}"
    )
    print(f"balanced accuracy sur le train  : {results['train_score']:.4f}")
    print(f"predictions sur le test          : {results['predicted_counts']}")

    metrics = results["metrics"]
    print(f"\n=== Metriques (jeu de test, {results['n_test']} observations) ===")
    for key, label in METRIC_LABELS:
        print(f"{label:<32} {_format_metric(metrics.get(key))}")
    print(f"{'ROC-AUC':<32} {_format_metric(metrics.get('roc_auc'))}")

    matrix = metrics["confusion_matrix"]
    print("\nMatrice de confusion (lignes = reel, colonnes = predit) :")
    print("            pred 0   pred 1")
    print(f"reel 0      {matrix[0][0]:>6}   {matrix[0][1]:>6}")
    print(f"reel 1      {matrix[1][0]:>6}   {matrix[1][1]:>6}")


def _load_classical_baselines() -> dict:
    """Recupere les resultats classiques deja publies, pour mise en perspective."""
    if not CLASSICAL_RESULTS_PATH.is_file():
        return {}
    try:
        payload = json.loads(CLASSICAL_RESULTS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        name: {
            "cv_f1_macro": values.get("cv_score"),
            "test_f1_macro": values.get("test", {}).get("f1_macro"),
            "test_roc_auc": values.get("test", {}).get("roc_auc"),
        }
        for name, values in payload.get("models", {}).items()
    }


def build_payload(models: dict, data) -> dict:
    """Construit le rapport machine-readable (aucune identite personnelle)."""
    import joblib
    import pandas as pd
    import sklearn

    _, _, y_train, _ = splitting.split_dataset(data, random_state=RANDOM_STATE)
    reference = next(iter(models.values()))
    return {
        "generated_by": "CS23",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "data_path": "data/raw/startup_data.csv",
            "features": list(splitting.INITIAL_FEATURES),
            "excluded": ["founding_month"],
            "scoring": CV_SCORING,
            "secondary_scoring": CV_SECONDARY_SCORING,
            "cv_folds": CV_FOLDS,
            "test_size": 0.20,
            "random_state": RANDOM_STATE,
            "max_year": MAX_YEAR,
            "scale": {"xgboost": False, "adaboost": False},
            "tuning_on_train_only": True,
            "test_used_once": True,
        },
        "dataset": {
            "raw_rows": int(len(data_loader.load_raw(DATASET_PATH))),
            "kept_rows": int(len(data)),
            "train_rows": int(reference["n_train"]),
            "test_rows": int(reference["n_test"]),
            "positive_share_train": float(y_train.mean()),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "xgboost": _xgboost_version(),
        },
        "models": {
            name: {
                "best_params": results["best_params"],
                "cv_score": results["cv_score"],
                "cv_std": results["cv_std"],
                "cv_balanced_accuracy": results["cv_balanced_accuracy"],
                "cv_balanced_std": results["cv_balanced_std"],
                "n_configurations": results["n_configurations"],
                "tune_seconds": results["tune_seconds"],
                "train_score": results["train_score"],
                "n_train": results["n_train"],
                "n_test": results["n_test"],
                "test": results["metrics"],
            }
            for name, results in models.items()
        },
        "comparison": {
            "classical_baselines": _load_classical_baselines(),
            "best_ensemble_by_cv": max(
                models, key=lambda name: models[name]["cv_score"]
            ),
            "best_ensemble_by_test": max(
                models, key=lambda name: models[name]["metrics"]["f1_macro"]
            ),
        },
    }


def _xgboost_version() -> str | None:
    """Version de XGBoost, ou None si la bibliotheque est absente."""
    try:
        import xgboost

        return str(xgboost.__version__)
    except ImportError:
        return None


def _fmt(value, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_markdown(payload: dict) -> str:
    """Genere le rapport lisible a partir du meme payload que le JSON."""
    proto = payload["protocol"]
    data = payload["dataset"]
    versions = payload["versions"]
    lines = [
        "# Resultats - modeles ensemble (XGBoost et AdaBoost)",
        "",
        "## Protocole",
        "",
        f"- Donnees : `{proto['data_path']}` (sans ISR), chemin explicite",
        "- Variables : " + ", ".join(proto["features"]),
        "- Exclue : founding_month (EDA uniquement) ; aucune variable de financement",
        f"- Validation croisee : StratifiedKFold({proto['cv_folds']}, shuffle=True, "
        f"random_state={proto['random_state']})",
        f"- Critere de selection unique : `{proto['scoring']}`",
        f"- Critere secondaire (information seulement) : `{proto['secondary_scoring']}`",
        f"- Decoupage : 80 / 20 stratifie, graine {proto['random_state']}",
        "- Pipeline partage : `build_pipeline(model, scale=False)` pour les deux modeles",
        "- Hyperparametres choisis sur le train uniquement ; test utilise une seule fois",
        "",
        "## Donnees",
        "",
        f"- Lignes brutes : {data['raw_rows']}",
        f"- Lignes retenues : {data['kept_rows']}",
        f"- Train / test : {data['train_rows']} / {data['test_rows']}",
        f"- Part de classe 1 (train) : {100 * data['positive_share_train']:.1f} %",
        "",
        "## Resultats",
        "",
        "| Modele | Meilleurs parametres | F1 macro (CV) | Bal. acc. (CV) | "
        "F1 macro (test) | Bal. acc. (test) | Accuracy (test) | ROC-AUC (test) | Duree (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in payload["models"].items():
        test = res["test"]
        params = ", ".join(f"{k}={v}" for k, v in res["best_params"].items())
        lines.append(
            f"| {name} | {params} "
            f"| {_fmt(res['cv_score'])} +/- {_fmt(res['cv_std'])} "
            f"| {_fmt(res['cv_balanced_accuracy'])} +/- {_fmt(res['cv_balanced_std'])} "
            f"| {_fmt(test['f1_macro'])} | {_fmt(test['balanced_accuracy'])} "
            f"| {_fmt(test['accuracy'])} | {_fmt(test['roc_auc'])} "
            f"| {_fmt(res['tune_seconds'], 1)} |"
        )

    lines += ["", "## Comparaison", ""]
    comparison = payload["comparison"]
    lines.append(
        f"- Meilleur modele ensemble en CV ({proto['scoring']}) : "
        f"**{comparison['best_ensemble_by_cv']}**"
    )
    lines.append(
        f"- Meilleur modele ensemble sur le test (F1 macro) : "
        f"**{comparison['best_ensemble_by_test']}**"
    )
    baselines = comparison["classical_baselines"]
    if baselines:
        lines += ["", "### Mise en perspective (modeles classiques de CS23)", "",
                  "| Modele | F1 macro (CV) | F1 macro (test) | ROC-AUC (test) |",
                  "|---|---|---|---|"]
        for name, values in baselines.items():
            lines.append(
                f"| {name} | {_fmt(values['cv_f1_macro'])} "
                f"| {_fmt(values['test_f1_macro'])} | {_fmt(values['test_roc_auc'])} |"
            )

    lines += ["", "## Versions", ""]
    for key, value in versions.items():
        lines.append(f"- {key} : {value if value is not None else 'non disponible'}")

    lines += [
        "",
        "## Rappel methodologique",
        "",
        "- Les hyperparametres sont choisis uniquement sur le train, en CV 5 folds.",
        "- Le jeu de test n'est utilise qu'une seule fois, apres la CV.",
        "- Le preprocesseur est ajuste dans chaque fold : aucune fuite de donnees.",
        "- Ces scores mesurent la discrimination sur un jeu de donnees historique ; "
        "ils ne constituent pas une promesse de succes futur.",
        "",
    ]
    return "\n".join(lines)


def export_results(
    payload: dict,
    path: Path = RESULTS_PATH,
    markdown_path: Path = MARKDOWN_PATH,
) -> Path:
    """Ecrit le JSON et le rapport Markdown (tous deux au meme emplacement)."""
    path = Path(path)
    markdown_path = Path(markdown_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(format_markdown(payload), encoding="utf-8")
    return path


def main() -> int:
    """Point d'entree en ligne de commande."""
    if not DATASET_PATH.is_file():
        print(f"Dataset introuvable : {DATASET_PATH}")
        return 1

    print(f"Dataset : {DATASET_PATH}")
    data = load_prepared_data()
    print(f"Lignes retenues : {len(data)}")

    xgb_results = run_xgboost(data)
    print_model_report(xgb_results)

    ada_results = run_adaboost(data)
    print_model_report(ada_results)

    payload = build_payload(
        {"xgboost": xgb_results, "adaboost": ada_results},
        data,
    )
    json_path = export_results(payload)
    print(f"\nComparaison : {payload['comparison']['best_ensemble_by_cv']} (CV)")
    print(f"Rapport JSON : {json_path}")
    print(f"Rapport Markdown : {MARKDOWN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
