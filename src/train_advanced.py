"""Modeles avances (Random Forest et SVM) alignes sur le protocole commun.

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales (mini-projet Data Mining).

Role de ce module
-----------------
- Construire les ``Pipeline`` Random Forest et SVM a partir du preprocesseur
  commun de l'equipe (``src/preprocessing.py``) : aucune duplication de
  l'imputation, de l'encodage ou de la standardisation.
- Definir les grilles d'hyperparametres et les objets ``GridSearchCV``
  (validation croisee stratifiee a 5 folds, score ``f1_macro``).
- Charger les donnees par la chaine officielle, entrainer les deux modeles et
  evaluer leur performance sur le jeu de test avec ``src/evaluation.py``.

Chaine officielle respectee
---------------------------
::

    data_loader.load_raw()
      -> features.build_features(max_year=2026)
      -> targets.prepare_target()
      -> splitting.split_dataset()          # 6 variables, split stratifie 80/20
      -> preprocessing.build_pipeline()     # imputation + one-hot + scaling
      -> RandomForestClassifier / SVC

Protocole de variables
----------------------
Les variables d'entree sont exactement ``splitting.INITIAL_FEATURES`` :
``country_code``, ``state_code``, ``region``, ``city``, ``primary_category``,
``founding_year``. ``founding_month`` est calcule par ``features.py`` pour
l'EDA mais **exclu** des modeles (artefact de precision de date).

Reglages
--------
- ``scale=False`` pour Random Forest (les arbres sont insensibles a l'echelle).
- ``scale=True`` pour SVM (noyau RBF et variables numeriques sensibles a
  l'echelle).
- ``min_frequency=preprocessing.DEFAULT_MIN_FREQUENCY`` (20) pour les deux
  modeles, afin que les colonnes one-hot soient identiques d'un modele a
  l'autre et comparables aux autres travaux de l'equipe.

Usage
-----
::

    python src/train_advanced.py                    # affiche la configuration
    python src/train_advanced.py --run              # entraine et evalue
    python src/train_advanced.py --run --limit 3000 # test rapide
    python src/train_advanced.py --run --output reports/bl_advanced_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Rendre les modules de src/ importables, que le fichier soit execute
# directement (python src/train_advanced.py) ou importe comme package.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import data_loader
import features
import preprocessing
import splitting
import targets
from evaluation import evaluate_classifier

# ---------------------------------------------------------------------------
# Constantes du protocole commun
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
SCORING: str = "f1_macro"
CV_FOLDS: int = 5

# Seuil de regroupement des modalites rares : on reprend la valeur de l'equipe
# (20) au lieu de la valeur provisoire 10 utilisee avant l'alignement.
MIN_FREQUENCY: int | None = preprocessing.DEFAULT_MIN_FREQUENCY

# Variables d'entree : une seule source de verite, celle de SA.
FEATURE_COLUMNS: tuple[str, ...] = tuple(splitting.INITIAL_FEATURES)

DEFAULT_DATA_PATH = Path("data/raw/startup_data.csv")
DEFAULT_MAX_YEAR: int = 2026


# ---------------------------------------------------------------------------
# A. Verification du protocole
# ---------------------------------------------------------------------------


def check_protocol() -> None:
    """Verifie que le preprocessing et le split utilisent les memes variables.

    Leve
    ----
    RuntimeError
        Si ``preprocessing.FEATURE_COLUMNS`` diverge de
        ``splitting.INITIAL_FEATURES`` : les deux modules doivent rester
        synchronises, sinon le ``ColumnTransformer`` reclame une colonne
        absente de ``X``.
    """
    preprocessing_features = tuple(preprocessing.FEATURE_COLUMNS)
    if preprocessing_features != FEATURE_COLUMNS:
        raise RuntimeError(
            "Protocole incoherent entre preprocessing et splitting : "
            f"{preprocessing_features} != {FEATURE_COLUMNS}"
        )


# ---------------------------------------------------------------------------
# B. Pipelines (preprocessing commun + modele)
# ---------------------------------------------------------------------------


def build_random_forest_pipeline(**estimator_params: Any) -> Pipeline:
    """Assemble le ``Pipeline`` Random Forest (non entraine).

    Parametres
    ----------
    **estimator_params :
        Parametres transmis a ``RandomForestClassifier``. Valeurs par defaut :
        ``random_state=42``, ``class_weight="balanced"``, ``n_jobs=1``.

    Retour
    ------
    Pipeline
        Etapes nommees ``"preprocessor"`` et ``"model"`` (contrat de
        ``preprocessing.build_pipeline``). Les hyperparametres du modele sont
        donc prefixes par ``model__`` dans les grilles.
    """
    params: dict[str, Any] = {
        "random_state": RANDOM_STATE,
        "class_weight": "balanced",
        # Mono-thread : le parallelisme est porte par GridSearchCV, evitant la
        # sur-souscription CPU (plusieurs essais x plusieurs coeurs par essai).
        "n_jobs": 1,
    }
    params.update(estimator_params)
    return preprocessing.build_pipeline(
        RandomForestClassifier(**params),
        scale=False,
        min_frequency=MIN_FREQUENCY,
    )


def build_svm_pipeline(**estimator_params: Any) -> Pipeline:
    """Assemble le ``Pipeline`` SVM (non entraine).

    Parametres
    ----------
    **estimator_params :
        Parametres transmis a ``SVC``. Valeurs par defaut : ``kernel="rbf"``,
        ``C=1.0``, ``gamma="scale"``, ``class_weight="balanced"``,
        ``random_state=42``.

    Retour
    ------
    Pipeline
        Etapes ``"preprocessor"`` (avec standardisation) et ``"model"``.

    Notes
    -----
    ``probability`` n'est pas transmis : il est deprecie depuis scikit-learn
    1.9. Le ROC-AUC est donc calcule depuis ``decision_function`` par
    ``src.evaluation.evaluate_classifier``.
    """
    params: dict[str, Any] = {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    params.update(estimator_params)
    return preprocessing.build_pipeline(
        SVC(**params),
        scale=True,
        min_frequency=MIN_FREQUENCY,
    )


# ---------------------------------------------------------------------------
# C. Grilles d'hyperparametres
# ---------------------------------------------------------------------------


def get_random_forest_param_grid() -> dict[str, list[Any]]:
    """Grille Random Forest : 8 combinaisons (40 ajustements en 5 folds).

    ``n_estimators`` : 200, 400
    ``max_depth`` : None, 20
    ``min_samples_leaf`` : 1, 2

    ``clf__n_jobs`` n'apparait pas : la foret reste mono-thread, le
    parallelisme etant porte par ``GridSearchCV``.
    """
    return {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 20],
        "model__min_samples_leaf": [1, 2],
    }


def get_svm_param_grid() -> list[dict[str, list[Any]]]:
    """Grille SVM : 5 combinaisons (25 ajustements en 5 folds).

    - ``kernel="linear"`` : ``C`` = 0.1, 1, 10
    - ``kernel="rbf"`` : ``C`` = 1, 10 ; ``gamma="scale"``

    Notes
    -----
    Les noyaux polynomiaux sont volontairement omis : sur 10 000 lignes et
    plusieurs centaines de colonnes one-hot, leur cout d'ajustement est tres
    superieur aux gains attendus. Cette limitation est documentee dans le
    rapport de resultats.
    """
    return [
        {"model__kernel": ["linear"], "model__C": [0.1, 1.0, 10.0]},
        {"model__kernel": ["rbf"], "model__C": [1.0, 10.0], "model__gamma": ["scale"]},
    ]


def build_cross_validation(
    n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """Validation croisee stratifiee a ``n_splits`` folds (5 par defaut)."""
    if n_splits < 5:
        raise ValueError("Le protocole impose au moins 5 folds de validation.")
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )


# ---------------------------------------------------------------------------
# D. Objets de recherche (non entraines)
# ---------------------------------------------------------------------------


def build_random_forest_search(
    pipeline: Pipeline | None = None,
    param_grid: Mapping[str, Iterable[Any]] | None = None,
    cv: StratifiedKFold | None = None,
    scoring: str = SCORING,
    n_jobs: int = -1,
    verbose: int = 0,
    error_score: str | float = "raise",
) -> GridSearchCV:
    """Prepare la recherche Random Forest (objet non entraine)."""
    if pipeline is None:
        pipeline = build_random_forest_pipeline()
    return GridSearchCV(
        estimator=pipeline,
        param_grid=(
            dict(param_grid)
            if param_grid is not None
            else get_random_forest_param_grid()
        ),
        cv=cv if cv is not None else build_cross_validation(),
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        error_score=error_score,
        verbose=verbose,
    )


def build_svm_search(
    pipeline: Pipeline | None = None,
    param_grid: Sequence[Mapping[str, Iterable[Any]]] | None = None,
    cv: StratifiedKFold | None = None,
    scoring: str = SCORING,
    n_jobs: int = 1,
    verbose: int = 0,
    error_score: str | float = "raise",
) -> GridSearchCV:
    """Prepare la recherche SVM (objet non entraine).

    ``n_jobs=1`` par defaut : un ajustement SVC alloue un cache memoire
    important, et plusieurs essais simultanes peuvent saturer la RAM.
    """
    if pipeline is None:
        pipeline = build_svm_pipeline()
    grid = list(param_grid) if param_grid is not None else get_svm_param_grid()
    return GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        cv=cv if cv is not None else build_cross_validation(),
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        error_score=error_score,
        verbose=verbose,
    )


def prepare_experiment(
    n_splits: int = CV_FOLDS,
    scoring: str = SCORING,
    rf_n_jobs: int = -1,
    svm_n_jobs: int = 1,
    verbose: int = 0,
    error_score: str | float = "raise",
) -> dict[str, GridSearchCV]:
    """Prepare les deux recherches sans les entrainer.

    Retour
    ------
    dict[str, GridSearchCV]
        ``{"random_forest": search_rf, "svm": search_svm}``. Les objets sont
        NON ENTRAINES ; apres ``fit``, le modele optimise est
        ``search.best_estimator_``.
    """
    check_protocol()
    cross_validation = build_cross_validation(n_splits=n_splits)
    return {
        "random_forest": build_random_forest_search(
            pipeline=build_random_forest_pipeline(),
            cv=cross_validation,
            scoring=scoring,
            n_jobs=rf_n_jobs,
            verbose=verbose,
            error_score=error_score,
        ),
        "svm": build_svm_search(
            pipeline=build_svm_pipeline(),
            cv=cross_validation,
            scoring=scoring,
            n_jobs=svm_n_jobs,
            verbose=verbose,
            error_score=error_score,
        ),
    }


# ---------------------------------------------------------------------------
# E. Chargement des donnees par la chaine officielle
# ---------------------------------------------------------------------------


def load_dataset(
    path: str | Path = DEFAULT_DATA_PATH,
    max_year: int = DEFAULT_MAX_YEAR,
    limit: int | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Charge et separe les donnees avec la chaine officielle de l'equipe.

    Parametres
    ----------
    path : str | Path
        CSV brut (``data/raw/startup_data.csv`` par defaut).
    max_year : int
        Annee maximale plausible pour ``features.build_features``.
    limit : int | None
        Sous-echantillon aleatoire (test rapide). ``None`` = dataset complet.

    Retour
    ------
    (X_train, X_test, y_train, y_test, audit)
        ``audit`` contient les effectifs reels, pour verification.
    """
    raw = data_loader.load_raw(path)
    enriched = features.build_features(raw, max_year=max_year)
    prepared = targets.prepare_target(enriched)

    if limit is not None:
        if limit < 100:
            raise ValueError("--limit doit valoir au moins 100 lignes.")
        if limit < len(prepared):
            prepared = prepared.sample(n=int(limit), random_state=RANDOM_STATE)

    X_train, X_test, y_train, y_test = splitting.split_dataset(
        prepared, features=FEATURE_COLUMNS
    )

    audit = {
        "raw_rows": int(len(raw)),
        "kept_rows": int(len(prepared)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_share": float(prepared["target"].mean()),
        "status_counts": {
            str(status): int(count)
            for status, count in prepared["status"].value_counts().items()
        },
    }
    return X_train, X_test, y_train, y_test, audit


# ---------------------------------------------------------------------------
# F. Entrainement et evaluation
# ---------------------------------------------------------------------------


def fit_and_evaluate(
    name: str,
    search: GridSearchCV,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
) -> dict[str, Any]:
    """Entraine ``search`` sur le train puis evalue le meilleur modele sur le test.

    Le jeu de test n'est utilise qu'une seule fois, apres la validation
    croisee : aucun choix d'hyperparametre ne depend de lui.
    """
    print(f"[{name}] ajustement en cours ({len(search.param_grid)} grilles x 5 folds)...", flush=True)
    start = time.perf_counter()
    search.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    best = search.best_estimator_
    metrics = evaluate_classifier(best, X_test, y_test)
    print(
        f"[{name}] termine en {fit_seconds:.1f}s | "
        f"CV {search.scoring}={search.best_score_:.4f} | "
        f"test f1_macro={metrics['f1_macro']:.4f} | "
        f"test roc_auc={metrics['roc_auc']}",
        flush=True,
    )
    return {
        "best_params": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in search.best_params_.items()
        },
        "best_cv_score": float(search.best_score_),
        "scoring": search.scoring,
        "fit_seconds": round(float(fit_seconds), 2),
        "test": metrics,
    }


def run_experiment(
    path: str | Path = DEFAULT_DATA_PATH,
    max_year: int = DEFAULT_MAX_YEAR,
    limit: int | None = None,
    only: str = "both",
    rf_n_jobs: int = -1,
    svm_n_jobs: int = 1,
    verbose: int = 0,
) -> dict[str, Any]:
    """Execute l'experience complete : donnees, CV, evaluation sur le test.

    Retour
    ------
    dict
        Resultats serialisables en JSON (protocole, donnees, modeles).
    """
    check_protocol()
    X_train, X_test, y_train, y_test, audit = load_dataset(
        path=path, max_year=max_year, limit=limit
    )
    print(
        f"Donnees : {audit['raw_rows']} brutes -> {audit['kept_rows']} retenues | "
        f"train={audit['train_rows']} test={audit['test_rows']} | "
        f"classe 1={audit['positive_share']:.1%}",
        flush=True,
    )

    searches = prepare_experiment(
        rf_n_jobs=rf_n_jobs,
        svm_n_jobs=svm_n_jobs,
        verbose=verbose,
    )
    selected = {
        key: value for key, value in searches.items() if only in ("both", key)
    }

    models: dict[str, Any] = {}
    for name, search in selected.items():
        models[name] = fit_and_evaluate(
            name, search, X_train, y_train, X_test, y_test
        )

    return {
        "protocol": {
            "features": list(FEATURE_COLUMNS),
            "excluded": list(features.EDA_ONLY_COLUMNS),
            "min_frequency": MIN_FREQUENCY,
            "scoring": SCORING,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "max_year": max_year,
            "limit": limit,
            "data_path": str(path),
        },
        "dataset": audit,
        "models": models,
    }


# ---------------------------------------------------------------------------
# G. Restitution
# ---------------------------------------------------------------------------


def format_markdown(results: Mapping[str, Any]) -> str:
    """Met en forme les resultats en tableau Markdown copiable."""
    protocol = results["protocol"]
    dataset = results["dataset"]
    lines = [
        "# Resultats - modeles avances (Random Forest et SVM)",
        "",
        "## Protocole",
        "",
        f"- Variables : {', '.join(protocol['features'])}",
        f"- Exclues : {', '.join(protocol['excluded'])} (EDA uniquement)",
        f"- `min_frequency` : {protocol['min_frequency']}",
        f"- Validation croisee : {protocol['cv_folds']} folds stratifies, "
        f"score `{protocol['scoring']}`",
        "",
        "## Donnees",
        "",
        f"- Lignes brutes : {dataset['raw_rows']}",
        f"- Lignes retenues : {dataset['kept_rows']}",
        f"- Train / test : {dataset['train_rows']} / {dataset['test_rows']}",
        f"- Part de classe 1 : {dataset['positive_share']:.1%}",
        "",
        "## Resultats",
        "",
        "| Modele | Meilleurs parametres | F1 macro (CV) | F1 macro (test) | "
        "ROC-AUC (test) | Accuracy (test) | Duree (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, payload in results["models"].items():
        test = payload["test"]
        params = ", ".join(
            f"{key.replace('model__', '')}={value}"
            for key, value in payload["best_params"].items()
        )
        roc_auc = "n/a" if test["roc_auc"] is None else f"{test['roc_auc']:.4f}"
        lines.append(
            f"| {name} | {params} | {payload['best_cv_score']:.4f} | "
            f"{test['f1_macro']:.4f} | {roc_auc} | {test['accuracy']:.4f} | "
            f"{payload['fit_seconds']:.1f} |"
        )

    lines += [
        "",
        "## Rappel methodologique",
        "",
        "- Les hyperparametres sont choisis uniquement sur le train, en CV 5 folds.",
        "- Le jeu de test n'est utilise qu'une fois, apres la CV.",
        "- Le preprocesseur est ajuste dans chaque fold : aucune fuite de donnees.",
        "- Les scores ne sont pas des probabilites de succes futur : ce sont des "
        "mesures de discrimination sur ce jeu de donnees historique.",
        "",
    ]
    return "\n".join(lines)


def save_results(results: Mapping[str, Any], output: str | Path) -> Path:
    """Ecrit le JSON et le rapport Markdown a cote du chemin demande."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    markdown = output.with_suffix(".md")
    markdown.write_text(format_markdown(results), encoding="utf-8")
    return output


# ---------------------------------------------------------------------------
# H. Point d'entree
# ---------------------------------------------------------------------------


def _print_configuration() -> None:
    """Affiche la configuration sans entrainer (mode par defaut)."""
    check_protocol()
    print(f"Variables ({len(FEATURE_COLUMNS)}) : {', '.join(FEATURE_COLUMNS)}")
    print(f"Exclues (EDA) : {', '.join(features.EDA_ONLY_COLUMNS)}")
    print(f"min_frequency={MIN_FREQUENCY} | CV={CV_FOLDS} folds | scoring={SCORING}")
    for name, search in prepare_experiment().items():
        print(f"[{name}] etapes={list(search.estimator.named_steps)}")
        print(f"[{name}] grille={search.param_grid}")
    print("Configuration prete. Aucun entrainement lance (utiliser --run).")


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="entrainer et evaluer")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="CSV brut")
    parser.add_argument("--limit", type=int, default=None, help="sous-echantillon")
    parser.add_argument(
        "--only", choices=["both", "random_forest", "svm"], default="both"
    )
    parser.add_argument("--output", default=None, help="chemin du JSON de resultats")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.run:
        _print_configuration()
        return 0

    results = run_experiment(
        path=args.data,
        limit=args.limit,
        only=args.only,
        verbose=args.verbose,
    )
    print()
    print(format_markdown(results))
    if args.output:
        saved = save_results(results, args.output)
        print(f"\nResultats ecrits dans : {saved}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
