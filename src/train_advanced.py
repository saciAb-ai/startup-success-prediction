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
    python src/train_advanced.py --run --output reports/advanced_models_results.json
    python src/train_advanced.py --importance       # variables importantes (RF)
    python src/train_advanced.py --robustness       # variation des folds
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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
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

# Resultats de reference (produits par --run) : ils contiennent les meilleurs
# hyperparametres reutilises par --importance et --robustness.
DEFAULT_RESULTS_PATH = Path("reports/advanced_models_results.json")
DEFAULT_IMPORTANCE_PATH = Path("reports/feature_importance.json")
DEFAULT_ROBUSTNESS_PATH = Path("reports/robustness.json")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)


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
# G. Analyses complementaires (independantes de la revue de SA)
# ---------------------------------------------------------------------------


def build_pipeline_from_params(
    model_name: str, params: Mapping[str, Any] | None = None
) -> Pipeline:
    """Reconstruit un ``Pipeline`` a partir de parametres eventuellement prefixes.

    Les cles peuvent venir telles quelles de ``GridSearchCV.best_params_``
    (prefixe ``model__``) ou du JSON de resultats : le prefixe est retire avant
    d'appeler ``RandomForestClassifier`` / ``SVC``.

    Parametres
    ----------
    model_name : str
        ``"random_forest"`` ou ``"svm"``.
    params : Mapping[str, Any] | None
        Hyperparametres du modele. ``None`` = valeurs par defaut du module.

    Retour
    ------
    Pipeline
        Pipeline NON ENTRAINE, identique au protocole commun.
    """
    cleaned: dict[str, Any] = {}
    for key, value in (params or {}).items():
        cleaned[key.replace("model__", "")] = value

    if model_name == "random_forest":
        return build_random_forest_pipeline(**cleaned)
    if model_name == "svm":
        return build_svm_pipeline(**cleaned)
    raise ValueError(
        f"Modele inconnu : {model_name!r} (attendu : 'random_forest' ou 'svm')."
    )


def load_best_params(
    path: str | Path = DEFAULT_RESULTS_PATH,
) -> dict[str, dict[str, Any]]:
    """Lit les meilleurs hyperparametres du JSON produit par ``--run``.

    Retour
    ------
    dict[str, dict[str, Any]]
        ``{"random_forest": {...}, "svm": {...}}``. Dictionnaire vide si le
        fichier est absent : les analyses retombent alors sur les valeurs par
        defaut du module.
    """
    path = Path(path)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    models = payload.get("models", {})
    return {
        name: dict(model.get("best_params", {}))
        for name, model in models.items()
    }


def _source_feature(column: str) -> str:
    """Retrouve la variable d'origine d'une colonne one-hot.

    Exemples : ``cat__city_Paris`` -> ``city`` ; ``num__founding_year`` ->
    ``founding_year``. Les noms de variables contiennent des underscores, donc
    on compare aux colonnes connues en commencant par les plus longues.
    """
    if column.startswith("num__"):
        return column[len("num__"):]
    body = column[len("cat__"):] if column.startswith("cat__") else column
    for feature_name in sorted(FEATURE_COLUMNS, key=len, reverse=True):
        if body == feature_name or body.startswith(feature_name + "_"):
            return feature_name
    return body


def extract_feature_importances(
    fitted_pipeline: Pipeline, top_n: int = 20
) -> dict[str, Any]:
    """Extrait l'importance des variables d'un Random Forest entraine.

    Deux lectures sont fournies :

    - ``top_columns`` : les ``top_n`` colonnes one-hot les plus importantes ;
    - ``grouped_by_source`` : la somme des importances par variable d'origine
      (les 6 variables du protocole), plus lisible qu'un classement de 300
      modalites.

    Limite methodologique
    ---------------------
    ``feature_importances_`` mesure la baisse d'impurete dans l'arbre. Cette
    mesure favorise les variables a forte cardinalite (``city``, ``region``) :
    elle decrit le modele, elle ne prouve pas une relation causale.
    """
    if top_n < 1:
        raise ValueError("top_n doit valoir au moins 1.")

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    model = fitted_pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            "Ce modele n'expose pas feature_importances_ "
            f"({type(model).__name__}). Utiliser un Random Forest."
        )

    names = preprocessing.feature_names_out(preprocessor)
    values = np.asarray(model.feature_importances_, dtype=float)
    if len(names) != len(values):
        raise RuntimeError(
            f"Incoherence : {len(names)} colonnes pour {len(values)} importances."
        )

    order = np.argsort(values)[::-1]
    top_columns = [
        {
            "column": names[index],
            "source_feature": _source_feature(names[index]),
            "importance": float(values[index]),
        }
        for index in order[:top_n]
    ]

    grouped: dict[str, float] = {feature: 0.0 for feature in FEATURE_COLUMNS}
    for name, value in zip(names, values):
        source = _source_feature(name)
        grouped[source] = grouped.get(source, 0.0) + float(value)
    grouped_by_source = sorted(
        (
            {"source_feature": source, "importance": round(value, 6)}
            for source, value in grouped.items()
        ),
        key=lambda row: row["importance"],
        reverse=True,
    )

    return {
        "n_columns": len(names),
        "top_columns": top_columns,
        "grouped_by_source": grouped_by_source,
        "impurity_total": float(values.sum()),
    }


def run_importance(
    path: str | Path = DEFAULT_DATA_PATH,
    max_year: int = DEFAULT_MAX_YEAR,
    limit: int | None = None,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    top_n: int = 20,
) -> dict[str, Any]:
    """Entraine le Random Forest sur le train et analyse ses importances.

    Le jeu de test n'est PAS utilise : cette analyse est descriptive et ne doit
    pas influencer le choix des hyperparametres.
    """
    check_protocol()
    X_train, _X_test, y_train, _y_test, audit = load_dataset(
        path=path, max_year=max_year, limit=limit
    )
    params = load_best_params(results_path).get("random_forest")
    described = (
        ", ".join(
            f"{key.replace('model__', '')}={value}"
            for key, value in params.items()
        )
        if params
        else "parametres par defaut"
    )
    print(f"[importance] Random Forest ({described})", flush=True)
    pipeline = build_pipeline_from_params("random_forest", params)
    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start
    importance = extract_feature_importances(pipeline, top_n=top_n)
    print(
        f"[importance] termine en {fit_seconds:.1f}s | "
        f"{importance['n_columns']} colonnes analysees",
        flush=True,
    )
    return {
        "protocol": {
            "features": list(FEATURE_COLUMNS),
            "excluded": list(features.EDA_ONLY_COLUMNS),
            "min_frequency": MIN_FREQUENCY,
            "scoring": SCORING,
            "random_state": RANDOM_STATE,
            "max_year": max_year,
            "limit": limit,
            "data_path": str(path),
        },
        "dataset": audit,
        "used_best_params": params or {},
        "fit_seconds": round(float(fit_seconds), 2),
        "importance": importance,
    }


def compute_robustness(
    model_name: str,
    X: Any,
    y: Any,
    params: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_splits: int = CV_FOLDS,
    scoring: str = SCORING,
    n_jobs: int = 1,
) -> dict[str, Any]:
    """Mesure la variation du score quand le decoupage des folds change.

    Le protocole n'est pas modifie : seules les graines de ``StratifiedKFold``
    varient. Chaque graine rejoue donc la meme validation croisee 5 folds, ce
    qui donne une idee de la stabilite du score (et non un intervalle de
    confiance statistique).
    """
    if not seeds:
        raise ValueError("Fournir au moins une graine.")
    pipeline = build_pipeline_from_params(model_name, params)
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cross_validation = build_cross_validation(
            n_splits=n_splits, random_state=int(seed)
        )
        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cross_validation,
            scoring=scoring,
            n_jobs=n_jobs,
            error_score="raise",
        )
        rows.append(
            {
                "seed": int(seed),
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "folds": [float(score) for score in scores],
            }
        )

    means = [row["mean"] for row in rows]
    return {
        "model": model_name,
        "params": {
            key.replace("model__", ""): value
            for key, value in (params or {}).items()
        }
        or "defaults",
        "scoring": scoring,
        "n_splits": int(n_splits),
        "seeds": rows,
        "overall_mean": float(np.mean(means)),
        "overall_std": float(np.std(means)),
        "overall_min": float(np.min(means)),
        "overall_max": float(np.max(means)),
    }


def run_robustness(
    path: str | Path = DEFAULT_DATA_PATH,
    max_year: int = DEFAULT_MAX_YEAR,
    limit: int | None = None,
    models: Sequence[str] = ("random_forest", "svm"),
    seeds: Sequence[int] = DEFAULT_SEEDS,
    results_path: str | Path = DEFAULT_RESULTS_PATH,
    rf_n_jobs: int = -1,
    svm_n_jobs: int = 1,
) -> dict[str, Any]:
    """Rejoue la validation croisee sur le train avec plusieurs graines.

    Le jeu de test n'est PAS utilise : la robustesse se mesure uniquement sur
    le train, conformement au protocole.
    """
    check_protocol()
    X_train, _X_test, y_train, _y_test, audit = load_dataset(
        path=path, max_year=max_year, limit=limit
    )
    params_by_model = load_best_params(results_path)

    payload: dict[str, Any] = {}
    for model_name in models:
        print(
            f"[robustesse] {model_name} : {len(seeds)} graines x "
            f"{CV_FOLDS} folds...",
            flush=True,
        )
        start = time.perf_counter()
        payload[model_name] = compute_robustness(
            model_name,
            X_train,
            y_train,
            params=params_by_model.get(model_name),
            seeds=seeds,
            n_jobs=rf_n_jobs if model_name == "random_forest" else svm_n_jobs,
        )
        elapsed = time.perf_counter() - start
        payload[model_name]["fit_seconds"] = round(float(elapsed), 2)
        print(
            f"[robustesse] {model_name} termine en {elapsed:.1f}s | "
            f"moyenne={payload[model_name]['overall_mean']:.4f} "
            f"(min={payload[model_name]['overall_min']:.4f}, "
            f"max={payload[model_name]['overall_max']:.4f})",
            flush=True,
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
            "seeds": [int(seed) for seed in seeds],
        },
        "dataset": audit,
        "models": payload,
    }


def format_importance_markdown(results: Mapping[str, Any]) -> str:
    """Met en forme l'analyse des variables importantes du Random Forest."""
    protocol = results["protocol"]
    dataset = results["dataset"]
    importance = results["importance"]
    lines = [
        "# Importance des variables - Random Forest",
        "",
        "## Protocole",
        "",
        f"- Variables : {', '.join(protocol['features'])}",
        f"- Exclues : {', '.join(protocol['excluded'])} (EDA uniquement)",
        f"- `min_frequency` : {protocol['min_frequency']}",
        f"- Colonnes one-hot analysees : {importance['n_columns']}",
        f"- Entrainement : {dataset['train_rows']} lignes (le test n'est pas utilise)",
        "",
        "## Importance par variable d'origine",
        "",
        "| Variable | Importance cumulee |",
        "|---|---|",
    ]
    for row in importance["grouped_by_source"]:
        lines.append(
            f"| {row['source_feature']} | {row['importance']:.4f} |"
        )
    lines += [
        "",
        "## Principales modalites",
        "",
        "| Rang | Colonne | Variable | Importance |",
        "|---|---|---|---|",
    ]
    for rank, row in enumerate(importance["top_columns"], start=1):
        lines.append(
            f"| {rank} | `{row['column']}` | {row['source_feature']} | "
            f"{row['importance']:.4f} |"
        )
    lines += [
        "",
        "## Rappel methodologique",
        "",
        "- Le jeu de test n'est pas utilise : analyse descriptive sur le train.",
        "- `feature_importances_` mesure la baisse d'impurete dans les arbres ; "
        "elle favorise les variables a forte cardinalite (`city`, `region`).",
        "- Ces importances decrivent le modele, elles ne prouvent pas une "
        "relation causale entre la variable et l'issue de la startup.",
        "",
    ]
    return "\n".join(lines)


def format_robustness_markdown(results: Mapping[str, Any]) -> str:
    """Met en forme la variation des scores selon la graine des folds."""
    protocol = results["protocol"]
    dataset = results["dataset"]
    lines = [
        "# Robustesse des modeles avances (variation des folds)",
        "",
        "## Protocole",
        "",
        f"- Variables : {', '.join(protocol['features'])}",
        f"- `min_frequency` : {protocol['min_frequency']}",
        f"- Score : `{protocol['scoring']}`",
        f"- Validation croisee : {protocol['cv_folds']} folds stratifies",
        f"- Graines testees : {', '.join(str(seed) for seed in protocol['seeds'])}",
        f"- Entrainement : {dataset['train_rows']} lignes (le test n'est pas utilise)",
        "",
        "## Resultats par graine",
        "",
        "| Modele | Graine | F1 macro moyen | Ecart-type | Min | Max |",
        "|---|---|---|---|---|---|",
    ]
    for model_name, payload in results["models"].items():
        for row in payload["seeds"]:
            lines.append(
                f"| {model_name} | {row['seed']} | {row['mean']:.4f} | "
                f"{row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
            )
    lines += [
        "",
        "## Synthese",
        "",
        "| Modele | Moyenne des graines | Ecart-type | Min | Max | Duree (s) |",
        "|---|---|---|---|---|---|",
    ]
    for model_name, payload in results["models"].items():
        lines.append(
            f"| {model_name} | {payload['overall_mean']:.4f} | "
            f"{payload['overall_std']:.4f} | {payload['overall_min']:.4f} | "
            f"{payload['overall_max']:.4f} | {payload['fit_seconds']:.1f} |"
        )
    lines += [
        "",
        "## Rappel methodologique",
        "",
        "- Seules les graines de decoupage changent ; les hyperparametres sont "
        "ceux choisis par `--run` sur le train.",
        "- Le jeu de test n'est pas utilise : la robustesse se mesure sur le train.",
        "- L'ecart observe indique la stabilite du score, ce n'est pas un "
        "intervalle de confiance statistique.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# H. Restitution
# ---------------------------------------------------------------------------

# Libelles lisibles pour les rapports Markdown (les cles du JSON restent
# `random_forest` et `svm`).
MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "svm": "SVM",
}
CLASS_LABELS: dict[int, str] = {
    0: "fermeture",
    1: "IPO ou acquisition",
}


def _model_label(name: str) -> str:
    """Nom d'affichage d'un modele (repli sur la cle JSON si inconnue)."""
    return MODEL_LABELS.get(name, name)


def _class_metrics_from_confusion(
    matrix: Sequence[Sequence[int]],
) -> dict[int, dict[str, float | int]]:
    """Derive precision, rappel, F1 et effectif par classe d'une matrice 2x2.

    Parametres
    ----------
    matrix : Sequence[Sequence[int]]
        Matrice de confusion ``[[TN, FP], [FN, TP]]`` : lignes = classe
        reelle, colonnes = classe predite (ordre ``[0, 1]``).

    Retour
    ------
    dict[int, dict[str, float | int]]
        ``{0: {...}, 1: {...}}`` avec les cles ``precision``, ``recall``,
        ``f1`` et ``support``.

    Notes
    -----
    Le JSON produit par ``--run`` ne stocke que les mesures de la classe 0
    (``precision_fermeture``, etc.). Ce calcul retrouve exactement les mesures
    de la classe 1 a partir de la matrice, sans reentrainer ni reevaluer le
    modele : aucune valeur n'est inventee.
    """
    values = np.asarray(matrix, dtype=float)
    if values.shape != (2, 2):
        raise ValueError(
            "Matrice de confusion attendue en 2x2, recu "
            f"{values.shape}."
        )

    metrics: dict[int, dict[str, float | int]] = {}
    for label in (0, 1):
        true_positive = values[label, label]
        predicted = values[:, label].sum()
        actual = values[label, :].sum()
        precision = float(true_positive / predicted) if predicted else 0.0
        recall = float(true_positive / actual) if actual else 0.0
        denominator = precision + recall
        f1 = float(2 * precision * recall / denominator) if denominator else 0.0
        metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(actual),
        }
    return metrics


def format_markdown(results: Mapping[str, Any]) -> str:
    """Met en forme les resultats en tableaux Markdown copiables.

    Le rapport contient les performances macro, les metriques par classe
    (precision, rappel, F1, effectif) et les matrices de confusion du jeu de
    test. Les metriques par classe sont derivees de la matrice de confusion
    stockee dans le JSON : aucune valeur n'est inventee.
    """
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
        "| Modele | Meilleurs parametres | F1 macro (CV) | Precision macro | "
        "Rappel macro | F1 macro (test) | ROC-AUC (test) | Accuracy (test) | "
        "Accuracy equilibree | Duree (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, payload in results["models"].items():
        test = payload["test"]
        params = ", ".join(
            f"{key.replace('model__', '')}={value}"
            for key, value in payload["best_params"].items()
        )
        roc_auc = "n/a" if test["roc_auc"] is None else f"{test['roc_auc']:.4f}"
        lines.append(
            f"| {_model_label(name)} | {params} | {payload['best_cv_score']:.4f} | "
            f"{test['precision_macro']:.4f} | {test['recall_macro']:.4f} | "
            f"{test['f1_macro']:.4f} | {roc_auc} | {test['accuracy']:.4f} | "
            f"{test['balanced_accuracy']:.4f} | {payload['fit_seconds']:.1f} |"
        )

    lines += [
        "",
        "## Metriques par classe (jeu de test)",
        "",
        "| Modele | Classe | Precision | Rappel | F1 | Effectif |",
        "|---|---|---|---|---|---|",
    ]
    for name, payload in results["models"].items():
        per_class = _class_metrics_from_confusion(
            payload["test"]["confusion_matrix"]
        )
        for label in (0, 1):
            row = per_class[label]
            lines.append(
                f"| {_model_label(name)} | {label} ({CLASS_LABELS[label]}) | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | "
                f"{row['f1']:.4f} | {row['support']} |"
            )

    lines += [
        "",
        "## Matrices de confusion (jeu de test)",
        "",
        "Lignes = classe reelle, colonnes = classe predite (ordre 0, 1).",
        "",
        "| Modele | Reel \\ Predit | 0 (fermeture) | 1 (IPO ou acquisition) |",
        "|---|---|---|---|",
    ]
    for name, payload in results["models"].items():
        matrix = payload["test"]["confusion_matrix"]
        for label in (0, 1):
            lines.append(
                f"| {_model_label(name)} | {label} ({CLASS_LABELS[label]}) | "
                f"{matrix[label][0]} | {matrix[label][1]} |"
            )

    lines += [
        "",
        "## Rappel methodologique",
        "",
        "- Les hyperparametres sont choisis uniquement sur le train, en CV 5 folds.",
        "- Le jeu de test n'est utilise qu'une fois, apres la CV.",
        "- Le preprocesseur est ajuste dans chaque fold : aucune fuite de donnees.",
        "- Les metriques par classe sont derivees de la matrice de confusion du "
        "test (memes predictions, aucune reevaluation).",
        "- Les scores ne sont pas des probabilites de succes futur : ce sont des "
        "mesures de discrimination sur ce jeu de donnees historique.",
        "",
    ]
    return "\n".join(lines)


def save_report(
    results: Mapping[str, Any],
    output: str | Path,
    formatter: Any = None,
) -> Path:
    """Ecrit un JSON et son rapport Markdown a cote du chemin demande."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if formatter is not None:
        markdown = output.with_suffix(".md")
        markdown.write_text(formatter(results), encoding="utf-8")
    return output


def save_results(results: Mapping[str, Any], output: str | Path) -> Path:
    """Ecrit le JSON et le rapport Markdown des resultats principaux."""
    return save_report(results, output, format_markdown)


# ---------------------------------------------------------------------------
# I. Point d'entree
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
    parser.add_argument(
        "--importance",
        action="store_true",
        help="analyse des variables importantes du Random Forest",
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="variation des scores selon la graine des folds",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="CSV brut")
    parser.add_argument("--limit", type=int, default=None, help="sous-echantillon")
    parser.add_argument(
        "--only", choices=["both", "random_forest", "svm"], default="both"
    )
    parser.add_argument("--output", default=None, help="chemin du JSON de resultats")
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="graines CV pour --robustness (ex. 0,1,2)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="nombre de modalites affichees par --importance",
    )
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.importance:
        results = run_importance(
            path=args.data,
            limit=args.limit,
            top_n=args.top,
        )
        print()
        print(format_importance_markdown(results))
        saved = save_report(
            results,
            args.output or DEFAULT_IMPORTANCE_PATH,
            format_importance_markdown,
        )
        print(f"\nResultats ecrits dans : {saved}")
        return 0

    if args.robustness:
        seeds = tuple(int(seed) for seed in args.seeds.split(",") if seed.strip())
        results = run_robustness(
            path=args.data,
            limit=args.limit,
            seeds=seeds,
        )
        print()
        print(format_robustness_markdown(results))
        saved = save_report(
            results,
            args.output or DEFAULT_ROBUSTNESS_PATH,
            format_robustness_markdown,
        )
        print(f"\nResultats ecrits dans : {saved}")
        return 0

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
