"""Preparation des modeles avances (Random Forest et SVM).

Projet : Prediction du succes ou de l'echec des startups a partir de leurs
caracteristiques initiales (mini-projet Data Mining).

Role de ce module
-----------------
Construire, sans les entrainer :

- le preprocesseur commun (``ColumnTransformer``) ;
- les ``Pipeline`` scikit-learn pour Random Forest et SVM ;
- les grilles d'hyperparametres ;
- les objets de recherche (``GridSearchCV``) avec validation croisee
  stratifiee a 5 folds et score ``f1_macro``.

Tous les objets retournes sont NON ENTRAINES : aucune methode d'ajustement
n'est appelee dans ce fichier. Apres ``search.fit(X_train, y_train)``, le
modele optimise est ``search.best_estimator_`` ; il ne faut pas confondre
``search.estimator`` (Pipeline vierge) avec ``search`` (objet de recherche).

Regles respectees
-----------------
- Les caracteristiques proviennent du protocole commun de SA
  (``src/splitting.INITIAL_FEATURES``) ; aucune liste n'est inventee.
- Aucune interface n'est supposee pour les futurs fichiers de l'equipe tas
  (``src/features.py``, ``src/preprocessing.py``, ``src/data_loader.py``) :
  ce module n'en depend pas.
- Le preprocesseur reste dans le ``Pipeline`` afin d'eviter toute fuite de
  donnees : il sera ajuste uniquement sur les folds d'entrainement.
- Le parallelisme est porte par ``GridSearchCV`` ; la foret reste
  volontairement mono-thread pendant l'optimisation (voir
  ``build_random_forest_pipeline``).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:  # Execution depuis la racine du projet.
    from src.splitting import FORBIDDEN_FEATURES, INITIAL_FEATURES
except ImportError:  # Execution directe du fichier (python src/train_advanced.py).
    from splitting import FORBIDDEN_FEATURES, INITIAL_FEATURES

# ---------------------------------------------------------------------------
# Constantes du protocole commun
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
SCORING: str = "f1_macro"
CV_FOLDS: int = 5

# Seuil de regroupement des modalites rares de ``OneHotEncoder``.
# Valeur prudente : sur un jeu d'environ 10 000 lignes, une modalite vue moins
# de 10 fois represente moins de 0,1 % des observations. Ce seuil limite
# l'explosion du nombre de colonnes one-hot (city, region, ...) sans supprimer
# de variable. A REVALIDER sur le vrai dataset, selon la cardinalite reelle
# mesuree par l'equipe tas.
MIN_FREQUENCY: int = 10

# Variables numeriques du protocole initial : annee et mois de creation,
# a creer par l'equipe tas a partir de founded_at.
DEFAULT_NUMERIC_FEATURES: tuple[str, ...] = ("founding_year", "founding_month")

# Variables categorielles du protocole initial.
DEFAULT_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "country_code",
    "state_code",
    "region",
    "city",
    "primary_category",
)


# ---------------------------------------------------------------------------
# A. Preprocessing
# ---------------------------------------------------------------------------


def _as_feature_list(value: Sequence[str], label: str) -> list[str]:
    """Convertit un parametre de colonnes en liste de noms, avec controle de type.

    Une chaine seule (``"city"``) est refusee explicitement : elle serait
    sinon decoupee en caracteres et produirait une erreur trompeuse.
    """
    if isinstance(value, str):
        raise TypeError(
            f"{label} doit etre une sequence de noms de colonnes, pas une chaine "
            f"unique ({value!r}). Ecrire par exemple [{value!r}]."
        )
    names = list(value)
    invalid = [name for name in names if not isinstance(name, str)]
    if invalid:
        raise TypeError(
            f"{label} doit contenir uniquement des chaines de caracteres ; "
            f"valeurs invalides : {invalid!r}"
        )
    return names


def _duplicates(names: Sequence[str]) -> list[str]:
    """Retourne la liste triee des noms presents plus d'une fois."""
    seen: set[str] = set()
    duplicated: set[str] = set()
    for name in names:
        if name in seen:
            duplicated.add(name)
        seen.add(name)
    return sorted(duplicated)


def resolve_feature_types(
    features: Sequence[str] = INITIAL_FEATURES,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Separe les caracteristiques fournies en variables numeriques et categorielles.

    Parametres
    ----------
    features : sequence de str
        Liste complete des caracteristiques du modele (par defaut celle de SA).
        Une chaine seule est refusee (``TypeError``).
    numeric_features : sequence de str, optionnel
        Variables numeriques declarees. Si ``None``, les variables numeriques
        connues du protocole (annee et mois de creation) sont retenues
        lorsque elles font partie de ``features``.
    categorical_features : sequence de str, optionnel
        Variables categorielles declarees. Si ``None``, les variables
        categorielles connues du protocole sont retenues.

    Retour
    ------
    tuple[list[str], list[str]]
        ``(numeric, categorical)``. L'ordre suit
        ``DEFAULT_NUMERIC_FEATURES`` et ``DEFAULT_CATEGORICAL_FEATURES``
        lorsque les listes sont deduites, sinon l'ordre des listes fournies.

    Leve
    ----
    TypeError
        Si un parametre est une chaine seule ou contient des noms non-str.
    ValueError
        Si une caracteristique est dupliquee, interdite (fuite de donnees),
        declaree mais absente de ``features``, non couverte par le typage,
        a la fois numerique et categorielle, ou si l'une des deux listes est
        vide. Chaque cas dispose d'un message dedie.

    Notes
    -----
    Le typage est declare explicitement, pas deduit des donnees : le module ne
    charge aucun dataset et ne fait aucune hypothese sur les types pandas.
    """
    features = _as_feature_list(features, "features")
    if not features:
        raise ValueError("La liste des caracteristiques est vide.")
    duplicated = _duplicates(features)
    if duplicated:
        raise ValueError(f"Caracteristiques dupliquees dans features : {duplicated}")
    if not isinstance(FORBIDDEN_FEATURES, (set, frozenset)):
        raise TypeError("FORBIDDEN_FEATURES doit etre un ensemble.")

    forbidden = sorted(set(features) & set(FORBIDDEN_FEATURES))
    if forbidden:
        raise ValueError(
            f"Caracteristiques interdites (fuite temporelle ou cible) : {forbidden}"
        )

    if numeric_features is None:
        numeric = [name for name in DEFAULT_NUMERIC_FEATURES if name in features]
    else:
        numeric = _as_feature_list(numeric_features, "numeric_features")
    if categorical_features is None:
        categorical = [name for name in DEFAULT_CATEGORICAL_FEATURES if name in features]
    else:
        categorical = _as_feature_list(categorical_features, "categorical_features")

    duplicated_numeric = _duplicates(numeric)
    if duplicated_numeric:
        raise ValueError(f"Variables numeriques dupliquees : {duplicated_numeric}")
    duplicated_categorical = _duplicates(categorical)
    if duplicated_categorical:
        raise ValueError(
            f"Variables categorielles dupliquees : {duplicated_categorical}"
        )

    overlap = sorted(set(numeric) & set(categorical))
    if overlap:
        raise ValueError(
            f"Variables declarees a la fois numeriques et categorielles : {overlap}"
        )

    unknown = sorted((set(numeric) | set(categorical)) - set(features))
    if unknown:
        raise ValueError(
            "Variables declarees numeriques ou categorielles mais absentes de "
            f"features : {unknown}"
        )

    unsupported = sorted(set(features) - (set(numeric) | set(categorical)))
    if unsupported:
        raise ValueError(
            f"Variables non prises en charge par le preprocessing : {unsupported}. "
            "Declarer leur type via numeric_features ou categorical_features."
        )

    if not numeric:
        raise ValueError(
            "Aucune variable numerique : numeric_features est vide ou ne contient "
            "aucune colonne presente dans features. Colonnes numeriques du "
            f"protocole : {list(DEFAULT_NUMERIC_FEATURES)}. Le SVM et "
            "StandardScaler exigent au moins une variable numerique."
        )
    if not categorical:
        raise ValueError(
            "Aucune variable categorielle : categorical_features est vide ou ne "
            "contient aucune colonne presente dans features. Colonnes "
            f"categorielles du protocole : {list(DEFAULT_CATEGORICAL_FEATURES)}."
        )

    return numeric, categorical


def build_preprocessor(
    features: Sequence[str] = INITIAL_FEATURES,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    min_frequency: int | float | None = MIN_FREQUENCY,
    sparse_output: bool = True,
) -> ColumnTransformer:
    """Construit le preprocesseur commun aux deux modeles (non entraine).

    Parametres
    ----------
    features : sequence de str
        Caracteristiques du modele (par defaut ``INITIAL_FEATURES``).
    numeric_features, categorical_features : sequence de str, optionnel
        Typage explicite ; voir :func:`resolve_feature_types`.
    min_frequency : int | float | None
        Seuil de regroupement des modalites rares de ``OneHotEncoder``
        (``MIN_FREQUENCY`` par defaut, soit 10). ``None`` desactive tout
        regroupement. Valeur a valider sur le vrai dataset : elle depend de
        la cardinalite reelle de ``city``, ``region``, etc.
    sparse_output : bool
        Sortie creuse de ``OneHotEncoder`` (par defaut ``True``) : adaptee
        aux variables categorielles de forte cardinalite et acceptee par
        ``RandomForestClassifier`` comme par ``SVC``.

    Retour
    ------
    ColumnTransformer
        Objet NON ENTRAINE.
        Numeriques : imputation mediane puis standardisation.
        Categorielles : imputation par la modalite la plus frequente puis
        encodage one-hot tolerant les modalites inconnues.
        Colonnes non declarees : abandonnees (``remainder="drop"``).

    Notes
    -----
    Ce preprocesseur doit rester a l'interieur d'un ``Pipeline`` afin que
    l'imputation, l'encodage et la standardisation soient appris uniquement
    sur les donnees d'entrainement de chaque fold.
    """
    numeric, categorical = resolve_feature_types(
        features=features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse_output=sparse_output,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# B. Random Forest
# ---------------------------------------------------------------------------


def build_random_forest_pipeline(
    features: Sequence[str] = INITIAL_FEATURES,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    min_frequency: int | float | None = MIN_FREQUENCY,
    class_weight: str | Mapping[int, float] | None = "balanced",
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,
    **estimator_params: Any,
) -> Pipeline:
    """Assemble le ``Pipeline`` Random Forest (preprocessing + classifieur).

    Parametres
    ----------
    features, numeric_features, categorical_features :
        Voir :func:`build_preprocessor`.
    min_frequency : int | float | None
        Seuil de regroupement des modalites rares du one-hot
        (``MIN_FREQUENCY`` par defaut) ; voir :func:`build_preprocessor`.
    class_weight : str | Mapping[int, float] | None
        Ponderation des classes ; ``"balanced"`` par defaut afin de limiter
        l'effet d'un eventuel desequilibre entre succes et fermetures.
    random_state : int
        Graine de reproductibilite (42 par defaut).
    n_jobs : int
        Parallelisme interne de la foret (``1`` par defaut). Pendant
        ``GridSearchCV``, une foret mono-thread evite le double parallelisme :
        sinon chaque essai mobiliserait tous les coeurs alors que plusieurs
        essais tournent deja en parallele. Cette sur-souscription CPU (et
        memoire) ralentit l'ensemble ; le parallelisme utile est porte par
        ``GridSearchCV(n_jobs=...)``.
    **estimator_params :
        Parametres supplementaires transmis a ``RandomForestClassifier``.

    Retour
    ------
    Pipeline
        Pipeline NON ENTRAINE. Etapes nommees ``"preprocess"`` puis ``"clf"`` ;
        les hyperparametres du modele sont donc prefixes par ``clf__`` dans
        les grilles.
    """
    preprocessor = build_preprocessor(
        features=features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        min_frequency=min_frequency,
    )
    classifier = RandomForestClassifier(
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
        **estimator_params,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", classifier)])


def get_random_forest_param_grid() -> dict[str, list[Any]]:
    """Grille d'hyperparametres Random Forest (24 combinaisons).

    Retour
    ------
    dict[str, list[Any]]
        ``n_estimators`` : 200, 400
        ``max_depth`` : None, 10, 20
        ``min_samples_split`` : 2, 5
        ``min_samples_leaf`` : 1, 2

    Notes
    -----
    La taille de la grille est volontairement moderee : 24 combinaisons x 5
    folds = 120 ajustements, ce qui reste raisonnable sur un jeu de quelques
    dizaines de milliers de lignes. ``max_features`` et d'autres parametres
    pourront etre ajoutes apres un premier retour de validation croisee.

    ``clf__n_jobs`` n'apparait pas dans la grille : la foret doit rester
    mono-thread (``n_jobs=1``) pour ne pas se superposer au parallelisme de
    ``GridSearchCV``.
    """
    return {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf": [1, 2],
    }


# ---------------------------------------------------------------------------
# C. SVM
# ---------------------------------------------------------------------------


def build_svm_pipeline(
    features: Sequence[str] = INITIAL_FEATURES,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    min_frequency: int | float | None = MIN_FREQUENCY,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str | Mapping[int, float] | None = "balanced",
    random_state: int = RANDOM_STATE,
    **estimator_params: Any,
) -> Pipeline:
    """Assemble le ``Pipeline`` SVM (preprocessing + ``SVC``).

    Parametres
    ----------
    features, numeric_features, categorical_features :
        Voir :func:`build_preprocessor`. La standardisation des variables
        numeriques est indispensable au SVM.
    min_frequency : int | float | None
        Seuil de regroupement des modalites rares du one-hot
        (``MIN_FREQUENCY`` par defaut) ; voir :func:`build_preprocessor`.
    kernel : str
        Noyau du ``SVC`` (``"rbf"`` par defaut).
    C : float
        Parametre de regularisation.
    gamma : str | float
        Coefficient du noyau (``"scale"`` par defaut, noyaux rbf et poly).
    class_weight : str | Mapping[int, float] | None
        Ponderation des classes, ``"balanced"`` par defaut.
    random_state : int
        Graine transmise au ``SVC`` ; elle n'a d'effet que si une estimation
        probabiliste est activee.
    **estimator_params :
        Parametres supplementaires transmis a ``SVC``.

    Retour
    ------
    Pipeline
        Pipeline NON ENTRAINE. Etapes nommees ``"preprocess"`` puis ``"clf"``.

    Notes
    -----
    ``probability`` n'est volontairement pas transmis a ``SVC`` : ce parametre
    est deprecie depuis scikit-learn 1.9 (suppression prevue en 1.11). Son
    defaut effectif reste ``False``, donc ``predict_proba`` n'est pas
    disponible et le ROC-AUC est calcule depuis ``decision_function`` par
    ``src.evaluation.evaluate_classifier``. Si une probabilite devient
    necessaire (application Streamlit), utiliser plus tard
    ``CalibratedClassifierCV(SVC(), ensemble=False)`` plutot que
    ``SVC(probability=True)`` ; aucune calibration n'est mise en place ici.
    """
    preprocessor = build_preprocessor(
        features=features,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        min_frequency=min_frequency,
    )
    classifier = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        random_state=random_state,
        **estimator_params,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("clf", classifier)])


def get_svm_param_grid() -> list[dict[str, list[Any]]]:
    """Grille d'hyperparametres SVM (11 combinaisons, listes de dictionnaires).

    Retour
    ------
    list[dict[str, list[Any]]]
        Trois blocs, afin de ne tester ``gamma`` que la ou il a un sens :

        - ``kernel="linear"`` : ``C`` = 0.1, 1, 10  (3 combinaisons)
        - ``kernel="rbf"`` : ``C`` = 1, 10 ; ``gamma`` = "scale", 0.01  (4)
        - ``kernel="poly"`` : ``C`` = 1, 10 ; ``gamma`` = "scale" ;
          ``degree`` = 2, 3  (4)

    Notes
    -----
    ``gamma`` est ignore par le noyau lineaire ; le decouper en blocs evite
    des combinaisons inutiles dans la recherche. ``probability`` n'est pas
    recherche : la calibration probabiliste est hors perimetre a ce stade.
    """
    return [
        {
            "clf__kernel": ["linear"],
            "clf__C": [0.1, 1.0, 10.0],
        },
        {
            "clf__kernel": ["rbf"],
            "clf__C": [1.0, 10.0],
            "clf__gamma": ["scale", 0.01],
        },
        {
            "clf__kernel": ["poly"],
            "clf__C": [1.0, 10.0],
            "clf__gamma": ["scale"],
            "clf__degree": [2, 3],
        },
    ]


# ---------------------------------------------------------------------------
# D. Optimisation (construction des objets de recherche, sans ajustement)
# ---------------------------------------------------------------------------


def build_cross_validation(
    n_splits: int = CV_FOLDS, random_state: int = RANDOM_STATE
) -> StratifiedKFold:
    """Validation croisee stratifiee a ``n_splits`` folds (5 par defaut).

    Parametres
    ----------
    n_splits : int
        Nombre de folds, au moins 5 conformement au protocole.
    random_state : int
        Graine du melange des folds.

    Retour
    ------
    StratifiedKFold
        ``shuffle=True`` pour des folds melanges et reproductibles.
    """
    if n_splits < 5:
        raise ValueError("Le protocole impose au moins 5 folds de validation.")
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )


def build_random_forest_search(
    pipeline: Pipeline | None = None,
    param_grid: Mapping[str, Iterable[Any]] | None = None,
    cv: StratifiedKFold | None = None,
    scoring: str = SCORING,
    n_jobs: int = -1,
    verbose: int = 0,
    error_score: str | float = "raise",
) -> GridSearchCV:
    """Prepare la recherche d'hyperparametres Random Forest.

    Parametres
    ----------
    pipeline : Pipeline | None
        Pipeline a optimiser ; construit par defaut avec
        :func:`build_random_forest_pipeline`.
    param_grid : Mapping | None
        Grille a utiliser ; par defaut :func:`get_random_forest_param_grid`.
    cv : StratifiedKFold | None
        Validation croisee ; par defaut :func:`build_cross_validation`.
    scoring : str
        Mesure de comparaison, ``"f1_macro"`` par defaut.
    n_jobs : int
        Parallelisme des essais (``-1`` = tous les coeurs). C'est le seul
        niveau de parallelisme pendant l'optimisation : la foret reste
        mono-thread (``clf__n_jobs=1``) pour eviter la sur-souscription.
    verbose : int
        Verbosite de ``GridSearchCV`` (``0`` par defaut).
    error_score : str | float
        Valeur utilisee lorsqu'un ajustement echoue. ``"raise"`` par defaut :
        une erreur de fold est signalee immediatement au lieu de produire un
        score ``NaN`` silencieux.

    Retour
    ------
    GridSearchCV
        Objet NON ENTRAINE, pret a etre ajuste plus tard ; **aucun ajustement**
        n'est declenche par cette fonction. Le modele optimise sera
        ``search.best_estimator_`` apres ``search.fit(X_train, y_train)``.
    """
    if pipeline is None:
        pipeline = build_random_forest_pipeline()
    return GridSearchCV(
        estimator=pipeline,
        param_grid=dict(param_grid) if param_grid is not None
        else get_random_forest_param_grid(),
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
    n_jobs: int = -1,
    verbose: int = 0,
    error_score: str | float = "raise",
) -> GridSearchCV:
    """Prepare la recherche d'hyperparametres SVM.

    Parametres
    ----------
    pipeline : Pipeline | None
        Pipeline a optimiser ; construit par defaut avec
        :func:`build_svm_pipeline`.
    param_grid : Sequence[Mapping] | None
        Grille a utiliser ; par defaut :func:`get_svm_param_grid`.
    cv, scoring, n_jobs, verbose, error_score :
        Voir :func:`build_random_forest_search`.

    Retour
    ------
    GridSearchCV
        Objet NON ENTRAINE, pret a etre ajuste plus tard ; **aucun ajustement**
        ici. Le modele optimise sera ``search.best_estimator_`` apres
        ``search.fit(X_train, y_train)``.
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
    features: Sequence[str] = INITIAL_FEATURES,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
    min_frequency: int | float | None = MIN_FREQUENCY,
    n_splits: int = CV_FOLDS,
    scoring: str = SCORING,
    n_jobs: int = -1,
    verbose: int = 0,
    error_score: str | float = "raise",
) -> dict[str, GridSearchCV]:
    """Prepare, sans entrainement, les deux experiences d'optimisation.

    Parametres
    ----------
    features, numeric_features, categorical_features :
        Caracteristiques et typage explicite (voir
        :func:`resolve_feature_types`).
    min_frequency : int | float | None
        Seuil de regroupement des modalites rares du one-hot
        (``MIN_FREQUENCY`` par defaut) ; voir :func:`build_preprocessor`.
    n_splits : int
        Nombre de folds (>= 5).
    scoring : str
        Mesure de comparaison (``"f1_macro"``).
    n_jobs : int
        Parallelisme des essais, porte uniquement par ``GridSearchCV``.
    verbose : int
        Verbosite des recherches (``0`` par defaut).
    error_score : str | float
        Comportement en cas d'echec d'ajustement ; ``"raise"`` par defaut.

    Retour
    ------
    dict[str, GridSearchCV]
        ``{"random_forest": search_rf, "svm": search_svm}``. Les deux objets
        sont NON ENTRAINES. Le Pipeline vierge est ``search.estimator`` ;
        apres ``search.fit(X_train, y_train)``, le modele optimise est
        ``search.best_estimator_`` (Pipeline ajuste, pret pour
        ``src.evaluation.evaluate_classifier``).

    Notes
    -----
    Aucun objet ``Pipeline`` n'est retourne directement afin d'eviter toute
    confusion entre le Pipeline non ajuste et l'objet de recherche. La suite
    du projet consistera a ajuster ``search`` sur ``X_train``/``y_train``
    issus de ``src.splitting.split_dataset``, puis a evaluer
    ``search.best_estimator_`` sur le jeu de test.
    """
    cross_validation = build_cross_validation(n_splits=n_splits)

    forest_search = build_random_forest_search(
        pipeline=build_random_forest_pipeline(
            features=features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            min_frequency=min_frequency,
        ),
        cv=cross_validation,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score=error_score,
    )
    svm_search = build_svm_search(
        pipeline=build_svm_pipeline(
            features=features,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            min_frequency=min_frequency,
        ),
        cv=cross_validation,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        error_score=error_score,
    )

    return {"random_forest": forest_search, "svm": svm_search}


def main() -> None:
    """Affiche la configuration preparee, sans entrainer aucun modele."""
    experiment = prepare_experiment()
    for name, search in experiment.items():
        print(f"[{name}] pipeline={list(search.estimator.named_steps)}")
        print(f"[{name}] cv={search.cv} scoring={search.scoring}")
        print(f"[{name}] grille={search.param_grid}")
        print(
            f"[{name}] n_jobs={search.n_jobs} verbose={search.verbose} "
            f"error_score={search.error_score!r}"
        )
    print("Configuration prete. Aucun entrainement n'est lance par ce module.")
    print(
        "Apres search.fit(X_train, y_train), utiliser search.best_estimator_ "
        "(et non search.estimator)."
    )


if __name__ == "__main__":
    main()
