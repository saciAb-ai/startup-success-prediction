# Application Streamlit : lancement et intégration du modèle

L'application affiche et charge : aucun modèle n'y est entraîné.

## 1. Lancer l'application (local)

Depuis la racine du dépôt (sous Windows, activer avec `.venv\Scripts\Activate.ps1`) :

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    streamlit run src/app.py

Le CSV Kaggle doit se trouver directement dans `data/raw/` (dossier ignoré par Git).
Autre emplacement : variable d'environnement `STARTUP_DATA_PATH`.

## 2. Pages

| Page | Contenu |
|---|---|
| Dashboard EDA | 1. statistiques descriptives ; 2. valeurs manquantes ; 3. histogrammes + KDE ; 4. fréquences catégorielles ; 5. corrélations (Spearman/Pearson, V de Cramér) ; 6. filtres interactifs (issue, pays, secteur, années) ; 7. tests statistiques (chi-deux secteur × issue avec résidus de Pearson, V de Cramér et sensibilité ; Mann-Whitney sur l'année de création et le financement ; correction de Holm ; limites ; valeurs de référence), recalculés sur la sélection filtrée ; en complément : carte du taux de succès par pays et tendance par année de création |
| Analyse de profil | formulaire des variables d'entrée du modèle (`features.FEATURE_COLUMNS`), classe prédite, score, repères historiques, avertissements |
| À propos | question étudiée, cible, variables, limites |

## 3. Organisation du code

| Fichier | Rôle |
|---|---|
| `src/app.py` | point d'entrée, navigation, cache |
| `src/app_data.py` | préparation et calculs du dashboard (sans Streamlit, testé) |
| `src/dashboard_eda.py` | affichage du dashboard EDA |
| `src/models.py` | contrat de chargement du pipeline final |
| `src/dashboard_prediction.py` | page d'analyse de profil |
| `src/ui.py` | thème des graphiques, blocs d'interface, format français des nombres |
| `src/app_insights.py` | taux de succès par pays et par année de création (testé) |
| `src/app_inference.py` | tests statistiques de l'onglet 7 et valeurs de référence (sans Streamlit, testé) |

Modules de l'équipe réutilisés sans modification : `data_loader.py`, `features.py`,
`targets.py`, `splitting.py`.

**Année maximale de création.** `app_data.build_eda_frame` transmet
`max_year=2026` (`app_data.EDA_MAX_YEAR`) à `features.build_features`, comme les
scripts d'entraînement et le notebook EDA : une date de création postérieure à
2026 est traitée comme manquante, quelle que soit l'année en cours.

**Onglet « Tests statistiques ».** Méthodes et valeurs de référence : passation
de ZM (`docs/handoff_eda.md`, `notebooks/01_eda.ipynb` §17). Les tests portent sur
`app_data.modeling_rows(data)` (issue connue, après filtres), sans imputation.
Sur l'ensemble labellisé complet, ils reproduisent exactement les valeurs de
référence (vérifié par `tests/test_app_inference.py`). Si un filtre rend un test
impossible (une seule classe, moins de deux secteurs après regroupement, groupe
de moins de deux valeurs), l'onglet affiche un message au lieu d'un résultat.

## 4. Contrat du modèle final (à valider par SA et l'équipe modèles)

- Fichier : `models/final_pipeline.joblib` (autre emplacement : `STARTUP_MODEL_PATH`).
- Contenu : le `Pipeline` scikit-learn **entraîné** (prétraitement + modèle), appris sur `X_train` uniquement.
- Entrées : DataFrame avec les colonnes de `features.FEATURE_COLUMNS` ; valeur manquante = `NaN`.
- Classes : `0` = fermeture, `1` = IPO ou acquisition.
- Métadonnées facultatives : `models/final_pipeline.json`.

Exemple d'enregistrement, côté équipe modèles, une fois le modèle retenu :

    import json
    from pathlib import Path

    import joblib
    import sklearn

    from evaluation import evaluate_classifier

    final_pipeline = search.best_estimator_      # GridSearchCV déjà ajusté sur X_train, y_train
    Path("models").mkdir(exist_ok=True)
    joblib.dump(final_pipeline, "models/final_pipeline.joblib")

    metrics = evaluate_classifier(final_pipeline, X_test, y_test)
    metadata = {
        "model_name": "Random Forest",           # nom du modèle retenu
        "trained_at": "2026-11-05",
        "trained_by": "IASDxx",                  # code étudiant, jamais de nom réel
        "sklearn_version": sklearn.__version__,
        "features": list(X_train.columns),
        "test_metrics": {k: v for k, v in metrics.items() if k != "confusion_matrix"},
    }
    Path("models/final_pipeline.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

À chaque chargement, l'application vérifie : méthode `predict`, classes `[0, 1]`,
colonnes d'entrée, prédiction sur un profil test et version de scikit-learn.

Points d'attention :

- même version de scikit-learn à l'entraînement et dans l'application ;
- le pipeline utilise `src/preprocessing.py` : ce fichier doit être présent ;
- committer ou non le fichier `.joblib` : décision d'équipe (taille, reproductibilité) ;
- sécurité : ne charger que le fichier produit par l'équipe (un fichier joblib peut exécuter du code).

## 5. Tests

Depuis la racine du dépôt :

    pip install -r requirements-dev.txt
    python -m pytest -q

Données synthétiques pour l'essentiel : aucun vrai modèle n'est nécessaire.
`tests/test_app_smoke.py` exécute réellement l'application (Streamlit AppTest).
`tests/test_app_inference.py` vérifie aussi les valeurs de référence des tests
statistiques sur `data/raw/startup_data.csv` ; ces tests sont ignorés si le CSV
est absent ou différent.

## 6. Dépannage

| Symptôme | Solution |
|---|---|
| « Dataset introuvable » | placer le CSV directement dans `data/raw/` |
| « Aucun modèle final » | normal tant que le pipeline final n'est pas déposé dans `models/` |
| `ModuleNotFoundError` au chargement du modèle | vérifier la présence de `src/preprocessing.py` |
| avertissement de version scikit-learn | installer la version indiquée dans les métadonnées |
| « Le module scipy est introuvable » (onglet Tests statistiques) | `pip install -r requirements.txt` |
| port 8501 occupé | `streamlit run src/app.py --server.port 8502` |

## 7. Formulation pour la démonstration

« Ce système classe le profil d'une startup par rapport aux issues observées dans un
échantillon historique de Crunchbase. Il constitue une aide à l'analyse, pas une
prédiction certaine de son avenir. »
