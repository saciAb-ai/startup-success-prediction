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
| Dashboard EDA | 1. statistiques descriptives ; 2. valeurs manquantes ; 3. histogrammes + KDE ; 4. fréquences catégorielles ; 5. corrélations (Spearman/Pearson, V de Cramér) ; 6. filtres interactifs (issue, pays, secteur, années) |
| Analyse de profil | formulaire des 7 variables initiales, classe prédite, score, repères historiques, avertissements |
| À propos | question étudiée, cible, variables, limites |

## 3. Organisation du code

| Fichier | Rôle |
|---|---|
| `src/app.py` | point d'entrée, navigation, cache |
| `src/app_data.py` | préparation et calculs du dashboard (sans Streamlit, testé) |
| `src/dashboard_eda.py` | affichage du dashboard EDA |
| `src/models.py` | contrat de chargement du pipeline final |
| `src/dashboard_prediction.py` | page d'analyse de profil |

Modules de l'équipe réutilisés sans modification : `data_loader.py`, `features.py`,
`targets.py`, `splitting.py`.

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

Données synthétiques uniquement : ni le CSV Kaggle ni un vrai modèle ne sont nécessaires.
`tests/test_app_smoke.py` exécute réellement l'application (Streamlit AppTest).

## 6. Dépannage

| Symptôme | Solution |
|---|---|
| « Dataset introuvable » | placer le CSV directement dans `data/raw/` |
| « Aucun modèle final » | normal tant que le pipeline final n'est pas déposé dans `models/` |
| `ModuleNotFoundError` au chargement du modèle | vérifier la présence de `src/preprocessing.py` |
| avertissement de version scikit-learn | installer la version indiquée dans les métadonnées |
| port 8501 occupé | `streamlit run src/app.py --server.port 8502` |

## 7. Formulation pour la démonstration

« Ce système classe le profil d'une startup par rapport aux issues observées dans un
échantillon historique de Crunchbase. Il constitue une aide à l'analyse, pas une
prédiction certaine de son avenir. »
