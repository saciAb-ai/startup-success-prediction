# startup-success-prediction

Prédiction du succès ou de l'échec des startups à partir de leurs caractéristiques initiales (mini-projet Data Mining, classification supervisée).

## Données

- CSV de référence (équipe, sans ISR) : data/raw/startup_data.csv
- Après préparation (	argets + eatures) : **13 129** lignes retenues (closed 6 161, acquired 5 463, ipo 1 505), split stratifié 80/20 : **10 503 / 2 626** (seed 42).
- Ne pas ajouter d'autres CSV volumineux dans data/raw/ (voir .gitignore).

## Lancer l'application (Streamlit)

`ash
pip install -r requirements.txt
# optionnel : pip install -r requirements-dev.txt
="src"   # PowerShell
streamlit run src/app.py
`

Variables utiles :
- STARTUP_DATA_PATH : chemin explicite du CSV (si besoin)
- STARTUP_MODEL_PATH : chemin du pipeline final (models/final_pipeline.joblib attendu par l'app ; voir docs/app_integration.md)

## Modèle final

Le dossier models/ et models/final_pipeline.joblib ne sont pas versionnés tant que le modèle final unique n'est pas validé/exporté. Voir docs/app_integration.md pour le contrat.

## Structure

- data/ : données (référence brute dans data/raw/)
- src/ : code (préparation, modèles, app)
- docs/ : méthodologie et intégration
- eports/ : résultats (ex. RF/SVM)
