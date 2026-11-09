# Prédiction du succès des startups

Mini-projet de **Data Mining** (classification supervisée) : prédire si une startup aura un **succès** (`ipo` ou `acquired`) ou un **échec** (`closed`) à partir de caractéristiques disponibles au moment de sa création.

## Objectif

Construire, comparer et évaluer plusieurs modèles de classification selon un protocole commun, puis intégrer le modèle retenu dans une application Streamlit.

## Données

- **Jeu de données de référence (équipe)** : `data/raw/startup_data.csv`
- **Variables d’entrée (protocole)** : `country_code`, `state_code`, `region`, `city`, `primary_category`, `founding_year`
- **Cible** : `success` (1 = `ipo`/`acquired`, 0 = `closed` ; `operating` exclu)
- Ne pas ajouter d’autres fichiers CSV volumineux dans `data/raw/` (voir `.gitignore`).

Les choix de préparation, le découpage des données et le critère d’évaluation sont décrits dans [`docs/methodology.md`](docs/methodology.md).

## Installation

```bash
pip install -r requirements.txt
```

Optionnel (développement/tests) :

```bash
pip install -r requirements-dev.txt
```

## Lancer l’application Streamlit

Depuis la racine du projet, avec `src` accessible dans le `PYTHONPATH` :

**PowerShell**
```powershell
$env:PYTHONPATH = "src"
streamlit run src/app.py
```

**Bash**
```bash
export PYTHONPATH=src
streamlit run src/app.py
```

### Variables d’environnement utiles

| Variable | Rôle |
|---|---|
| `STARTUP_DATA_PATH` | Chemin explicite vers le CSV à utiliser (optionnel) |
| `STARTUP_MODEL_PATH` | Chemin vers le pipeline final à charger (optionnel) |

## Modèle final et intégration

Le pipeline final attendu par l’application est `models/final_pipeline.joblib` (voir le contrat dans [`docs/app_integration.md`](docs/app_integration.md)).

> Le dossier `models/` et les artefacts du pipeline final ne sont pas versionnés tant que le modèle final unique n’a pas été validé et exporté selon le protocole d’équipe.

## Structure du dépôt

```text
.
├── data/        # Données (référence dans data/raw/)
├── docs/        # Méthodologie, intégration, notes d’équipe
├── notebooks/   # Analyses et expérimentations
├── reports/     # Résultats et rapports générés
├── src/         # Code source (préparation, modèles, application)
├── tests/       # Tests automatisés
├── requirements.txt
└── requirements-dev.txt
```

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — protocole et choix méthodologiques
- [`docs/app_integration.md`](docs/app_integration.md) — contrat d’intégration du modèle dans l’application
- [`docs/handoff_preprocessing.md`](docs/handoff_preprocessing.md) — notes de passage pour le prétraitement/pipeline
