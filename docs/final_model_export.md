# Export du modèle final pour l'application

> Rédigé par `bl` (GitHub `bl7j`).
> Le contrat d'intégration est défini dans `docs/app_integration.md` et vérifié
> par `src/models.py`.

---

## 1. À quoi sert ce script

`src/export_final_model.py` produit les deux fichiers attendus par
l'application Streamlit :

| Fichier | Contenu |
|---|---|
| `models/final_pipeline.joblib` | `Pipeline` scikit-learn **entraîné** (prétraitement + modèle) |
| `models/final_pipeline.json` | Métadonnées : modèle, version de scikit-learn, variables, hyperparamètres, métriques test enregistrées, effectifs |

Il ne choisit pas le modèle à la place de l'équipe : le modèle est passé en
argument. Il ne committe rien : l'écriture se fait localement.

---

## 2. Prérequis

1. Le CSV brut doit être présent dans `data/raw/startup_data.csv`.
2. Les résultats de `--run` doivent exister :

   ```bash
   python src/train_advanced.py --run --output reports/advanced_models_results.json
   ```

   Le script y lit les meilleurs hyperparamètres **et** les métriques de test
   déjà enregistrées : il ne réutilise donc pas le jeu de test.

---

## 3. Utilisation

```bash
# Random Forest
python src/export_final_model.py --model random_forest

# SVM
python src/export_final_model.py --model svm

# Autre emplacement
python src/export_final_model.py --model random_forest --output models/final_pipeline.joblib
```

Options disponibles : `--data`, `--results`, `--output`, `--max-year`,
`--limit`, `--trained-by`.

Le script :

1. recharge les meilleurs paramètres du modèle depuis le JSON de `--run` ;
2. reconstruit le `Pipeline` commun (`preprocessing.build_pipeline`) ;
3. l'entraîne sur **`X_train` uniquement** ;
4. écrit le `.joblib` (compressé) et le `.json` ;
5. recharge le fichier avec `src/models.py` pour vérifier les classes `[0, 1]`,
   les six colonnes d'entrée et la prédiction sur un profil vide.

---

## 4. Métadonnées produites

```json
{
  "model_key": "random_forest",
  "model_name": "Random Forest",
  "trained_by": "bl7j",
  "sklearn_version": "1.9.1",
  "features": [
    "country_code", "state_code", "region",
    "city", "primary_category", "founding_year"
  ],
  "best_params": {
    "max_depth": null, "min_samples_leaf": 2, "n_estimators": 200
  },
  "test_metrics": {
    "accuracy": 0.7011,
    "balanced_accuracy": 0.6979,
    "precision_macro": 0.7004,
    "recall_macro": 0.6979,
    "f1_macro": 0.6984,
    "precision_fermeture": 0.6949,
    "recall_fermeture": 0.6469,
    "f1_fermeture": 0.6700,
    "roc_auc": 0.7764
  },
  "dataset": {
    "raw_rows": 65403,
    "kept_rows": 13129,
    "train_rows": 10503,
    "test_rows": 2626,
    "positive_share": 0.5307
  },
  "exported_at": "2026-11-09T12:00:00+00:00"
}
```

La matrice de confusion est volontairement exclue des métadonnées : elle reste
dans `reports/advanced_models_results.json` et dans le rapport.

---

## 5. Vérification manuelle

```python
import sys
sys.path.insert(0, "src")

import models

loaded = models.load_model("models/final_pipeline.joblib")
print(loaded.name, loaded.notes)

X = models.make_input_frame({"city": "Paris", "founding_year": 2015})
prediction = models.predict_profile(loaded, X)
print(prediction.predicted_class, prediction.score, prediction.score_kind)
```

Attendu :

- Random Forest : `score_kind == "probability"` ;
- SVM : `score_kind == "decision"` (pas de `probability=True`, déprécié dans
  scikit-learn 1.9).

---

## 6. Faut-il committer le `.joblib` ?

C'est une **décision d'équipe**, pas une décision technique :

- committer le fichier rend l'application reproductible sans réentraînement,
  mais ajoute un binaire au dépôt (taille, versions de scikit-learn) ;
- ne pas le committer impose de lancer le script avant toute démonstration.

Dans les deux cas, le fichier doit être produit avec la **même version de
scikit-learn** que l'application.

---

## 7. Périmètre

Le script gère les deux modèles de `bl` : `random_forest` et `svm`. Si l'équipe
retient un modèle classique (Zero-R, KNN, Decision Tree, Naïve Bayes), il faudra
soit ajouter son constructeur, soit étendre le script après la décision
d'équipe.
