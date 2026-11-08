# Resultats - modeles avances (Random Forest et SVM)

## Protocole

- Variables : country_code, state_code, region, city, primary_category, founding_year
- Exclues : founding_month (EDA uniquement)
- `min_frequency` : 20
- Validation croisee : 5 folds stratifies, score `f1_macro`

## Donnees

- Lignes brutes : 65403
- Lignes retenues : 13129
- Train / test : 10503 / 2626
- Part de classe 1 : 53.1%

## Resultats

| Modele | Meilleurs parametres | F1 macro (CV) | F1 macro (test) | ROC-AUC (test) | Accuracy (test) | Duree (s) |
|---|---|---|---|---|---|---|
| random_forest | max_depth=None, min_samples_leaf=2, n_estimators=200 | 0.6984 | 0.6984 | 0.7764 | 0.7011 | 129.8 |
| svm | C=1.0, gamma=scale, kernel=rbf | 0.6997 | 0.7012 | 0.7632 | 0.7049 | 535.8 |

## Rappel methodologique

- Les hyperparametres sont choisis uniquement sur le train, en CV 5 folds.
- Le jeu de test n'est utilise qu'une fois, apres la CV.
- Le preprocesseur est ajuste dans chaque fold : aucune fuite de donnees.
- Les scores ne sont pas des probabilites de succes futur : ce sont des mesures de discrimination sur ce jeu de donnees historique.
