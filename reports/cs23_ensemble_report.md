# Resultats - modeles ensemble (XGBoost et AdaBoost)

## Protocole

- Donnees : `data/raw/startup_data.csv` (sans ISR), chemin explicite
- Variables : country_code, state_code, region, city, primary_category, founding_year
- Exclue : founding_month (EDA uniquement) ; aucune variable de financement
- Validation croisee : StratifiedKFold(5, shuffle=True, random_state=42)
- Critere de selection unique : `f1_macro`
- Critere secondaire (information seulement) : `balanced_accuracy`
- Decoupage : 80 / 20 stratifie, graine 42
- Pipeline partage : `build_pipeline(model, scale=False)` pour les deux modeles
- Hyperparametres choisis sur le train uniquement ; test utilise une seule fois

## Donnees

- Lignes brutes : 65403
- Lignes retenues : 13129
- Train / test : 10503 / 2626
- Part de classe 1 (train) : 53.1 %

## Resultats

| Modele | Meilleurs parametres | F1 macro (CV) | Bal. acc. (CV) | F1 macro (test) | Bal. acc. (test) | Accuracy (test) | ROC-AUC (test) | Duree (s) |
|---|---|---|---|---|---|---|---|---|
| xgboost | colsample_bytree=1.0, learning_rate=0.1, max_depth=3, n_estimators=400, subsample=0.8 | 0.7029 +/- 0.0124 | 0.7026 +/- 0.0122 | 0.6989 | 0.6988 | 0.7049 | 0.7774 | 71.4 |
| adaboost | estimator__max_depth=3, learning_rate=0.5, n_estimators=200 | 0.6937 +/- 0.0114 | 0.6949 +/- 0.0112 | 0.6798 | 0.6823 | 0.6915 | 0.7704 | 670.3 |

## Comparaison

- Meilleur modele ensemble en CV (f1_macro) : **xgboost**
- Meilleur modele ensemble sur le test (F1 macro) : **xgboost**

### Mise en perspective (modeles classiques de CS23)

| Modele | F1 macro (CV) | F1 macro (test) | ROC-AUC (test) |
|---|---|---|---|
| zero_r | - | 0.3468 | 0.5000 |
| knn | 0.6847 | 0.6806 | 0.7517 |
| decision_tree | 0.6856 | 0.6872 | 0.7548 |
| naive_bayes | 0.6496 | 0.6372 | 0.6756 |

## Versions

- python : 3.13.9
- numpy : 2.1.3
- pandas : 2.2.3
- scikit-learn : 1.6.1
- joblib : 1.4.2
- xgboost : 3.4.1

## Rappel methodologique

- Les hyperparametres sont choisis uniquement sur le train, en CV 5 folds.
- Le jeu de test n'est utilise qu'une seule fois, apres la CV.
- Le preprocesseur est ajuste dans chaque fold : aucune fuite de donnees.
- Ces scores mesurent la discrimination sur un jeu de donnees historique ; ils ne constituent pas une promesse de succes futur.
