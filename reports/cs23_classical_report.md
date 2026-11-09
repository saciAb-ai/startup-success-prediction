# Resultats - modeles classiques (Zero-R, KNN, Decision Tree, Naive Bayes)

## Protocole

- Donnees : `data/raw/startup_data.csv` (sans ISR), chemin explicite via `data_loader.load_raw`
- Variables : country_code, state_code, region, city, primary_category, founding_year
- Exclues : founding_month (EDA uniquement) ; aucune variable de financement
- Validation croisee : `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- Critere de selection unique : `f1_macro` (aligne sur `train_advanced.py`)
- Critere secondaire (information seulement) : `balanced_accuracy`
- Decoupage : 80 / 20 stratifie, `random_state=42`
- Pipeline partage : `build_pipeline(model, scale=...)` ; `scale=True` pour KNN, `scale=False` pour Zero-R, Decision Tree et Naive Bayes

## Donnees

- Lignes brutes : 65 403
- Lignes retenues : 13 129
- Train / test : 10 503 / 2 626
- Part de classe 1 (train) : 53.1 %

## Resultats

| Modele | Meilleurs parametres | F1 macro (CV) | Bal. acc. (CV) | F1 macro (test) | Bal. acc. (test) | Accuracy (test) | ROC-AUC (test) | Duree (s) |
|---|---|---|---|---|---|---|---|---|
| Zero-R | - (aucun reglage) | - | - | 0.3468 | 0.5000 | 0.5308 | 0.5000 | - |
| KNN | n_neighbors=201, weights=uniform | 0.6847 +/- 0.0106 | 0.6857 +/- 0.0134 | 0.6806 | 0.6824 | 0.6908 | 0.7517 | 22.9 |
| Decision Tree | criterion=gini, max_depth=20, min_samples_split=2, min_samples_leaf=10, class_weight=balanced | 0.6856 +/- 0.0145 | 0.6854 +/- 0.0144 | 0.6872 | 0.6868 | 0.6908 | 0.7548 | 309.3 |
| Naive Bayes | var_smoothing=1e-05 | 0.6496 +/- 0.0104 | 0.6501 +/- 0.0101 | 0.6372 | 0.6376 | 0.6379 | 0.6756 | 3.8 |

## Matrices de confusion (test)

| Modele | TN | FP | FN | TP |
|---|---|---|---|---|
| Zero-R | 0 | 1232 | 0 | 1394 |
| KNN | 673 | 559 | 253 | 1141 |
| Decision Tree | 766 | 466 | 346 | 1048 |
| Naive Bayes | 781 | 451 | 500 | 894 |

## Versions

- Python : 3.13.9
- numpy : 2.1.3
- pandas : 2.2.3
- scikit-learn : 1.6.1
- joblib : 1.4.2

## Rappel methodologique

- Les hyperparametres sont choisis uniquement sur le train, en CV 5 folds.
- Le jeu de test n'est utilise qu'une seule fois, apres la CV, pour l'evaluation finale.
- Le preprocesseur est ajuste dans chaque fold : aucune fuite de donnees.
- Zero-R n'a pas d'hyperparametre : il predit toujours la classe majoritaire du train et sert de plancher de reference.
- Les metriques publiees dans les PR #10 a #12 (13 334 lignes) sont remplacees par celles-ci, calculees sur `data/raw/startup_data.csv` sans ISR.
- Ces scores mesurent la discrimination sur un jeu de donnees historique ; ils ne constituent pas une promesse de succes futur.
