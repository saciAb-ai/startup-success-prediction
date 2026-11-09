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

| Modele | Meilleurs parametres | F1 macro (CV) | Precision macro | Rappel macro | F1 macro (test) | ROC-AUC (test) | Accuracy (test) | Accuracy equilibree | Duree (s) |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | max_depth=None, min_samples_leaf=2, n_estimators=200 | 0.6984 | 0.7004 | 0.6979 | 0.6984 | 0.7764 | 0.7011 | 0.6979 | 129.8 |
| SVM | C=1.0, gamma=scale, kernel=rbf | 0.6997 | 0.7052 | 0.7007 | 0.7012 | 0.7632 | 0.7049 | 0.7007 | 535.8 |

## Metriques par classe (jeu de test)

| Modele | Classe | Precision | Rappel | F1 | Effectif |
|---|---|---|---|---|---|
| Random Forest | 0 (fermeture) | 0.6949 | 0.6469 | 0.6700 | 1232 |
| Random Forest | 1 (IPO ou acquisition) | 0.7059 | 0.7489 | 0.7268 | 1394 |
| SVM | 0 (fermeture) | 0.7075 | 0.6323 | 0.6678 | 1232 |
| SVM | 1 (IPO ou acquisition) | 0.7030 | 0.7690 | 0.7345 | 1394 |

## Matrices de confusion (jeu de test)

Lignes = classe reelle, colonnes = classe predite (ordre 0, 1).

| Modele | Reel \ Predit | 0 (fermeture) | 1 (IPO ou acquisition) |
|---|---|---|---|
| Random Forest | 0 (fermeture) | 797 | 435 |
| Random Forest | 1 (IPO ou acquisition) | 350 | 1044 |
| SVM | 0 (fermeture) | 779 | 453 |
| SVM | 1 (IPO ou acquisition) | 322 | 1072 |

## Rappel methodologique

- Les hyperparametres sont choisis uniquement sur le train, en CV 5 folds.
- Le jeu de test n'est utilise qu'une fois, apres la CV.
- Le preprocesseur est ajuste dans chaque fold : aucune fuite de donnees.
- Les metriques par classe sont derivees de la matrice de confusion du test (memes predictions, aucune reevaluation).
- Les scores ne sont pas des probabilites de succes futur : ce sont des mesures de discrimination sur ce jeu de donnees historique.
