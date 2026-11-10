# Comparaison des six modeles — protocole commun

> **Statut : comparaison informative, non validee.** Le modele final n'est pas designe ici : le choix et la validation appartiennent a SA.

> Genere par `src/compare_models.py` le 2026-11-10T18:47:36+00:00.

## 1. Protocole

| Element | Valeur |
|---|---|
| CSV | `data/raw/startup_data.csv` |
| Variables | country_code, state_code, region, city, primary_category, founding_year |
| Exclues | founding_month |
| Critere CV | `f1_macro` |
| Folds | 5 |
| Graine | 42 |

| Effectif | Valeur |
|---|---|
| Lignes brutes | 65403 |
| Lignes retenues | 13129 |
| Train / test | 10503 / 2626 |
| Part classe 1 (train) | 0,5307 |

## 2. Performances

| Modele | Famille | F1 macro CV | F1 macro test | ROC-AUC | Accuracy | Accuracy equilibree | Precision macro | Rappel macro |
|---|---|---|---|---|---|---|---|---|
| Zero-R | classiques | — | 0,3468 | 0,5000 | 0,5308 | 0,5000 | 0,2654 | 0,5000 |
| KNN | classiques | 0,6847 | 0,6806 | 0,7517 | 0,6908 | 0,6824 | 0,6990 | 0,6824 |
| Decision Tree | classiques | 0,6856 | 0,6872 | 0,7548 | 0,6908 | 0,6868 | 0,6905 | 0,6868 |
| Gaussian Naive Bayes | classiques | 0,6496 | 0,6372 | 0,6756 | 0,6379 | 0,6376 | 0,6372 | 0,6376 |
| Random Forest | avances | 0,6984 | 0,6984 | 0,7764 | 0,7011 | 0,6979 | 0,7004 | 0,6979 |
| SVM | avances | 0,6997 | 0,7012 | 0,7632 | 0,7049 | 0,7007 | 0,7052 | 0,7007 |

### Metriques de la classe 0 (fermeture)

| Modele | Precision | Rappel | F1 |
|---|---|---|---|
| Zero-R | 0,0000 | 0,0000 | 0,0000 |
| KNN | 0,7268 | 0,5463 | 0,6237 |
| Decision Tree | 0,6888 | 0,6218 | 0,6536 |
| Gaussian Naive Bayes | 0,6097 | 0,6339 | 0,6216 |
| Random Forest | 0,6949 | 0,6469 | 0,6700 |
| SVM | 0,7075 | 0,6323 | 0,6678 |

### Matrices de confusion (lignes = reel, colonnes = predit)

| Modele | 0 / 0 | 0 / 1 | 1 / 0 | 1 / 1 |
|---|---|---|---|---|
| Zero-R | 0 | 1232 | 0 | 1394 |
| KNN | 673 | 559 | 253 | 1141 |
| Decision Tree | 766 | 466 | 346 | 1048 |
| Gaussian Naive Bayes | 781 | 451 | 500 | 894 |
| Random Forest | 797 | 435 | 350 | 1044 |
| SVM | 779 | 453 | 322 | 1072 |

## 3. Versions

| Bibliotheque | Classiques | Avances |
|---|---|---|
| python | 3.13.9 | — |
| numpy | 2.1.3 | — |
| pandas | 2.2.3 | — |
| scikit-learn | 1.6.1 | — |
| joblib | 1.4.2 | — |

> ⚠️ Les deux sources n'ont pas ete produites avec les memes versions de bibliotheques (ou les versions avancees ne sont pas enregistrees). Pour une comparaison stricte, rejouer les deux modules dans le meme environnement avant la decision finale.

## 4. Lecture

Ce tableau ne tranche pas : il rassemble les six modeles sur le meme CSV, le meme split et le meme critere CV `f1_macro`. La decision du modele final revient a SA, apres verification des versions et du protocole.

Regeneration :

```bash
python src/compare_models.py
```
