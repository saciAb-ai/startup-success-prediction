# Fiche d'oral — modèles avancés (Random Forest et SVM)

> Rédigée par `bl` (GitHub `bl7j`) pour la soutenance du mini-projet.
> Toutes les valeurs proviennent des exécutions réelles du module
> `src/train_advanced.py` sur `data/raw/startup_data.csv` (version actuelle de
> `main`, filtrée sans les startups `operating`).
> Détails : `docs/advanced_models.md`, `docs/advanced_models_report_section.md`,
> `reports/advanced_models_results.json`.

---

## 1. En une phrase

> « J'ai entraîné un Random Forest et un SVM sur six variables disponibles à la
> création d'une startup ; les deux atteignent environ 0,70 de F1 macro sur un
> jeu de test de 2 626 startups, avec un léger avantage du Random Forest en
> ROC-AUC et un avantage du SVM en accuracy. »

---

## 2. Chiffres à connaître par cœur

| Élément | Random Forest | SVM |
|---|---|---|
| Meilleurs paramètres | `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2` | `kernel=rbf`, `C=1.0`, `gamma=scale` |
| F1 macro (validation croisée 5 folds) | 0,6984 | 0,6997 |
| F1 macro (test) | **0,6984** | **0,7012** |
| Précision macro (test) | 0,7004 | 0,7052 |
| Rappel macro (test) | 0,6979 | 0,7007 |
| ROC-AUC (test) | **0,7764** | 0,7632 |
| Accuracy (test) | 0,7011 | 0,7049 |
| Accuracy équilibrée (test) | 0,6979 | 0,7007 |
| Durée d'entraînement | 129,8 s | 535,8 s |

Autres chiffres :

| Élément | Valeur |
|---|---|
| Lignes brutes | 65 403 |
| Lignes retenues (IPO + acquises + fermées) | 13 129 |
| IPO / acquises / fermées | 1 505 / 5 463 / 6 161 |
| Split train / test | 10 503 / 2 626 (80 / 20, stratifié) |
| Part de classe 1 | 53,07 % |
| Colonnes après one-hot | 300 |
| Valeurs manquantes après prétraitement | 0 |

Métriques par classe (test) :

| Modèle | Classe | Précision | Rappel | F1 |
|---|---|---|---|---|
| Random Forest | fermeture (0) | 0,6949 | 0,6469 | 0,6700 |
| Random Forest | IPO ou acquisition (1) | 0,7059 | 0,7489 | 0,7268 |
| SVM | fermeture (0) | 0,7075 | 0,6323 | 0,6678 |
| SVM | IPO ou acquisition (1) | 0,7030 | 0,7690 | 0,7345 |

Matrices de confusion (lignes = réel, colonnes = prédit) :

| Modèle | Réel \ Prédit | 0 | 1 |
|---|---|---|---|
| Random Forest | 0 | 797 | 435 |
| Random Forest | 1 | 350 | 1 044 |
| SVM | 0 | 779 | 453 |
| SVM | 1 | 322 | 1 072 |

---

## 3. Questions probables et réponses

### Q1. Pourquoi avoir choisi un Random Forest et un SVM ?

Parce que les deux couvrent des hypothèses différentes et sont adaptés à des
variables majoritairement catégorielles :

- le **Random Forest** combine des arbres de décision ; il capte des relations
  non linéaires et des interactions sans transformation manuelle ;
- le **SVM** cherche une frontière qui maximise la marge ; avec un noyau RBF et
  des variables standardisées, il peut produire une frontière non linéaire.

Ils servent aussi de comparaison : un modèle d'ensemble face à un modèle à
marge maximale.

### Q2. Pourquoi seulement six variables ?

C'est le protocole commun de l'équipe (`splitting.INITIAL_FEATURES`) :
`country_code`, `state_code`, `region`, `city`, `primary_category`,
`founding_year`. `founding_month` est calculé pour l'analyse exploratoire mais
**exclu des modèles** : une grande partie des dates Crunchbase est fixée au
1er janvier, donc le mois est un artefact de précision de date.

### Q3. À quoi sert le `Pipeline` scikit-learn ?

Il enchaîne le prétraitement et le modèle dans un seul objet :

```text
imputation -> encodage one-hot (min_frequency=20) -> [standardisation] -> modèle
```

Deux conséquences :

1. l'application charge un seul fichier `.joblib` et n'a rien à recoder ;
2. en validation croisée, le prétraitement est **réappris sur chaque fold
   d'entraînement** : aucune information du fold de validation ne fuit.

La standardisation est activée pour le SVM (`scale=True`) et désactivée pour le
Random Forest (`scale=False`), qui est insensible à l'échelle.

### Q4. Pourquoi le F1 macro comme score d'optimisation ?

Le F1 macro est la moyenne **non pondérée** des F1 des deux classes. Il ne
récompense pas un modèle qui prédit toujours la classe majoritaire, contrairement
à l'accuracy. Les classes sont ici à 46,9 % / 53,1 % : l'écart n'est pas extrême,
mais le F1 macro reste plus prudent et plus lisible par classe.

### Q5. Pourquoi 5 folds ?

C'est le protocole de l'équipe. Cinq folds donnent un compromis classique entre
coût de calcul et stabilité de l'estimation, avec `StratifiedKFold` pour
conserver la proportion des classes dans chaque fold.

### Q6. Comment garantissez-vous l'absence de fuite de données ?

Trois règles :

1. le jeu de test n'est utilisé **qu'une seule fois**, après le choix des
   hyperparamètres ;
2. le prétraitement est **dans** le `Pipeline`, donc ajusté fold par fold ;
3. l'analyse d'importance et la robustesse n'utilisent que le train.

### Q7. Le SVM n'a pas de `predict_proba`. Comment calculez-vous le ROC-AUC ?

`probability=True` est déprécié dans scikit-learn 1.9. Le SVM est donc entraîné
sans probabilités et le ROC-AUC est calculé à partir de `decision_function`,
qui fournit un score de classement orienté vers la classe 1. L'application
`src/models.py` gère déjà les deux cas (`probability` et `decision`).

### Q8. Qu'est-ce que le ROC-AUC, concrètement ?

C'est la probabilité qu'une startup positive tirée au hasard reçoive un score
plus élevé qu'une startup négative tirée au hasard. 0,5 = hasard, 1 = parfait.
Ici : 0,776 pour le Random Forest et 0,763 pour le SVM.

### Q9. Les scores de 0,70, est-ce bon ?

Pour ce jeu de données, oui dans le sens où c'est nettement au-dessus d'une
baseline naïve. Un Zero-R qui prédirait toujours la classe majoritaire aurait
environ 53,1 % d'accuracy et 0,35 de F1 macro (calcul théorique). Mais 0,70
reste un score **moyen** : les variables décrivent mal une issue future.

### Q10. Pourquoi les deux modèles ont-ils des scores si proches ?

Ils exploitent le même signal, porté surtout par `founding_year` et
`primary_category`. L'écart de F1 macro est inférieur à 0,003, ce qui est très
faible. La différence pratique est ailleurs : le Random Forest est **4 fois plus
rapide** (129,8 s contre 535,8 s) et fournit des probabilités ; le SVM a une
accuracy légèrement supérieure (0,7049 contre 0,7011) mais manque davantage de
fermetures.

### Q11. Quelles variables comptent le plus ?

Importance cumulée du Random Forest (300 colonnes one-hot) :

| Variable | Importance |
|---|---|
| `founding_year` | 0,2696 |
| `primary_category` | 0,2258 |
| `country_code` | 0,1334 |
| `city` | 0,1262 |
| `region` | 0,1244 |
| `state_code` | 0,1206 |

L'année de création et le secteur concentrent l'essentiel du signal. Les
modalités « Inconnu » ressortent aussi : l'absence d'information est associée à
l'issue dans cet échantillon.

### Q12. Le modèle est-il stable ?

Oui sur le découpage des folds. En rejouant la validation croisée avec les
graines 0, 1 et 2 (hyperparamètres fixés) :

| Modèle | Moyenne | Écart-type | Min | Max |
|---|---|---|---|---|
| Random Forest | 0,6980 | 0,0021 | 0,6953 | 0,7003 |
| SVM | 0,6977 | 0,0010 | 0,6967 | 0,6991 |

L'écart entre graines est inférieur à 0,01. Ce n'est **pas** un intervalle de
confiance statistique : c'est une mesure de stabilité du découpage.

### Q13. Peut-on en déduire une probabilité de succès pour une nouvelle startup ?

Non. Les scores sont des mesures de discrimination sur un jeu **historique**,
conditionnées à l'exclusion des startups encore en activité. Ils ne constituent
pas une probabilité de succès futur. L'application présente d'ailleurs un
« profil davantage associé à » et non une prédiction certaine.

### Q14. Pourquoi les rappels sont-ils plus élevés sur la classe 1 ?

Les deux modèles retrouvent mieux les issues positives (rappel 0,75 à 0,77) que
les fermetures (rappel 0,63 à 0,65). Le SVM, malgré une accuracy légèrement
meilleure, manque 453 fermetures contre 435 pour le Random Forest. C'est
pourquoi le F1 macro et les métriques par classe sont plus informatifs qu'un
seul taux d'accuracy.

### Q15. Pourquoi ne pas avoir équilibré les classes par sur-échantillonnage ?

Le déséquilibre est modéré (53 / 47). J'utilise `class_weight="balanced"`, qui
pondère les erreurs sans modifier les données ni dupliquer des lignes. Cela
évite d'introduire des exemples synthétiques avant la validation croisée.

### Q16. Quelles sont les limites principales ?

1. une seule graine (`random_state=42`) pour le choix des hyperparamètres ;
2. noyaux SVM polynomiaux non explorés (coût de calcul) ;
3. SVM sans probabilités calibrées ;
4. pas de découpage temporel : le modèle ne teste pas une extrapolation au futur ;
5. `feature_importances_` mesure la baisse d'impureté et favorise les variables
   à forte cardinalité ; ce n'est pas une preuve de causalité ;
6. le jeu de test ne doit plus servir à re-sélectionner des hyperparamètres.

### Q17. Si vous aviez plus de données, que feriez-vous ?

Je n'ai pas mesuré de courbe d'apprentissage, donc je ne peux pas affirmer que
plus de données améliorerait le score. La réponse méthodologique serait :
mesurer d'abord la courbe d'apprentissage, puis décider si le plafond vient du
volume de données ou des variables disponibles.

### Q18. Pourquoi ne pas avoir utilisé le mois de création ?

Parce que c'est un artefact de précision de date : une grande partie des dates
Crunchbase est fixée au 1er janvier. L'inclure reviendrait à apprendre un
comportement de saisie, pas un signal économique.

---

## 4. Phrases de secours

- « Le score est moyen parce que les six variables disponibles décrivent mal une
  issue future ; ce n'est pas seulement une limite des modèles. »
- « Le Random Forest est plus rapide et fournit des probabilités ; le SVM a une
  accuracy légèrement supérieure mais rate plus de fermetures. »
- « Les métriques par classe montrent que les deux modèles sont plus sensibles
  aux issues positives qu'aux fermetures. »
- « Toutes les valeurs du rapport sont régénérables avec
  `python src/train_advanced.py --run` ; le jeu de test n'est utilisé qu'une
  fois. »

---

## 5. Ce qu'il ne faut pas dire

- ❌ « Le modèle prédit la probabilité de succès d'une startup. »
  → C'est un score de discrimination sur un jeu historique.
- ❌ « L'année de création cause l'issue. »
  → L'importance décrit le modèle, pas une relation causale.
- ❌ « Le SVM est meilleur parce que son F1 est 0,7012 contre 0,6984. »
  → L'écart est inférieur à 0,003 ; les deux modèles sont équivalents en F1
  macro, avec des compromis différents.
- ❌ « Le modèle est validé pour prédire des startups récentes. »
  → Aucun test temporel n'a été réalisé.
