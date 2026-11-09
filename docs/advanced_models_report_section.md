# Section de rapport — Modèles avancés (Random Forest et SVM)

> Rédigée par `bl` (GitHub `bl7j`) pour le rapport final du mini-projet.
> Toutes les valeurs proviennent des exécutions réelles sur
> `data/raw/startup_data.csv` (version actuelle de `main`, filtrée sans ISR).
> Les résultats détaillés sont dans `reports/` et la note technique dans
> `docs/advanced_models.md`.

---

## 1. Objectif de cette partie

Le projet compare six classifieurs supervisés : Zero-R, KNN, Decision Tree,
Random Forest, Naïve Bayes et SVM. Cette section couvre les **deux modèles
avancés** :

- **Random Forest** : ensemble d'arbres de décision entraînés sur des
  sous-échantillons de lignes et de variables, puis agrégés par vote. Il
  capture des relations non linéaires et des interactions entre variables
  catégorielles sans transformation manuelle.
- **SVM (machine à vecteurs de support)** : recherche d'une frontière de
  décision qui maximise la marge entre les classes. Avec un noyau RBF, il peut
  produire une frontière non linéaire après standardisation des variables.

Les modèles classiques (Zero-R, KNN, Decision Tree, Naïve Bayes) relèvent d'un
autre membre de l'équipe ; ils ne sont pas traités ici.

---

## 2. Données et protocole

### 2.1 Données

| Élément | Valeur mesurée |
|---|---|
| Lignes brutes | 65 403 |
| Lignes retenues (IPO + acquises + fermées) | 13 129 |
| IPO / acquises / fermées | 1 505 / 5 463 / 6 161 |
| Split train / test | 10 503 / 2 626 (80 / 20, stratifié) |
| Part de classe 1 | 53,07 % |
| Colonnes après prétraitement | 300 |
| Valeurs manquantes après prétraitement | 0 |

### 2.2 Variables d'entrée

Six variables, conformes au protocole commun de l'équipe
(`splitting.INITIAL_FEATURES`) :

`country_code`, `state_code`, `region`, `city`, `primary_category`,
`founding_year`.

`founding_month` est calculé par `features.py` pour l'analyse exploratoire mais
**exclu des modèles** : il s'agit d'un artefact de précision de date (une large
part des dates Crunchbase est fixée au 1er janvier).

### 2.3 Chaîne de traitement

```text
data_loader.load_raw
  -> features.build_features(max_year=2026)
  -> targets.prepare_target
  -> splitting.split_dataset          (6 variables, split stratifié 80/20)
  -> preprocessing.build_pipeline     (imputation + one-hot + scaling)
  -> RandomForestClassifier / SVC
```

Le prétraitement est placé **à l'intérieur** du `Pipeline` scikit-learn :

- imputation des variables catégorielles par la modalité « Inconnu » ;
- imputation des variables numériques par la médiane ;
- encodage one-hot avec `min_frequency=20` pour regrouper les modalités rares ;
- standardisation uniquement pour le SVM (`scale=True`), pas pour le Random
  Forest (`scale=False`).

Ainsi, les paramètres d'imputation, d'encodage et de standardisation sont
réappris **à chaque fold** de validation croisée : aucune information du jeu de
test ne fuit dans l'entraînement.

### 2.4 Validation croisée

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ;
- score d'optimisation : **F1 macro** (moyenne non pondérée des F1 des deux
  classes, adaptée au déséquilibre) ;
- `GridSearchCV` avec `error_score="raise"` ;
- le jeu de test n'est utilisé **qu'une seule fois**, après la sélection des
  hyperparamètres.

---

## 3. Hyperparamètres explorés

### 3.1 Random Forest — 8 combinaisons (40 ajustements)

| Paramètre | Valeurs |
|---|---|
| `n_estimators` | 200, 400 |
| `max_depth` | `None`, 20 |
| `min_samples_leaf` | 1, 2 |

`class_weight="balanced"` compense le déséquilibre entre succès et fermetures.
La forêt reste mono-thread (`n_jobs=1`) : le parallélisme est porté par
`GridSearchCV`.

### 3.2 SVM — 5 combinaisons (25 ajustements)

| Noyau | Paramètres |
|---|---|
| `linear` | `C` = 0,1 / 1 / 10 |
| `rbf` | `C` = 1 / 10 ; `gamma="scale"` |

Les noyaux polynomiaux sont volontairement omis : sur 10 500 lignes et 300
colonnes one-hot, leur coût d'ajustement est très supérieur aux gains attendus.
Le SVM est utilisé **sans `probability=True`** (déprécié dans scikit-learn
1.9) : le ROC-AUC est calculé à partir de `decision_function`.

---

## 4. Résultats

### 4.1 Performances sur le jeu de test

| Modèle | Meilleurs paramètres | F1 macro (CV) | Précision macro | Rappel macro | F1 macro (test) | ROC-AUC (test) | Accuracy (test) | Accuracy équilibrée | Durée |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2` | 0,6984 | 0,7004 | 0,6979 | **0,6984** | **0,7764** | 0,7011 | 0,6979 | 129,8 s |
| SVM | `kernel=rbf`, `C=1.0`, `gamma=scale` | 0,6997 | 0,7052 | 0,7007 | **0,7012** | **0,7632** | 0,7049 | 0,7007 | 535,8 s |

### 4.2 Métriques par classe

Le jeu de test contient 1 232 fermetures (classe 0) et 1 394 issues
positives (classe 1). Les valeurs ci-dessous sont calculées à partir de la
matrice de confusion du test, donc sur exactement les mêmes prédictions que le
tableau précédent.

| Modèle | Classe | Précision | Rappel | F1 | Effectif |
|---|---|---|---|---|---|
| Random Forest | 0 (fermeture) | 0,6949 | 0,6469 | 0,6700 | 1 232 |
| Random Forest | 1 (IPO ou acquisition) | 0,7059 | 0,7489 | 0,7268 | 1 394 |
| SVM | 0 (fermeture) | 0,7075 | 0,6323 | 0,6678 | 1 232 |
| SVM | 1 (IPO ou acquisition) | 0,7030 | 0,7690 | 0,7345 | 1 394 |

Les deux modèles retrouvent mieux la classe 1 (rappel 0,75 à 0,77) que la
classe 0 (rappel 0,63 à 0,65) : ils détectent mieux une issue positive qu'ils
ne signalent une fermeture.

### 4.3 Matrices de confusion

Lignes = classe réelle, colonnes = classe prédite (ordre 0, 1).

| Modèle | Réel \ Prédit | 0 (fermeture) | 1 (IPO ou acquisition) |
|---|---|---|---|
| Random Forest | 0 (fermeture) | 797 | 435 |
| Random Forest | 1 (IPO ou acquisition) | 350 | 1 044 |
| SVM | 0 (fermeture) | 779 | 453 |
| SVM | 1 (IPO ou acquisition) | 322 | 1 072 |

Lecture : le Random Forest classe correctement 797 fermetures et 1 044 issues
positives ; le SVM en classe 779 et 1 072. Le SVM rate donc davantage de
fermetures (453 contre 435), ce qui explique son rappel plus faible sur la
classe 0, tandis qu'il retrouve davantage d'issues positives (1 072 contre
1 044).

### 4.4 Importance des variables (Random Forest)

Entraînement sur le train seul, 300 colonnes one-hot, agrégation par variable
d'origine :

| Variable | Importance cumulée |
|---|---|
| `founding_year` | **0,2696** |
| `primary_category` | **0,2258** |
| `country_code` | 0,1334 |
| `city` | 0,1262 |
| `region` | 0,1244 |
| `state_code` | 0,1206 |

Les modalités les plus discriminantes sont `num__founding_year` (0,2696), puis
les modalités « Inconnu » de `city` (0,0516), `country_code` (0,0477) et
`primary_category` (0,0473).

### 4.5 Robustesse au découpage des folds

Même protocole, seules les graines de `StratifiedKFold` changent (0, 1, 2),
toujours sur le train :

| Modèle | Moyenne | Écart-type | Min | Max | Durée |
|---|---|---|---|---|---|
| Random Forest | **0,6980** | 0,0021 | 0,6953 | 0,7003 | 41,3 s |
| SVM | **0,6977** | 0,0010 | 0,6967 | 0,6991 | 349,0 s |

---

## 5. Interprétation

1. **Les deux modèles sont équivalents en F1 macro** (écart inférieur à 0,003).
   Le Random Forest est légèrement meilleur en ROC-AUC (0,776 contre 0,763).
2. **Le gain par rapport à Zero-R est net.** Un Zero-R qui prédit toujours la
   classe majoritaire plafonnerait autour de 53 % d'accuracy et 0,35 de F1
   macro. Les deux modèles dépassent 0,69 de F1 macro.
3. **Le score reste moyen (≈ 0,70).** Les variables disponibles (géographie,
   secteur, année) décrivent mal une issue future. C'est une limite du jeu de
   données autant que des modèles.
4. **L'année de création et le secteur portent l'essentiel du signal.** La
   géographie se partage le reste. L'importance élevée des modalités
   « Inconnu » indique que l'absence d'information est elle-même associée à
   l'issue dans cet échantillon.
5. **Les résultats sont stables.** Changer la graine des folds déplace le F1
   macro de moins de 0,01 : le classement entre modèles ne dépend pas du
   découpage.
6. **Les erreurs sont asymétriques.** Les deux modèles retrouvent mieux les
   issues positives (rappel 0,75 à 0,77) que les fermetures (rappel 0,63 à
   0,65). Le SVM, légèrement meilleur en accuracy, manque 453 fermetures
   contre 435 pour le Random Forest ; il compense en retrouvant 28 issues
   positives de plus (1 072 contre 1 044).

---

## 6. Limites

1. **Une seule graine pour le choix des hyperparamètres** (`random_state=42`).
   La robustesse est mesurée à part sur 3 graines de découpage, mais cela ne
   constitue pas un intervalle de confiance statistique.
2. **Noyaux SVM polynomiaux non explorés** (coût de calcul).
3. **SVM sans probabilités calibrées** : le ROC-AUC utilise
   `decision_function` ; l'application devra gérer ce cas.
4. **Les scores ne sont pas des probabilités de succès futur** : ce sont des
   mesures de discrimination sur un jeu historique, conditionnées à
   l'exclusion des startups `operating`.
5. **`feature_importances_` mesure la baisse d'impureté** et favorise les
   variables à forte cardinalité (`city`, `region`). Ces importances décrivent
   le modèle, elles ne prouvent pas une relation causale.
6. **Le jeu de test ne doit plus servir** à re-sélectionner des
   hyperparamètres.

---

## 7. Reproductibilité

```bash
# Configuration seule (aucun entraînement)
python src/train_advanced.py

# Expérience complète (environ 11 minutes)
python src/train_advanced.py --run --output reports/advanced_models_results.json

# Importance des variables (environ 11 secondes)
python src/train_advanced.py --importance

# Robustesse multi-graines (environ 6,5 minutes)
python src/train_advanced.py --robustness --seeds 0,1,2

# Tests du module (24 tests)
python -m pytest tests/test_train_advanced.py -q
```

Fichiers produits :

| Fichier | Contenu |
|---|---|
| `src/train_advanced.py` | Pipelines, grilles, CV, évaluation, importance, robustesse |
| `tests/test_train_advanced.py` | 24 tests |
| `reports/advanced_models_results.json` / `.md` | Résultats principaux |
| `reports/feature_importance.json` / `.md` | Importance des variables |
| `reports/robustness.json` / `.md` | Robustesse multi-graines |
| `docs/advanced_models.md` | Note technique complète |

---

## 8. Contribution personnelle

| Commit | Contenu |
|---|---|
| `11e6645` | Pipelines Random Forest et SVM, grilles d'hyperparamètres, `GridSearchCV` 5 folds |
| `66cc95f` | Fusion de `main` dans la branche (0 conflit) |
| `f6c3cd1` | Alignement sur le protocole commun (6 variables, `build_pipeline`, `min_frequency=20`) et premiers résultats réels |
| `06492ce` | 11 tests unitaires et note de synthèse |
| `ce3e72e` | Importance des variables, robustesse multi-graines, 21 tests au total |
| `ee301ca` | Section de rapport prête à coller (`docs/advanced_models_report_section.md`) |

La **PR #2** a été fusionnée dans `main` en **merge commit** (`41793f0`), sans
squash : les commits restent attribués à `bl7j`. Aucun fichier des autres
membres n'a été modifié. Une branche ultérieure,
`feature/bl-detailed-metrics`, ajoute les métriques par classe et les matrices
de confusion (24 tests au total).

---

## 9. Vérifications après fusion de la PR #2

Fusion effectuée le 8 novembre 2026 (`main` : `bd8a0cb` → `41793f0`) :

- [x] `main` contient `src/train_advanced.py`, `tests/test_train_advanced.py`,
      `docs/advanced_models.md`, `docs/advanced_models_report_section.md` et `reports/`.
- [x] Les commits `11e6645`, `66cc95f`, `f6c3cd1`, `06492ce`, `ce3e72e`,
      `ee301ca` apparaissent dans l'historique de `main`.
- [x] Le profil GitHub `bl7j` apparaît dans les contributeurs du dépôt.
- [x] Les tests du module passent sur `main` après fusion (21/21).
- [x] Aucune PR des autres membres n'a été touchée.
- [ ] Ne plus pousser sur la branche fusionnée : toute modification ultérieure
      passe par une nouvelle branche.

---

## 10. Phrase de conclusion pour la soutenance

> « Nous avons comparé un Random Forest et un SVM sur six variables
> disponibles à la création d'une startup. Les deux atteignent environ 0,70 de
> F1 macro sur le test, avec un léger avantage du Random Forest en ROC-AUC
> (0,78 contre 0,76). L'année de création et le secteur concentrent l'essentiel
> du signal, et les scores varient de moins de 0,01 quand on change le
> découpage des folds. Ces résultats restent des mesures de discrimination sur
> un jeu historique, pas des probabilités de succès futur. »
