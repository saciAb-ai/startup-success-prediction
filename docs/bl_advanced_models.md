# Modeles avances - Random Forest et SVM

**Auteur :** `bl` (GitHub `bl7j`)
**Fichier :** `src/train_advanced.py`
**Tests :** `tests/test_train_advanced.py` (11 tests)
**Statut :** Draft PR #2 - <https://github.com/saciAb-ai/startup-success-prediction/pull/2>

---

## 1. Perimetre

Ce module couvre les deux modeles avances du mini-projet :

- **Random Forest** : ensemble d'arbres de decision, robuste aux relations non
  lineaires et aux variables categorieles.
- **SVM** : machine a vecteurs de support avec noyaux lineaire et RBF, adaptee
  a une frontiere de decision complexe apres standardisation.

Il ne remplace pas les modeles classiques (Zero-R, KNN, Decision Tree, Naive
Bayes), qui restent du perimetre de `sido`.

---

## 2. Protocole respecte

| Element | Valeur |
|---|---|
| Variables d'entree | `country_code`, `state_code`, `region`, `city`, `primary_category`, `founding_year` |
| Variable exclue | `founding_month` (EDA uniquement) |
| Pretraitement | `preprocessing.build_pipeline` (imputation + one-hot + scaling) |
| `min_frequency` | `20` (`preprocessing.DEFAULT_MIN_FREQUENCY`) |
| `scale` | `False` pour Random Forest, `True` pour SVM |
| Validation croisee | `StratifiedKFold(5, shuffle=True, random_state=42)` |
| Score d'optimisation | `f1_macro` |
| Graine | `42` |

Chaine officielle :

```text
data_loader.load_raw
  -> features.build_features(max_year=2026)
  -> targets.prepare_target
  -> splitting.split_dataset          # 6 variables, split stratifie 80/20
  -> preprocessing.build_pipeline     # ajuste dans chaque fold
  -> RandomForestClassifier / SVC
```

Le preprocesseur est **a l'interieur** du `Pipeline` : il est reajuste a chaque
fold de validation croisee, donc aucune information du jeu de test ne fuit dans
l'entrainement.

---

## 3. Grilles d'hyperparametres

### Random Forest (8 combinaisons, 40 ajustements)

| Parametre | Valeurs |
|---|---|
| `n_estimators` | 200, 400 |
| `max_depth` | `None`, 20 |
| `min_samples_leaf` | 1, 2 |

`class_weight="balanced"` et `n_jobs=1` (le parallelisme est porte par
`GridSearchCV` pour eviter la sur-souscription CPU).

### SVM (5 combinaisons, 25 ajustements)

| Noyau | Parametres |
|---|---|
| `linear` | `C` = 0.1, 1, 10 |
| `rbf` | `C` = 1, 10 ; `gamma="scale"` |

Les noyaux polynomiaux sont omis : sur 10 500 lignes et 300 colonnes one-hot,
leur cout d'ajustement est tres superieur aux gains attendus.

---

## 4. Resultats

Dataset `data/raw/startup_data.csv` (version actuelle de `main`, filtree sans
ISR) :

| Element | Valeur |
|---|---|
| Lignes brutes | 65 403 |
| Lignes retenues | 13 129 |
| IPO / acquises / fermees | 1 505 / 5 463 / 6 161 |
| Train / test | 10 503 / 2 626 |
| Part de classe 1 | 53,07 % |
| Colonnes apres pretraitement | 300 |
| Valeurs manquantes apres pretraitement | 0 |

| Modele | Meilleurs parametres | F1 macro (CV) | F1 macro (test) | ROC-AUC (test) | Accuracy (test) | Duree |
|---|---|---|---|---|---|---|
| Random Forest | `n_estimators=200`, `max_depth=None`, `min_samples_leaf=2` | 0,6984 | 0,6984 | 0,7764 | 0,7011 | 129,8 s |
| SVM | `kernel=rbf`, `C=1.0`, `gamma=scale` | 0,6997 | 0,7012 | 0,7632 | 0,7049 | 535,8 s |

Lecture :

- les deux modeles sont **equivalents en F1 macro** (ecart < 0,003) ;
- le Random Forest est **legerement meilleur en ROC-AUC** (0,776 contre 0,763) ;
- un Zero-R (classe majoritaire) plafonnerait autour de 53 % d'accuracy et
  0,35 de F1 macro : les deux modeles apportent donc un gain net ;
- 0,70 reste un score **moyen** : les variables disponibles (geographie,
  secteur, annee) decrivent mal une issue future. C'est une limite du jeu de
  donnees, pas seulement des modeles.

---

## 5. Reproduire

```bash
# Configuration seule, sans entrainement
python src/train_advanced.py

# Test rapide sur 1 500 lignes
python src/train_advanced.py --run --limit 1500

# Experience complete (environ 11 minutes)
python src/train_advanced.py --run --output reports/bl_advanced_results.json

# Tests unitaires
python -m pytest tests/test_train_advanced.py -q
```

---

## 6. Fichiers livres

| Fichier | Role |
|---|---|
| `src/train_advanced.py` | Pipelines, grilles, CV, chargement, evaluation |
| `tests/test_train_advanced.py` | 11 tests (protocole, pipelines, grilles, bout en bout) |
| `reports/bl_advanced_results.json` | Resultats detailles et reproductibles |
| `reports/bl_advanced_results.md` | Tableau de resultats |

---

## 7. Limites

1. Une seule graine (`random_state=42`) : pas d'intervalle de confiance.
2. Noyaux SVM polynomiaux non explores.
3. SVM sans probabilites calibrees ; le ROC-AUC utilise `decision_function`.
4. Les scores ne sont pas des probabilites de succes futur.
5. Le jeu de test ne doit pas servir a re-selectionner les hyperparametres.

---

## 8. Note sur les chiffres du document de passation

`docs/handoff_preprocessing.md` indique 66 368 lignes brutes, 13 334 retenues
et 306 colonnes. Ces valeurs datent d'**avant** le filtrage ISR applique dans
`main` (commits `91a6a04` et `bd8a0cb`). Les valeurs de reference actuelles
sont celles du tableau ci-dessus. Toute comparaison entre modeles doit se faire
sur la meme version du dataset.

---

## 9. Apres le merge

- [ ] Verifier que `main` contient bien `src/train_advanced.py`.
- [ ] Verifier l'apparition des commits dans l'historique de `main`.
- [ ] Recuperer `main` en local et relancer les tests.
- [ ] Ne plus pousser sur la branche fusionnee : creer une nouvelle branche
      pour toute modification ulterieure.
