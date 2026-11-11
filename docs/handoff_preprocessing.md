# Passation - donnees et pretraitement

**De :** `tas` (donnees / nettoyage et pretraitement)
**Pour :** `sido` (modeles classiques), `bl` (modeles avances), `wlid` (application Streamlit)
**Etat :** `src/data_loader.py`, `src/features.py`, `src/preprocessing.py` sont dans `main`.

Ce document repond a une seule question : comment brancher votre code sur les
donnees sans recalculer ni dupliquer ce qui existe deja.

---

## 1. La chaine a respecter

```
CSV brut
  -> data_loader.load_raw()        lecture seule, aucun filtrage
  -> features.build_features()     6 variables d'entree (+ 1 colonne EDA)
  -> targets.prepare_target()      filtre ipo/acquired/closed, cree target (SA)
  -> splitting.split_dataset()     train/test stratifie 80/20 (SA)
  -> preprocessing.build_pipeline() imputation + encodage + standardisation
  -> votre modele
```

Ne sautez aucune etape et **ne refiltrez pas les statuts vous-memes** : `operating`
doit rester exclu, sinon la cible change de sens.

**Dataset de reference :** `data/raw/startup_data.csv`, **65 403 lignes, sans
startup israelienne**. L'ancien export Kaggle (66 368 lignes, ISR incluses) n'est
plus valable : il fausse tous les effectifs du rapport. `load_raw` affiche un
avertissement `[ATTENTION]` si le fichier charge n'est pas celui de reference
(mauvais nom, mauvais nombre de lignes, ou lignes `ISR` detectees).

```python
import sys
sys.path.insert(0, "src")

import data_loader, features, targets, splitting

brut = data_loader.load_raw("data/raw/startup_data.csv")   # chemin explicite
carac = features.build_features(brut, max_year=2026)
donnees = targets.prepare_target(carac)
X_train, X_test, y_train, y_test = splitting.split_dataset(donnees)
```

Un chemin different peut etre impose par la variable d'environnement
`STARTUP_DATA_PATH` (meme nom que dans l'application) ; `find_raw_file()` la
respecte avant de chercher le fichier canonique.

Resultat attendu : **13 129 lignes** conservees, **10 503 / 2 626** apres
decoupage, classe 1 = **53,1 %** des deux cotes.

---

## 2. Le contrat des variables

`splitting.INITIAL_FEATURES` (6 variables, c'est la reference) :

| Variable | Type | Traitement |
|---|---|---|
| `country_code` | categorielle | imputation "Inconnu" + one-hot |
| `state_code` | categorielle | idem |
| `region` | categorielle | idem |
| `city` | categorielle | idem |
| `primary_category` | categorielle | idem |
| `founding_year` | numerique | mediane (+ standardisation si `scale=True`) |

`founding_month` **n'est pas une entree** : 42,3 % des dates sont au `01-01`
(annee seule), le mois est un artefact de precision de date. La colonne reste
calculee pour l'EDA (`features.EDA_ONLY_COLUMNS`) et ne doit pas etre ajoutee a
la main dans vos listes de variables.

`preprocessing.CATEGORICAL_FEATURES` et `preprocessing.NUMERIC_FEATURES` sont
**derivees automatiquement** de `splitting.INITIAL_FEATURES` : si le protocole
change, le pretraitement suit sans modification. Ne codez pas ces listes en dur.

---

## 3. Utilisation selon votre role

### sido et bl : modeles

```python
from preprocessing import build_pipeline, feature_names_out

pipeline = build_pipeline(RandomForestClassifier(), scale=False)
pipeline.fit(X_train, y_train)          # imputation + encodage appris ici, train seul
pipeline.predict(X_test)

# Importance des variables / coefficients, avec les vrais noms de colonnes
noms = feature_names_out(pipeline.named_steps["preprocessor"])
```

`build_pipeline` accepte n'importe quel estimateur scikit-learn : `DummyClassifier`,
`KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`,
`GaussianNB`, `SVC`. Il est compatible `GridSearchCV` : le pretraitement est
re-appris a chaque fold, donc pas de fuite pendant la recherche d'hyperparametres.

### wlid : application

Le contrat de l'application est decrit dans `docs/app_integration.md` :
**`models/final_pipeline.joblib`** (ou `STARTUP_MODEL_PATH`), avec
`models/final_pipeline.json` en metadonnees facultatives. C'est le seul nom de
fichier a utiliser : le modele final est unique, choisi apres la comparaison des
six modeles.

```python
import joblib
from preprocessing import build_pipeline

# 1) entrainement (une fois, hors de l'app)
pipeline = build_pipeline(RandomForestClassifier(), scale=False)
pipeline.fit(X_train, y_train)
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/final_pipeline.joblib")   # ne pas versionner ce fichier

# 2) prediction d'une startup saisie dans le formulaire
pipeline = joblib.load("models/final_pipeline.joblib")
ligne = features.build_features(pd.DataFrame([saisie]), max_year=2026)
prediction = pipeline.predict(ligne)          # DataFrame avec les 6 colonnes
```

Deux points importants pour l'app :

- passer un **DataFrame** avec les colonnes du protocole, pas une liste ;
- une ville ou un secteur jamais vus ne provoquent pas d'erreur
  (`handle_unknown="ignore"`) : la valeur est simplement ignoree par le modele.

---

## 4. Quel reglage pour quel modele

| Modele | `scale` | Pourquoi |
|---|---|---|
| Zero-R (`DummyClassifier`) | `False` | aucune distance calculee |
| Arbre de decision, Random Forest | `False` | insensibles a l'echelle |
| Gaussian Naive Bayes | `False` | suppose des variables dans leur unite d'origine |
| KNN | `True` | repose sur des distances |
| SVM | `True` | sensible a l'echelle des variables |

`scale=True` standardise `founding_year` (moyenne 0, ecart-type 1 sur le train).

---

## 5. Chiffres attendus (pour verifier votre integration)

Reference : `data/raw/startup_data.csv`, 65 403 lignes, sans ISR.

| Indicateur | Valeur |
|---|---|
| Lignes brutes | 65 403 x 14 |
| Lignes conservees | 13 129 (IPO 1 505 + acquired 5 463 + closed 6 161) |
| Train / test | 10 503 / 2 626 (seed 42, stratifie) |
| Variables d'entree | 6 |
| Colonnes apres pretraitement | **300** (`min_frequency=20`) |
| Colonnes sans regroupement | 2 764 |
| Valeurs manquantes apres pretraitement | 0 |
| Villes encodees (train) | 1 480 |
| Villes presentes uniquement dans le test | 191 (transformees sans erreur) |

Si vous obtenez autre chose, c'est que la chaine a ete court-circuitee quelque part.

---

## 6. Verification rapide

```powershell
.\.venv\Scripts\python.exe src\features.py        # contrat 6 variables
.\.venv\Scripts\python.exe src\preprocessing.py   # imputation + encodage + echelle
.\.venv\Scripts\python.exe src\targets.py         # cible IPO/acquired/closed
```

Le notebook `notebooks/02_preprocessing_encoding.ipynb` deroule la chaine complete
sur le vrai dataset, avec les sorties attendues.

---

## 7. Trois erreurs a ne pas commettre

1. **`fit` sur le jeu complet.** `pre.fit(X)` puis `transform(X_test)` fait fuiter
   la mediane et les modalites du test. Toujours `fit` sur `X_train` uniquement,
   ou utiliser `build_pipeline` qui l'impose.
2. **Recalculer les variables.** `features.build_features` est appele une fois,
   en amont. Le refaire par modele produit des colonnes differentes d'un fichier
   a l'autre.
3. **Ajouter `founding_month`, `status` ou `target` en entree.** Le mois est un
   artefact de date ; `status` et `target` sont la cible, les utiliser revient a
   donner la reponse au modele (`splitting.FORBIDDEN_FEATURES`).

Questions sur cette partie : passer par un commentaire sur la PR ou ouvrir une
issue avec le code `tas`.
