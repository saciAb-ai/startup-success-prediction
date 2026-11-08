# Passation EDA — inférence statistique pour le dashboard Streamlit

**De :** ZM (EDA / analyse statistique)  
**Pour :** Wlid (dashboard Streamlit)  
**Branche :** `feature/zm-eda`  
**Source :** `notebooks/01_eda.ipynb`, sections 10 à 19  
**Périmètre de cette passation :** Phase 1 — inférence statistique formelle

> **Notation.** Les valeurs numériques de référence sont écrites en notation
> anglaise (point décimal, exposant `e`) pour être copiées directement dans les
> tests. L'interface affichera la virgule française. Les noms de variables et de
> fonctions Python restent en anglais.

> **Principe de propriété.** ZM possède l'analyse EDA et ce document. Wlid
> possède l'implémentation du dashboard (`src/app.py`, `src/dashboard_eda.py`,
> `src/app_data.py`, `src/app_insights.py`, `src/ui.py`, les tests et les
> dépendances). ZM ne modifie aucun de ces fichiers.

---

## Table des matières

1. [Objet](#1-objet)
2. [Contrat de données](#2-contrat-de-données)
3. [Ce qui existe déjà dans le dashboard](#3-ce-qui-existe-déjà-dans-le-dashboard)
4. [Ce que cette passation ajoute](#4-ce-que-cette-passation-ajoute)
5. [Méthodes statistiques](#5-méthodes-statistiques)
6. [Résultats de référence](#6-résultats-de-référence)
7. [Spécifications d'intégration](#7-spécifications-dintégration)
8. [Limites et interprétation](#8-limites-et-interprétation)
9. [Répartition des responsabilités](#9-répartition-des-responsabilités)
10. [Références](#10-références)
11. [Annexe A — Formules](#annexe-a--formules)
12. [Annexe B — Implémentation de référence](#annexe-b--implémentation-de-référence)
13. [Annexe C — Checklist d'acceptation](#annexe-c--checklist-dacceptation)

---

## 1. Objet

Ce document transmet à Wlid les résultats d'inférence statistique produits dans
`notebooks/01_eda.ipynb` et les spécifications nécessaires pour les intégrer au
dashboard Streamlit.

Il répond à trois questions :

1. **Quel est le contrat de données** que les tests doivent respecter ?
2. **Quelles méthodes et quelles valeurs de référence** doivent être affichées ?
3. **Où et comment** les intégrer sans dupliquer ce que le dashboard fait déjà ?

Le dashboard existant couvre déjà l'analyse descriptive. Cette passation
n'ajoute que l'**inférence formelle** : test du chi-deux, test de Mann-Whitney,
tailles d'effet, correction de Holm, diagnostics de résidus et interprétation.
Les intervalles de Wilson et la missingness différentielle par issue sont
**différés en Phase 2** (voir section 4.3).

---

## 2. Contrat de données

### 2.1 Chaîne officielle

```python
import data_loader
import features
import targets

raw = data_loader.load_raw(data_loader.find_raw_file(PROJECT_ROOT))
features_frame = features.build_features(raw, max_year=2026)
labelled = targets.prepare_target(features_frame)
```

- `data_loader.load_raw` : lecture seule, aucun filtrage.
- `features.build_features` : construit les 6 variables d'entrée et
  `founding_month` (EDA seulement).
- `targets.prepare_target` : garde `ipo`, `acquired`, `closed` ; `target = 1`
  pour `ipo`/`acquired`, `target = 0` pour `closed` ; `operating` est exclu.
- Résultat : **13 129 lignes labellisées**, dont **6 968 succès** et
  **6 161 échecs**, soit un taux de succès global de **53.07 %**.

### 2.2 Correspondance avec le dashboard

- `app_data.build_eda_frame(raw)` prépare le dashboard **sans supprimer de
  lignes** : les startups `operating` restent présentes avec `target = NA` et
  `issue = "Exclu (operating)"`.
- `app_data.modeling_rows(frame)` garde uniquement les lignes dont `target` est
  non nul, c'est-à-dire le périmètre labellisé.
- **Vérification effectuée :**
  `app_data.modeling_rows(app_data.build_eda_frame(raw))` est identique au
  `labelled` du notebook — même index, même `target`, mêmes valeurs de
  `primary_category`, `country_code`, `founding_year` et de financement.
  Les **valeurs** de financement sont équivalentes, mais les **dtypes pandas**
  peuvent différer : le notebook construit une Series d'analyse `funding_usd`
  à partir de la représentation texte (`"-"` → `NA`, puis conversion numérique),
  alors que la colonne de financement du dashboard est déjà parsée en numérique.
  Comparer les valeurs après conversion explicite, pas les dtypes.
- Conséquence pour Wlid : les tests dynamiques peuvent être calculés avec
  `app_data.modeling_rows(data)`, sans reparser le CSV ni recalculer les
  variables.

### 2.3 Variables

| Variable | Rôle | Utilisation dans les tests |
|---|---|---|
| `target` | cible officielle (0/1) | variable de regroupement |
| `primary_category` | entrée du modèle | test du chi-deux |
| `founding_year` | entrée du modèle | test de Mann-Whitney |
| `funding_total_usd` | exclue du modèle (`FORBIDDEN_FEATURES`) | Mann-Whitney descriptif |
| `funding_rounds` | exclue du modèle (`FORBIDDEN_FEATURES`) | non testée en Phase 1 |
| `founding_month` | EDA seulement (`EDA_ONLY_COLUMNS`) | non testée |
| `status` | sert uniquement à construire `target` | jamais en entrée |

Points importants :

- `funding_total_usd` et `funding_rounds` sont dans
  `splitting.FORBIDDEN_FEATURES`. Les tester est **descriptif** ; cela ne les
  autorise pas à entrer dans `X`.
- `founding_month` est un artefact de précision de date. Deux statistiques
  distinctes doivent être distinguées :
  - **60.73 %** des mois de création **connus** valent janvier
    (5 715 / 9 410 sur le périmètre labellisé ; `app_data.january_share` =
    0.6073, valeur affichée par le dashboard) ;
  - **42.27 %** des lignes labellisées sont datées **exactement au 01-01**
    (5 549 / 13 129).

  Ces deux pourcentages ne mesurent pas la même chose : le premier porte sur les
  lignes dont le mois est renseigné, le second sur toutes les lignes labellisées.
  Le notebook et le dashboard utilisent la première définition. `founding_month`
  ne doit pas être utilisé dans les tests.
- `target` et `status` ne sont jamais des variables d'entrée.

### 2.4 Valeurs manquantes

- **Aucune imputation, aucun encodage, aucune standardisation.**
- Chaque test utilise les lignes où la variable testée est observée,
  séparément pour les succès et les échecs (analyse des cas complets).
- **Chi-deux :** `primary_category` manquant est **exclu** de la table, ce n'est
  pas une modalité. C'est pourquoi n = **12 052** et non 13 129 : 1 077 lignes
  n'ont pas de secteur.
- **Mann-Whitney :** les manquants de `founding_year` et de
  `funding_total_usd` sont exclus par variable et par groupe. Les effectifs
  manquants sont reportés dans les tableaux.
- **Ne pas réutiliser `app_data.frequency_by_issue` pour le chi-deux :** cette
  fonction traite les manquants comme une modalité « Inconnu (manquant) » et
  regroupe par top-n, pas par seuil de rareté.

### 2.5 `max_year`

- Le notebook fixe `max_year=2026` :
  `features.build_features(raw, max_year=2026)`.
- **`app_data.build_eda_frame(raw)` n'accepte pas de paramètre `max_year`.** Il
  appelle `features.build_features(raw)` sans le transmettre : `founding_year`
  est donc recalculé avec l'année courante. En 2026, les deux coïncident, mais
  ce n'est pas reproductible si l'année change.
- Pour garantir la reproductibilité, deux voies (ZM ne modifie pas
  `app_data.py`) :
  1. **étendre explicitement `build_eda_frame`** pour qu'il accepte `max_year`
     et le transmette à `features.build_features(raw, max_year=max_year)` ;
  2. **construire les features en amont** avec
     `features.build_features(raw, max_year=2026)` **avant** de construire le
     cadre EDA. Attention : `build_eda_frame` rappelle `build_features` en
     interne ; le cadre utilisé pour les tests doit donc bien porter le
     `max_year` explicite, sinon `founding_year` et `founding_month` sont
     réécrits avec l'année courante.
- Dans les deux cas, documenter explicitement le choix `max_year=2026`.

### 2.6 Provenance

- Notebook : **71 cellules** (39 Markdown, 32 code), exécution `1…32`,
  **zéro erreur**, 12 figures intégrées.
- Assertions de base : 65 403 lignes brutes, 13 129 lignes labellisées,
  6 968 succès, 6 161 échecs.
- Commits : `2b03a39` (audit initial), `7804514` (statistiques descriptives).
  Les sections 15 à 19 sont dans l'état de travail de `feature/zm-eda` et ne
  sont pas commitées à la date de rédaction.

---

## 3. Ce qui existe déjà dans le dashboard

Le dashboard est déjà riche. **Ne pas dupliquer** les éléments suivants.

| Élément | Contenu existant | Fonctions principales |
|---|---|---|
| Vue d'ensemble | effectifs par issue, résumés numériques et catégoriels, part de janvier | `app_data.numeric_summary`, `categorical_summary`, `january_share` |
| Valeurs manquantes | taux de manquants par colonne, rôle de chaque colonne | `app_data.missing_table`, `variable_role` |
| Distributions | histogrammes, KDE, boîtes à moustaches, résumés par issue | `app_data.gaussian_kde`, `px.histogram` |
| Catégories | fréquences par issue, taux de succès par modalité, regroupement top-n | `app_data.frequency_by_issue`, `group_rare` |
| Corrélations | matrices Spearman/Pearson et V de Cramér | `app_data.correlation_matrix`, `cramers_v`, `cramers_v_matrix` |
| Géographie et tendances | carte des taux de succès par pays, tendance par année | `app_insights.success_by_country`, `success_by_year` |
| Filtres | issue, pays, secteur, années ; appliqués à tous les onglets | `app_data.apply_filters` |
| Aperçu | tableau des 200 premières lignes + téléchargement CSV | `dashboard_eda._preview` |

Le dashboard couvre donc déjà : statistiques descriptives, missingness globale,
histogrammes/KDE, fréquences catégorielles, taux de succès, corrélations,
géographie, tendances, filtres et export.

**Attention aux différences de logique :**

- La matrice de V de Cramér du dashboard utilise `group_rare(top_n=20)`.
  Le test du chi-deux du notebook utilise un **seuil de rareté** (< 30 lignes).
  Les deux ne produisent pas la même table et pas le même V.
- `frequency_by_issue` inclut les manquants comme modalité et conserve les
  lignes `operating` dans les totaux. Les tests doivent utiliser
  `modeling_rows` et exclure les manquants.
- Le dashboard affiche `log10_funding_total_usd` pour la lisibilité. Le
  Mann-Whitney du notebook porte sur `funding_total_usd` **en USD**, pas sur
  le logarithme.

---

## 4. Ce que cette passation ajoute

### 4.1 Phase 1 — inférence formelle

Un nouvel onglet **« Tests statistiques »** contenant :

1. **Test du chi-deux** : `primary_category` × `target`, avec regroupement des
   modalités rares, vérification des effectifs attendus, résidus de Pearson,
   V de Cramér et analyse de sensibilité.
2. **Test de Mann-Whitney** : `founding_year` et `funding_total_usd` entre
   succès et échecs, avec effectifs, manquants, médianes, U, p-value,
   r de rang-bisérial et probabilité de supériorité.
3. **Correction de Holm** sur la famille des trois tests primaires.
4. **Interprétation et limites** : signification statistique vs importance
   pratique, biais de sélection, missingness, fenêtre d'observation, absence de
   causalité.
5. **Valeurs de référence** de l'ensemble labellisé complet, affichées pour
   comparaison avec les résultats dynamiques.

### 4.2 Calcul dynamique vs valeurs de référence

- Les tests affichés dans l'onglet sont **recalculés sur la sélection filtrée
  courante**, restreinte à `app_data.modeling_rows(data)`.
- Les **valeurs de référence** de la section 6 portent sur l'ensemble labellisé
  complet (13 129 lignes, ou 12 052 pour le chi-deux après exclusion des
  manquants). Elles sont affichées dans un expander pour comparaison.
- Les deux peuvent différer quand des filtres sont actifs. C'est attendu.
- Afficher le périmètre courant : nombre de lignes, nombre de succès, nombre
  d'échecs, et éventuellement les manquants par variable.

### 4.3 Phase 2 — différée

Les éléments suivants ne font pas partie de cette passation :

- intervalles de confiance de Wilson sur les taux de succès par pays et par
  secteur (notebook §15.4–15.5) ;
- missingness différentielle par issue dans l'onglet « Valeurs manquantes »
  (notebook §15.6 pour le financement et §15.7 pour l'année de création) ;
- graphiques matplotlib du notebook : ils ne doivent pas être embarqués comme
  images. Si un graphique est souhaité, Wlid le réimplémente avec Plotly et
  `ui.chart`.

---

## 5. Méthodes statistiques

### 5.1 Test du chi-deux d'indépendance

**Question.** `primary_category` et `target` sont-ils indépendants ?

- **H0 :** les deux variables sont indépendantes.
- **H1 :** elles sont associées.
- Test **bilatéral**, sans correction de Yates (`correction=False`), car la
  table primaire est de forme k × 2 avec k = 66.

**Règle de regroupement des modalités rares.**

La colonne brute contient 503 modalités, dont une longue queue de modalités
très petites. La règle primaire est :

> Toute modalité avec **moins de 30 lignes labellisées** est regroupée dans une
> modalité résiduelle **« Other »** (affichée « Autres » dans l'interface).

Justification :

- c'est le seuil de rareté déjà utilisé dans la section 12 du notebook, donc
  l'analyse reste cohérente avec le travail descriptif ;
- cela garantit des effectifs attendus bien supérieurs au seuil de Cochran
  (minimum observé : 12.99) ;
- cela conserve 65 modalités interprétables au lieu de centaines de modalités
  instables.

La modalité « Other » est un résidu hétérogène : elle ne doit **jamais** être
interprétée comme un vrai secteur. Le choix du seuil étant une décision de
méthode, une analyse de sensibilité est fournie avec un seuil de **100 lignes**.

**Table de contingence.** Pour une table r × c d'effectifs observés
`O_ij`, les effectifs attendus sont :

```text
E_ij = (total ligne i) * (total colonne j) / n
```

La statistique et les degrés de liberté sont :

```text
chi2 = somme_ij (O_ij - E_ij)^2 / E_ij
ddl  = (r - 1) * (c - 1)
```

**Taille d'effet — V de Cramér :**

```text
V = sqrt( chi2 / (n * (min(r, c) - 1)) )
```

Pour une table 2 × k, `min(r, c) = 2`, donc `V = sqrt(chi2 / n)`. V varie de 0
(absence d'association) à 1. Ordre de grandeur indicatif : 0.1 petit, 0.3
modéré, 0.5 grand — mais ce repère dépend du contexte.

**Résidus de Pearson.** Pour diagnostiquer quelles modalités portent
l'association :

```text
residu_ij = (O_ij - E_ij) / sqrt(E_ij)
```

Un résidu positif signifie que la modalité est surreprésentée dans la classe
concernée ; un résidu négatif, qu'elle est sous-représentée. Le notebook
affiche les résidus de la colonne `target = 1` (succès), triés par valeur
absolue.

**Hypothèses et vérifications.**

- Observations indépendantes (hypothèse non testable ici ; les données sont
  observationnelles, pas un échantillon aléatoire).
- Effectifs attendus suffisants : règle de Cochran, aucune cellule < 5.
  Primaire : minimum 12.99, 0 cellule < 5, 0 cellule < 1.
  Sensibilité : minimum 43.29, 0 cellule < 5.
- Pas de cellules vides.
- Le test ne prouve pas une causalité.

### 5.2 Test de Mann-Whitney U

**Questions.**

- `founding_year` diffère-t-il entre succès et échecs ?
- `funding_total_usd` diffère-t-il entre succès et échecs ?

**Hypothèses.**

- **H0 :** les deux groupes ont la même distribution (égalité stochastique).
- **H1 :** les distributions diffèrent (test bilatéral).

**Pourquoi ce test.** Le financement est très asymétrique avec des valeurs
extrêmes ; `founding_year` est bornée et discrète. Le Mann-Whitney ne suppose
pas la normalité. Il est donc plus approprié qu'un test de Student.

**Gestion des manquants.** Chaque test utilise uniquement les lignes où la
variable est observée, séparément pour les succès et les échecs. Aucune
imputation. Les deux tests portent donc sur des sous-ensembles différents, et
les résultats sont conditionnels au fait d'être observé.

**Statistique et p-value.** `scipy.stats.mannwhitneyu` avec
`alternative="two-sided"` et `method="asymptotic"` (correction des ex æquo).
Le U retourné est celui du premier échantillon, ici les succès.

**Taille d'effet — r de rang-bisérial :**

```text
r_rb = 2 * U / (n_succes * n_echecs) - 1
```

- `r_rb > 0` : les succès tendent à avoir des valeurs **plus grandes**.
- `r_rb < 0` : les succès tendent à avoir des valeurs **plus petites**.
- `r_rb = 0` : pas de différence stochastique.

**Probabilité de supériorité :**

```text
P(succes > echec) = U / (n_succes * n_echecs)
```

C'est la probabilité qu'un succès tiré au hasard dépasse un échec tiré au
hasard (les ex æquo comptent pour moitié). Elle complète le r de rang-bisérial
et se lit plus facilement.

**Hypothèses.**

- Groupes indépendants.
- Approximation normale asymptotique avec correction des ex æquo.
- Pas d'hypothèse de normalité.
- Le test décrit les valeurs observées ; il ne corrige pas la missingness ni
  le biais de sélection.

### 5.3 Correction de Holm

**Problème.** Tester trois hypothèses au seuil 0.05 augmente le taux d'erreur
global de la famille (*family-wise error rate*, FWER) : la probabilité d'au
moins un faux positif n'est plus 5 %.

**Famille.** Les **trois tests primaires** :

1. chi-deux avec la règle de regroupement < 30 lignes ;
2. Mann-Whitney sur `founding_year` ;
3. Mann-Whitney sur `funding_total_usd`.

L'analyse de sensibilité (< 100 lignes) est une vérification de robustesse,
pas une hypothèse indépendante : elle ne fait pas partie de la famille. Les
analyses descriptives des sections 10 à 15 ne sont pas corrigées ici.

**Procédure step-down de Holm :**

1. trier les p-values brutes par ordre croissant ;
2. multiplier la plus petite par `m` (nombre de tests), la suivante par `m - 1`,
   etc. ;
3. imposer la monotonie par un maximum courant ;
4. plafonner chaque valeur à 1.

Holm contrôle le FWER sous dépendance arbitraire et est uniformément au moins
aussi puissant que Bonferroni.

**Implémentation.** `statsmodels` n'est pas installé dans l'environnement du
projet. `scipy.stats.false_discovery_control` implémente Benjamini-Hochberg,
**pas Holm** : ne pas l'utiliser pour cette correction. La fonction
`holm_adjust` de l'annexe B est la référence.

### 5.4 Règles de rapport

- Tous les tests sont **bilatéraux**.
- Toujours reporter : effectifs, statistique, degrés de liberté (chi-deux),
  p-value, taille d'effet et sens de l'effet.
- Les p-values qui sous-débordent doivent être affichées **`< 1e-300`**, jamais
  `0`.
- Ne pas confondre signification statistique et importance pratique : avec
  n > 12 000, une association modeste peut être très significative.
- Aucune affirmation causale.

---

## 6. Résultats de référence

Les valeurs ci-dessous sont celles de l'**ensemble labellisé complet**, telles
qu'exécutées dans le notebook. Elles servent de référence pour les tests
d'acceptation et pour l'expander « Valeurs de référence » du dashboard.

### 6.1 Périmètre

| Indicateur | Valeur |
|---|---:|
| Lignes labellisées | 13 129 |
| Succès (`target = 1`) | 6 968 |
| Échecs (`target = 0`) | 6 161 |
| Taux de succès global | 53.07 % |
| Lignes avec `primary_category` observé | 12 052 |
| Lignes avec `primary_category` manquant | 1 077 |

### 6.2 Test du chi-deux

| Règle de regroupement | n | Catégories | Lignes dans « Other » | chi2 | ddl | p-value | V de Cramér | Effectif attendu min. | Cellules < 5 | Cellules < 1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| < 30 lignes → Other | 12 052 | 66 | 2 415 | 715.992 | 65 | 2.215e-110 | 0.2437 | 12.99 | 0 | 0 |
| < 100 lignes → Other | 12 052 | 27 | 4 261 | 604.159 | 26 | 8.078e-111 | 0.2239 | 43.29 | 0 | 0 |

Lecture :

- L'association est statistiquement significative dans les deux spécifications.
- V ≈ 0.24 (primaire) et 0.22 (sensibilité) : association **petite à
  modérée**, pas un effet massif.
- Le minimum attendu est 12.99 dans la spécification primaire : la règle de
  Cochran est largement respectée.
- La conclusion est stable quand on change le seuil de regroupement.

### 6.3 Résidus de Pearson (colonne succès)

Top 10 des modalités par résidu absolu, spécification primaire (< 30 lignes) :

| Modalité | Résidu de Pearson |
|---|---:|
| Biotechnology | +8.17 |
| Enterprise Software | +5.12 |
| Other | -4.83 |
| Apps | -4.42 |
| Curated Web | -4.26 |
| Analytics | +3.77 |
| Semiconductors | +3.50 |
| Application Platforms | -3.37 |
| Software | +3.17 |
| Security | +3.17 |

Interprétation : l'association n'est pas un décalage uniforme de tous les
secteurs. Elle est portée par quelques modalités fortement surreprésentées
chez les succès (Biotechnology, Enterprise Software, Analytics,
Semiconductors) et quelques modalités sous-représentées (Apps, Curated Web,
Application Platforms). « Other » est un résidu hétérogène : son résidu négatif
signale que les petites modalités sont globalement moins souvent des succès,
mais il ne s'interprète pas comme un secteur.

### 6.4 Tests de Mann-Whitney

| Variable | n succès | n échecs | manquants succès | manquants échecs | médiane succès | médiane échecs | U | p-value | r rang-bisérial | P(succès > échec) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `founding_year` | 5 392 | 4 018 | 1 576 | 2 143 | 2004 | 2009 | 5 795 749.5 | < 1e-300 | -0.4650 | 0.2675 |
| `funding_total_usd` | 6 111 | 4 851 | 857 | 1 310 | 12 500 000 | 1 240 000 | 22 697 642.5 | < 1e-300 | 0.5313 | 0.7657 |

Lecture :

- **Année de création :** les succès sont **plus anciens**. Médiane 2004 contre
  2009 ; environ 73 % des paires succès-échec ont le succès fondé plus tôt.
  Le signe négatif du r de rang-bisérial est le sens de l'effet, pas une
  erreur.
- **Financement :** les succès tendent à avoir **plus de financement**.
  Médiane 12.5 M contre 1.24 M USD ; 76.6 % de probabilité de supériorité.
- Les deux p-values sous-débordent : afficher `< 1e-300`.
- Les effectifs manquants sont importants et **différentiels** entre succès et
  échecs : voir section 8.

### 6.5 Correction de Holm

| Test | p-value brute | p-value ajustée (Holm) | Significatif au seuil 0.05 |
|---|---:|---:|---|
| chi-deux, `primary_category` (< 30) | 2.215e-110 | 2.215e-110 | oui |
| Mann-Whitney, `founding_year` | < 1e-300 | < 1e-300 | oui |
| Mann-Whitney, `funding_total_usd` | < 1e-300 | < 1e-300 | oui |

La p-value du chi-deux est la plus grande des trois p-values brutes : le
multiplicateur de Holm vaut donc 1 et sa p-value ajustée est identique. Les
deux tests de Mann-Whitney restent sous `1e-300` après correction. La
correction ne change aucune conclusion, mais elle documente que le FWER est
contrôlé.

### 6.6 Valeurs attendues pour les tests d'acceptation

Wlid doit pouvoir retrouver exactement, sur l'ensemble labellisé complet :

- `n` chi-deux = **12 052**, catégories = **66**, « Other » = **2 415** ;
- `chi2` = **715.992**, `ddl` = **65**, `p` = **2.215e-110**, `V` = **0.2437**,
  attendu min. = **12.99**, cellules < 5 = **0** ;
- sensibilité : `chi2` = **604.159**, `ddl` = **26**, `p` = **8.078e-111**,
  `V` = **0.2239**, attendu min. = **43.29** ;
- Mann-Whitney `founding_year` : n = **5 392 / 4 018**, manquants =
  **1 576 / 2 143**, médianes = **2004 / 2009**, U = **5 795 749.5**,
  `p < 1e-300`, r = **-0.4650**, P(S > F) = **0.2675** ;
- Mann-Whitney `funding_total_usd` : n = **6 111 / 4 851**, manquants =
  **857 / 1 310**, médianes = **12 500 000 / 1 240 000**, U = **22 697 642.5**,
  `p < 1e-300`, r = **0.5313**, P(S > F) = **0.7657** ;
- Holm : les trois tests restent significatifs ; p ajustées
  **2.215e-110**, **< 1e-300**, **< 1e-300**.

---

## 7. Spécifications d'intégration

### 7.1 Nouvel onglet

Ajouter un septième onglet dans `src/dashboard_eda.py` :

```python
TABS = (
    ":material/dashboard: Vue d'ensemble",
    ":material/rule: Valeurs manquantes",
    ":material/bar_chart: Distributions",
    ":material/category: Catégories",
    ":material/hub: Corrélations",
    ":material/public: Géographie et tendances",
    ":material/science: Tests statistiques",   # nouveau
)
```

et l'ajouter à la liste des sections dans `render` :

```python
sections = (
    _overview,
    _missing_values,
    _distributions,
    _categories,
    _correlations,
    _geography,
    _inference,   # nouveau
)
```

La fonction `_inference(data)` reçoit la sélection **filtrée** retournée par
`_sidebar_filters`. Elle doit immédiatement restreindre le périmètre :

```python
scope = app_data.modeling_rows(data)
```

Si `scope` est vide, afficher :

```python
st.info("Aucune startup avec une issue connue dans cette sélection.")
```

et ne pas calculer les tests.

### 7.2 Calcul dynamique

Le calcul dynamique doit reproduire exactement les fonctions du notebook
(annexe B), avec les paramètres suivants :

- chi-deux primaire : `min_rows=30` ;
- chi-deux de sensibilité : `min_rows=100` ;
- Mann-Whitney : `alternative="two-sided"`, `method="asymptotic"` ;
- Holm : famille de trois p-values brutes, dans cet ordre :
  1. chi-deux primaire,
  2. Mann-Whitney `founding_year`,
  3. Mann-Whitney `funding_total_usd`.

Squelette :

```python
scope = app_data.modeling_rows(data)
if scope.empty or scope["target"].nunique() < 2:
    st.info("Aucune startup avec une issue connue dans cette sélection.")
    return

chi_primary = chi_square_report(
    scope["primary_category"], scope["target"], min_rows=30
)
chi_sensitivity = chi_square_report(
    scope["primary_category"], scope["target"], min_rows=100
)
mw_year = mann_whitney_report(
    scope.loc[scope["target"] == 1, "founding_year"],
    scope.loc[scope["target"] == 0, "founding_year"],
    "founding_year",
)
mw_funding = mann_whitney_report(
    scope.loc[scope["target"] == 1, "funding_total_usd"],
    scope.loc[scope["target"] == 0, "funding_total_usd"],
    "funding_total_usd",
)
adjusted = holm_adjust(
    [chi_primary["p"], mw_year["p-value"], mw_funding["p-value"]]
)
```

Afficher le périmètre courant :

```python
st.caption(
    f"Périmètre : {app_data.fmt_int(len(scope))} startups avec issue connue "
    f"({app_data.fmt_int(int((scope['target'] == 1).sum()))} succès, "
    f"{app_data.fmt_int(int((scope['target'] == 0).sum()))} échecs), après filtres."
)
```

Ajouter une note explicite :

> Les tests sont recalculés sur la sélection filtrée. Les valeurs de référence
> de l'expander portent sur l'ensemble labellisé complet (13 129 startups) ;
> un écart est attendu lorsque des filtres sont actifs.

Le regroupement < 30 est recalculé dans la sélection courante : le nombre de
catégories et de lignes dans « Other » peut donc différer des valeurs de
référence. De même, le `n` du chi-deux est celui de la table après exclusion
des `primary_category` manquants ; il peut être inférieur au `n` du périmètre
labellisé affiché au-dessus.

### 7.3 Valeurs de référence

Afficher les valeurs de la section 6 dans un expander :

```python
with st.expander("Valeurs de référence (ensemble labellisé)"):
    st.dataframe(reference_chi_table)
    st.dataframe(reference_mw_table)
    st.dataframe(reference_holm_table)
```

Ces tableaux peuvent être des constantes du module ou des DataFrames
construits à partir des valeurs de la section 6. Ils ne doivent **pas** être
recalculés sur la sélection filtrée.

### 7.4 Colonnes et libellés français

Utiliser des libellés français cohérents avec le reste du dashboard :

| Colonne technique | Libellé affiché |
|---|---|
| `test` | Test |
| `n success` | n succès |
| `n failure` | n échecs |
| `missing success` | manquants succès |
| `missing failure` | manquants échecs |
| `median success` | médiane succès |
| `median failure` | médiane échecs |
| `U statistic` | statistique U |
| `p-value` | p-value |
| `rank-biserial r` | r de rang-bisérial |
| `P(success > failure)` | P(succès > échec) |
| `grouping rule` | Règle de regroupement |
| `categories after grouping` | catégories après regroupement |
| `rows collapsed into 'Other'` | lignes regroupées dans « Autres » |
| `chi-square statistic` | statistique du chi-deux |
| `degrees of freedom` | degrés de liberté |
| `Cramer's V` | V de Cramér |
| `minimum expected count` | effectif attendu minimum |
| `cells with expected < 5` | cellules avec effectif attendu < 5 |
| `cells with expected < 1` | cellules avec effectif attendu < 1 |
| `Holm-adjusted p` | p-value ajustée (Holm) |
| `significant at 0.05` | significatif au seuil 0,05 |

Formatter les p-values avec une fonction dédiée :

```python
def format_p(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    if value == 0 or value < 1e-300:
        return "< 1e-300"
    return f"{value:.3e}".replace(".", ",")
```

Ne jamais afficher `0` pour une p-value sous-débordée. Pour les effectifs et les
médianes, réutiliser `app_data.fmt_int`, `app_data.fmt_pct` et `ui.fr_number`.

### 7.5 Cas limites à gérer

Les filtres peuvent rendre une table dégénérée. Wlid doit prévoir :

- `scope.empty` ou moins de deux classes dans `scope["target"]` → message
  d'information, pas de test.
- `primary_category` avec moins de deux modalités observées → message.
- Après regroupement, table avec moins de 2 lignes ou moins de 2 colonnes →
  message ; `chi2_contingency` ne doit pas être appelé.
- Effectif attendu < 5 → afficher le test mais avec un avertissement visible
  sur la validité de l'approximation ; effectif attendu < 1 → avertissement
  renforcé.
- Mann-Whitney avec un groupe vide, moins de deux observations, ou valeurs
  toutes identiques → message ; ne pas afficher un U ou un p-value trompeur.
- p-value `NaN` ou `None` → afficher « — ».
- `scipy` absent → message d'erreur explicite ; ajouter la dépendance (voir
  7.6).

### 7.6 Dépendances

- Ajouter `scipy>=1.11` à `requirements.txt` (Wlid).
- Ne pas ajouter `statsmodels` : la correction de Holm est implémentée
  manuellement dans l'annexe B.
- `pandas`, `numpy`, `plotly` et `streamlit` sont déjà présents.
- Aucun graphique n'est obligatoire en Phase 1 : les tableaux suffisent. Si un
  graphique de résidus est ajouté, utiliser `ui.chart` et Plotly, pas
  matplotlib.

### 7.7 Tests

Wlid ajoute un fichier de tests, par exemple `tests/test_app_inference.py`,
avec des données synthétiques :

- vérifier les formules sur une petite table connue (chi-deux, V, résidus,
  U, r de rang-bisérial, Holm) ;
- vérifier le regroupement `< 30` et `< 100` ;
- vérifier les cas limites (table 1 × 2, groupe vide, valeurs constantes) ;
- vérifier que les valeurs de référence de la section 6 sont reproduites sur
  le vrai jeu de données, si un test d'intégration le permet.

Les tests existants (`tests/test_app_smoke.py`) utilisent 240 lignes
synthétiques : ils ne suffisent pas à valider les tests statistiques, mais ils
doivent continuer à passer sans exception.

### 7.8 Documentation

Wlid met à jour `docs/app_integration.md` : dans la table intitulée **« Pages »**,
ajouter **« 7. Tests statistiques »** à la ligne existante **Dashboard EDA**
(qui liste déjà les éléments 1 à 6). Il ne s'agit pas d'une page applicative
séparée. ZM ne modifie pas ce document.

---

## 8. Limites et interprétation

### 8.1 Biais de sélection

- Les données sont **observationnelles**, pas un échantillon aléatoire.
- Le fichier commité a déjà retiré 965 lignes israéliennes. Cette attribution
  est fortement étayée par les effectifs (66 368 → 65 403 ; operating −760,
  closed −77, acquired −86, ipo −42 = 965), mais elle **attend encore la
  confirmation de TAS / du propriétaire du dépôt** : le CSV non filtré
  d'origine n'est pas disponible pour une comparaison octet par octet.
- `targets.prepare_target` exclut les startups `operating` : l'échantillon
  labellisé est conditionné au fait d'avoir connu une issue.
- Le contraste 1 vs 0 est un contraste **IPO/acquisition vs fermeture**, pas
  une mesure de qualité intrinsèque.

### 8.2 Missingness différentielle

- `founding_year` manque pour **34.8 %** des échecs contre **22.6 %** des
  succès.
- `funding_total_usd` manque pour **21.3 %** des échecs contre **12.3 %** des
  succès.
- Détail de ces deux missingness : notebook §15.6 (financement) et §15.7
  (année de création).
- Les tests en cas complets peuvent donc être biaisés. Aucune imputation n'est
  faite ; les résultats sont conditionnels au fait d'être observé.

### 8.3 Fenêtre d'observation

- Les données sont un instantané autour de 2016.
- Les startups jeunes ont eu moins de temps pour être acquises ou introduites
  en bourse ; les fermetures peuvent survenir plus tôt.
- La fin de la tendance par année doit être lue avec prudence.

### 8.4 Choix de regroupement

- Le chi-deux dépend de la règle de regroupement des modalités rares.
- L'analyse de sensibilité (< 100 lignes) est rassurante mais ne supprime pas
  l'hétérogénéité de « Other ».
- Le seuil < 30 est une décision de méthode, pas une vérité statistique.

### 8.5 Hypothèses des tests

- Chi-deux : observations indépendantes, effectifs attendus suffisants.
  Les effectifs sont vérifiés, mais l'indépendance et le biais de sélection ne
  sont pas corrigés par le test.
- Mann-Whitney : groupes indépendants, approximation asymptotique avec
  correction des ex æquo. Pas de normalité requise.
- Les p-values très petites ne disent rien sur la taille de l'effet : lire
  V, r de rang-bisérial et P(S > F).

### 8.6 Signification vs importance pratique

- Avec n > 12 000, même une petite association est statistiquement
  significative.
- Le chi-deux a un V ≈ 0.24 : association petite à modérée.
- Les effets de rang sont plus marqués : |r| ≈ 0.47 à 0.53.
- Le financement et l'année de création sont des **séparateurs descriptifs
  puissants**, mais ce sont des variables **interdites** en entrée du modèle :
  leur association est un avertissement de fuite temporelle, pas une
  opportunité de modélisation.

### 8.7 Absence de causalité

Aucun test de cette passation n'identifie un effet causal. Aucun modèle n'est
entraîné, aucune imputation, aucun encodage, aucune standardisation ne sont
effectués.

---

## 9. Répartition des responsabilités

| Tâche | ZM | Wlid |
|---|---|---|
| Analyse EDA et notebook | oui | non |
| Ce document de passation | oui | non |
| Choix des méthodes et valeurs de référence | oui | non |
| Implémentation de l'onglet « Tests statistiques » | non | oui |
| Fonctions pures dans `app_data.py` / `app_inference.py` | non | oui |
| Tests du dashboard | non | oui |
| Dépendances (`scipy`) | non | oui |
| Mise à jour de `docs/app_integration.md` | non | oui |
| Revue des résultats affichés | oui | oui |

ZM ne modifie pas `src/`, `tests/`, `requirements*.txt`, `data/` ni les
documents existants. Wlid reste propriétaire de l'implémentation et de
l'interface.

---

## 10. Références

- `notebooks/01_eda.ipynb` :
  - §10–12 : descriptif et profils ;
  - §15.1–15.3 : corrélations ;
  - §15.4–15.7 : taux par pays/secteur, financement et année par issue
    (Phase 2 pour les CI et la missingness différentielle) ;
  - §17.1 : chi-deux primaire ;
  - §17.2 : sensibilité ;
  - §17.3 : Mann-Whitney ;
  - §17.4 : Holm ;
  - §18 : synthèse et limites.
- `docs/methodology.md`, section 6 : tests statistiques attendus.
- `docs/handoff_preprocessing.md` : chaîne de données et contrat des variables.
- `docs/app_integration.md` : pages et organisation du dashboard.
- Commits : `2b03a39` (audit initial), `7804514` (statistiques descriptives).

---

## Annexe A — Formules

```text
Effectif attendu :
E_ij = (total ligne i) * (total colonne j) / n

Chi-deux :
chi2 = somme_ij (O_ij - E_ij)^2 / E_ij

Degrés de liberté :
ddl = (r - 1) * (c - 1)

V de Cramér :
V = sqrt( chi2 / (n * (min(r, c) - 1)) )

Résidu de Pearson :
residu_ij = (O_ij - E_ij) / sqrt(E_ij)

Rang-bisérial (U calculé avec les succès en premier échantillon) :
r_rb = 2 * U / (n_succes * n_echecs) - 1

Probabilité de supériorité :
P(succes > echec) = U / (n_succes * n_echecs)

Holm :
- trier p_(1) <= ... <= p_(m) ;
- p_ajustee_(i) = max_{j <= i} [ (m - j + 1) * p_(j) ] ;
- plafonner à 1.
```

---

## Annexe B — Implémentation de référence

Ces fonctions sont la référence du notebook. Wlid peut les copier dans
`src/app_data.py` ou dans un nouveau module `src/app_inference.py`, en
conservant les noms anglais.

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


def group_rare_categories(series: pd.Series, min_rows: int) -> pd.Series:
    """Regroupe les modalités ayant moins de ``min_rows`` lignes dans « Other »."""
    counts = series.value_counts()
    keep = counts[counts >= min_rows].index
    return series.mask(series.notna() & ~series.isin(keep), "Other")


def chi_square_report(series: pd.Series, target: pd.Series, min_rows: int) -> dict:
    grouped = group_rare_categories(series, min_rows)
    frame = pd.DataFrame({"category": grouped, "target": target}).dropna(
        subset=["category"]
    )
    table = pd.crosstab(frame["category"], frame["target"])
    chi2, p_value, dof, expected = chi2_contingency(table, correction=False)
    n = int(table.values.sum())
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    residuals = (table.values - expected) / np.sqrt(expected)
    success_column = list(table.columns).index(1)
    residual_series = pd.Series(residuals[:, success_column], index=table.index)
    summary = pd.DataFrame(
        [
            {"metric": "rows used (n)", "value": f"{n:,}"},
            {"metric": "categories after grouping", "value": f"{table.shape[0]:,}"},
            {
                "metric": "rows collapsed into 'Other'",
                "value": f"{int((grouped == 'Other').sum()):,}",
            },
            {"metric": "chi-square statistic", "value": f"{chi2:,.3f}"},
            {"metric": "degrees of freedom", "value": f"{dof:,}"},
            {"metric": "p-value", "value": f"{p_value:.3e}"},
            {"metric": "Cramer's V", "value": f"{cramers_v:.4f}"},
            {"metric": "minimum expected count", "value": f"{expected.min():,.2f}"},
            {"metric": "cells with expected < 5", "value": f"{int((expected < 5).sum()):,}"},
            {"metric": "cells with expected < 1", "value": f"{int((expected < 1).sum()):,}"},
        ]
    )
    return {
        "summary": summary,
        "table": table,
        "expected": expected,
        "residuals": residual_series.sort_values(key=np.abs, ascending=False),
        "chi2": chi2,
        "p": p_value,
        "dof": dof,
        "v": cramers_v,
        "n": n,
    }


def mann_whitney_report(
    success_values: pd.Series, failure_values: pd.Series, label: str
) -> dict:
    success = success_values.dropna()
    failure = failure_values.dropna()
    n_success, n_failure = len(success), len(failure)
    u_statistic, p_value = mannwhitneyu(
        success, failure, alternative="two-sided", method="asymptotic"
    )
    rank_biserial = 2 * u_statistic / (n_success * n_failure) - 1
    probability_superiority = u_statistic / (n_success * n_failure)
    return {
        "test": label,
        "n success": n_success,
        "n failure": n_failure,
        "missing success": int(success_values.isna().sum()),
        "missing failure": int(failure_values.isna().sum()),
        "median success": success.median(),
        "median failure": failure.median(),
        "U statistic": u_statistic,
        "p-value": p_value,
        "rank-biserial r": rank_biserial,
        "P(success > failure)": probability_superiority,
    }


def holm_adjust(p_values) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    m = len(values)
    order = np.argsort(values)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
```

Notes d'implémentation :

- `chi_square_report` suppose que la table contient les colonnes `0` et `1`.
  La validation (`table.shape[0] >= 2`, `table.shape[1] >= 2`, présence des
  deux classes `0` et `1`) doit être faite **à l'intérieur de
  `chi_square_report`, avant l'appel à `chi2_contingency`**, ou dans l'appelant
  **avant d'appeler `chi_square_report`**. Un contrôle placé seulement entre
  `chi_square_report` et `chi2_contingency` est trop tardif : la fonction
  exécute `list(table.columns).index(1)` avant d'atteindre `chi2_contingency`
  et peut lever une exception.
- `mann_whitney_report` suppose `n_success > 0` et `n_failure > 0`.
- `holm_adjust` reçoit les p-values brutes dans l'ordre de la famille ; il
  retourne un tableau dans le même ordre.
- `holm_adjust` suppose des p-values finies. Wlid doit traiter les `NaN`
  avant l'appel (les exclure de la famille ou afficher un message) ; sinon la
  correction propage `NaN`.
- Le label interne `"Other"` peut être remplacé par `"Autres"` pour
  l'affichage, à condition que ce libellé ne collisionne pas avec une
  modalité réelle. Dans le jeu actuel, aucune modalité ne s'appelle
  `Other`/`Autres` ; Wlid peut ajouter une assertion défensive.

---

## Annexe C — Checklist d'acceptation

- [ ] L'onglet « Tests statistiques » apparaît dans le dashboard.
- [ ] Les tests sont calculés sur `app_data.modeling_rows(data)`, donc sur le
      périmètre filtré et labellisé.
- [ ] Le chi-deux primaire reproduit : n = 12 052, 66 catégories,
      2 415 lignes dans « Other », chi2 = 715.992, ddl = 65,
      p = 2.215e-110, V = 0.2437, attendu min. = 12.99, 0 cellule < 5.
- [ ] La sensibilité reproduit : 27 catégories, 4 261 lignes dans « Other »,
      chi2 = 604.159, ddl = 26, p = 8.078e-111, V = 0.2239,
      attendu min. = 43.29.
- [ ] Les résidus de Pearson du top 10 correspondent à la section 6.3.
- [ ] Mann-Whitney `founding_year` reproduit : n = 5 392 / 4 018,
      manquants = 1 576 / 2 143, médianes = 2004 / 2009,
      U = 5 795 749.5, p < 1e-300, r = -0.4650, P(S > F) = 0.2675.
- [ ] Mann-Whitney `funding_total_usd` reproduit : n = 6 111 / 4 851,
      manquants = 857 / 1 310, médianes = 12 500 000 / 1 240 000,
      U = 22 697 642.5, p < 1e-300, r = 0.5313, P(S > F) = 0.7657.
- [ ] La correction de Holm reproduit les trois p-values ajustées et le fait
      que les trois tests restent significatifs.
- [ ] Les p-values sous-débordées sont affichées `< 1e-300`, jamais `0`.
- [ ] Les cas limites (sélection vide, une seule classe, table dégénérée,
      groupe constant) affichent un message et ne plantent pas.
- [ ] Les libellés sont en français ; les noms de variables restent en anglais.
- [ ] `scipy>=1.11` est déclaré dans `requirements.txt`.
- [ ] Les tests du dashboard passent (`python -m pytest -q`).
- [ ] Aucune variable interdite n'a été ajoutée aux entrées du modèle.
- [ ] Aucune affirmation causale n'est affichée.
