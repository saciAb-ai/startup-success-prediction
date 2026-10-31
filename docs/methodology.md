# MÃ©thodologie du mini-projet Data Mining

> **Projet :** Classification des issues observÃ©es des startups Ã  partir de leurs caractÃ©ristiques initiales  
> **Parcours :** Classification supervisÃ©e  
> **Dataset :** Crunchbase / Kaggle  
> **Statut :** Document initial de cadrage - projet Ã  dÃ©velopper Ã  partir de zÃ©ro

## 1. ProblÃ©matique et objectif business

**Ã‰tat actuel du projet :** le dÃ©pÃ´t GitHub a Ã©tÃ© crÃ©Ã© et clonÃ©, mais le dataset n'a pas encore Ã©tÃ© ajoutÃ© et aucun modÃ¨le n'a Ã©tÃ© entraÃ®nÃ©. Ce document dÃ©crit **ce que l'Ã©quipe prÃ©voit de rÃ©aliser**, et non un travail dÃ©jÃ  terminÃ©.

**Question Ã©tudiÃ©e :** parmi les startups historiques ayant connu une acquisition, une introduction en bourse ou une fermeture, peut-on distinguer leur issue Ã  partir de caractÃ©ristiques disponibles au dÃ©but de leur activitÃ© ?

**Objectif business :** fournir aux investisseurs et aux incubateurs un **outil exploratoire de comparaison des profils**. L'application indique si une startup prÃ©sente un profil davantage associÃ©, dans ce jeu de donnÃ©es historique, Ã  une **acquisition / IPO** ou Ã  une **fermeture**.

**Ce que le modÃ¨le ne fait pas :** il ne constate pas qu'une entreprise a actuellement rÃ©ussi, ne calcule pas le gain financier d'un investisseur et ne garantit pas le devenir d'une startup rÃ©cente.

## 2. Source et description du dataset

- **Source :** [Big Startup Success/Fail Dataset from Crunchbase (Kaggle)](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase/data).
- **Fichier Ã  rÃ©cupÃ©rer :** CSV d'origine depuis Kaggle ; vÃ©rifier son nom exact lors du tÃ©lÃ©chargement.
- **DonnÃ©es brutes annoncÃ©es :** **66 368 lignes et 14 colonnes**, effectifs Ã  confirmer sur le CSV tÃ©lÃ©chargÃ©.
- **Nature des donnÃ©es :** historique Crunchbase, ancien ; ne pas le prÃ©senter comme un Ã©chantillon reprÃ©sentatif des startups actuelles.

| Statut brut | Effectif annoncÃ© (Ã  vÃ©rifier) | Signification |
|---|---:|---|
| `operating` | 53 034 | Entreprise encore active dans la base |
| `acquired` | 5 549 | Entreprise acquise |
| `ipo` | 1 547 | Entreprise introduite en bourse |
| `closed` | 6 238 | Entreprise fermÃ©e |
| **Total** | **66 368** | |

Les dimensions annoncÃ©es satisfont les seuils du professeur (**au moins 10 000 lignes brutes et 10 attributs**). L'Ã©quipe vÃ©rifiera ces chiffres dÃ¨s le tÃ©lÃ©chargement et le chargement du CSV original.

## 3. ExpÃ©rience retenue : classification binaire unique

Nous rÃ©alisons **une seule expÃ©rience de classification**, avec deux classes :

| Statut d'origine | `target` | Classe interprÃ©tÃ©e |
|---|---:|---|
| `ipo` | **1** | SuccÃ¨s observÃ© (introduction en bourse) |
| `acquired` | **1** | SuccÃ¨s observÃ© (acquisition) |
| `closed` | **0** | Ã‰chec observÃ© (fermeture) |
| `operating` | **Exclu** | Issue non dÃ©terminÃ©e selon notre dÃ©finition |

**Pourquoi exclure `operating` ?** Une startup encore active n'est pas nÃ©cessairement rentable et n'a pas forcÃ©ment connu une acquisition ou une IPO. La ranger dans la classe Â« succÃ¨s Â» rendrait la cible ambiguÃ«.

**Effectifs attendus avant nettoyage :** 7 096 startups en classe 1 et 6 238 en classe 0, soit **13 334 lignes**, Ã  confirmer sur les donnÃ©es d'origine.

**Attention Ã  l'interprÃ©tation :** dans ce projet, Â« succÃ¨s Â» est une **convention de classification**, pas une preuve de rentabilitÃ©. L'exclusion des entreprises actives entraÃ®ne un **biais de sÃ©lection** : le modÃ¨le Ã©tudie uniquement les startups aux issues retenues. Il ne fournit donc pas la probabilitÃ© rÃ©elle de succÃ¨s de toutes les nouvelles startups.

## 4. EntrÃ©es (X) et sortie (y) du modÃ¨le

### EntrÃ©es proposÃ©es

| Variable de modÃ¨le | Origine | Utilisation |
|---|---|---|
| `country_code` | Colonne brute | Pays |
| `state_code` | Colonne brute | Ã‰tat / province |
| `region` | Colonne brute | RÃ©gion |
| `city` | Colonne brute | Ville |
| `primary_category` | PremiÃ¨re catÃ©gorie de `category_list` | Secteur d'activitÃ© |
| `founding_year` | AnnÃ©e extraite de `founded_at` | AnnÃ©e de crÃ©ation |

Ces variables sont une **liste initiale Ã  vÃ©rifier**. L'Ã©quipe doit contrÃ´ler les valeurs manquantes, les dates aberrantes, les catÃ©gories rares et la disponibilitÃ© des informations au moment auquel la prÃ©diction est censÃ©e Ãªtre faite.

**Variables exclues des entrÃ©es :** `status`, `target`, `funding_total_usd`, `funding_rounds`, `first_funding_at`, `last_funding_at` et les variables dÃ©rivÃ©es du financement total ou des financements futurs. Elles peuvent divulguer des informations postÃ©rieures Ã  la crÃ©ation.

### Sortie

- `1` : profil associÃ© aux startups ayant connu une **IPO ou une acquisition** dans l'Ã©chantillon historique.
- `0` : profil associÃ© aux startups **fermÃ©es** dans cet Ã©chantillon.

L'interface pourra afficher un score du modÃ¨le, mais **pas comme une probabilitÃ© garantie de rÃ©ussite future**.

## 5. Protocole de traitement et d'apprentissage

1. **Charger et contrÃ´ler les donnÃ©es brutes :** types, doublons, valeurs manquantes, incohÃ©rences de dates et rÃ©partition de `status`.
2. **Construire la cible :** ne conserver que `ipo`, `acquired` et `closed`, puis crÃ©er `target` selon la table de la section 3.
3. **PrÃ©parer les caractÃ©ristiques :** extraction de la catÃ©gorie principale, de l'annÃ©e et du mois de crÃ©ation ; examen des valeurs aberrantes.
4. **SÃ©parer les observations :** 80 % pour entraÃ®nement/validation et 20 % pour test final (`random_state=42`, stratification par classe). Conserver le test hors de toute optimisation.
5. **Construire le prÃ©traitement dans des `Pipeline` scikit-learn :** imputation, encodage `OneHotEncoder` et standardisation selon le modÃ¨le ; apprendre ces transformations sur les folds d'entraÃ®nement seulement.
6. **EntraÃ®ner les six modÃ¨les obligatoires :** Zero-R (`DummyClassifier`), KNN, Decision Tree, Random Forest, NaÃ¯ve Bayes et SVM.
7. **Optimiser les modÃ¨les rÃ©glables :** `GridSearchCV` ou `RandomizedSearchCV` avec validation croisÃ©e **K >= 5**. Zero-R est exemptÃ© d'optimisation.
8. **Ã‰valuer tous les modÃ¨les selon le mÃªme protocole :** Accuracy, Precision, Recall, F1, matrices de confusion et ROC-AUC ; utiliser si possible les mÃ©triques macro et les scores par classe.
9. **Comparer et interprÃ©ter les rÃ©sultats :** tableau multicritÃ¨re, visualisations, importance des caractÃ©ristiques lorsque pertinente et analyse des erreurs.
10. **IntÃ©grer le pipeline retenu dans Streamlit :** formulaire de saisie, prÃ©diction et dashboard EDA ; tester l'exÃ©cution locale.

## 6. EDA et tests statistiques obligatoires

Le dashboard EDA de l'application comprendra **six Ã©lÃ©ments** : statistiques descriptives ; taux de valeurs manquantes ; histogrammes et densitÃ©s des variables numÃ©riques ; frÃ©quences catÃ©gorielles ; matrice de corrÃ©lation ; au moins un filtre interactif.

**Deux tests statistiques prÃ©vus (Ã  justifier et Ã  valider) :**

- **Test du chi-deux (Ï‡Â²) d'indÃ©pendance :** rechercher une association entre le secteur (`primary_category`) et la classe `target`. VÃ©rifier les effectifs attendus et regrouper les catÃ©gories peu frÃ©quentes si nÃ©cessaire.
- **Test de Mann-Whitney U :** comparer les distributions de `founding_year` entre les deux classes, en prÃ©cisant les limites d'interprÃ©tation dues aux diffÃ©rentes durÃ©es d'observation.

Pour chaque test : dÃ©finir H0/H1, justifier le choix, fournir statistique et p-value, puis interprÃ©ter le rÃ©sultat. Un test statistique ne prouve pas une causalitÃ©.

## 7. Livrables attendus

| Livrable | Contenu minimal |
|---|---|
| `notebooks/` | EDA, prÃ©traitement, entraÃ®nement et Ã©valuation reproductibles |
| `src/` | Code de prÃ©traitement, modÃ¨les et application Streamlit |
| Application locale | Dashboard EDA Ã  six fonctions + formulaire de prÃ©diction avec score/confiance |
| `docs/Rapport_MiniProject.pdf` | ProblÃ©matique, mÃ©thodes, tests, rÃ©sultats et limites |
| `README.md` | Objectifs, dataset, installation locale, rÃ´les et rÃ©sultats |
| GitHub public | Historique de commits personnels rÃ©guliers et arborescence demandÃ©e |

Le dÃ©ploiement en ligne est **facultatif** ; le fonctionnement **local** est obligatoire. Les modÃ¨les de Deep Learning et les LLM ne peuvent pas servir de moteur principal de classification.

## 8. Limites et prÃ©cautions de conclusion

- Le jeu de donnÃ©es est **historique** : les relations apprises ne reprÃ©sentent pas nÃ©cessairement le marchÃ© actuel.
- L'absence des dates fiables de toutes les issues ne permet pas de garantir une prÃ©diction Ã  horizon fixe (par exemple, Â« rÃ©ussira dans cinq ans Â»).
- Les startups `operating` sont exclues : les scores du modÃ¨le sont **conditionnels Ã  cette sÃ©lection** et ne doivent pas Ãªtre interprÃ©tÃ©s comme des probabilitÃ©s de gain financier.
- Les variables prÃ©dictives peuvent Ãªtre corrÃ©lÃ©es au rÃ©sultat sans en Ãªtre la cause.

**Formulation recommandÃ©e pour la dÃ©monstration :** Â« Ce systÃ¨me classe le profil d'une startup par rapport aux issues observÃ©es dans un Ã©chantillon historique de Crunchbase. Il constitue une aide Ã  l'analyse, pas une prÃ©diction certaine de son avenir. Â»

---

**Remarque :** ce fichier prÃ©sente la mÃ©thodologie prÃ©vue. Les donnÃ©es seront chargÃ©es et prÃ©parÃ©es, puis les modÃ¨les seront entraÃ®nÃ©s et Ã©valuÃ©s par notre Ã©quipe. Aucun rÃ©sultat de performance n'est encore disponible.
