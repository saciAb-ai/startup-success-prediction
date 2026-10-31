# Méthodologie du mini-projet Data Mining

> **Projet :** Classification des issues observées des startups à partir de leurs caractéristiques initiales  
> **Parcours :** Classification supervisée  
> **Dataset :** Crunchbase / Kaggle  
> **Statut :** Document initial de cadrage - projet à développer à partir de zéro

## 1. Problématique et objectif business

**État actuel du projet :** le dépôt GitHub a été créé et cloné, mais le dataset n'a pas encore été ajouté et aucun modèle n'a été entraîné. Ce document décrit **ce que l'équipe prévoit de réaliser**, et non un travail déjà terminé.

**Question étudiée :** parmi les startups historiques ayant connu une acquisition, une introduction en bourse ou une fermeture, peut-on distinguer leur issue à partir de caractéristiques disponibles au début de leur activité ?

**Objectif business :** fournir aux investisseurs et aux incubateurs un **outil exploratoire de comparaison des profils**. L'application indique si une startup présente un profil davantage associé, dans ce jeu de données historique, à une **acquisition / IPO** ou à une **fermeture**.

**Ce que le modèle ne fait pas :** il ne constate pas qu'une entreprise a actuellement réussi, ne calcule pas le gain financier d'un investisseur et ne garantit pas le devenir d'une startup récente.

## 2. Source et description du dataset

- **Source :** [Big Startup Success/Fail Dataset from Crunchbase (Kaggle)](https://www.kaggle.com/datasets/yanmaksi/big-startup-secsees-fail-dataset-from-crunchbase/data).
- **Fichier à récupérer :** CSV d'origine depuis Kaggle ; vérifier son nom exact lors du téléchargement.
- **Données brutes annoncées :** **66 368 lignes et 14 colonnes**, effectifs à confirmer sur le CSV téléchargé.
- **Nature des données :** historique Crunchbase, ancien ; ne pas le présenter comme un échantillon représentatif des startups actuelles.

| Statut brut | Effectif annoncé (à vérifier) | Signification |
|---|---:|---|
| `operating` | 53 034 | Entreprise encore active dans la base |
| `acquired` | 5 549 | Entreprise acquise |
| `ipo` | 1 547 | Entreprise introduite en bourse |
| `closed` | 6 238 | Entreprise fermée |
| **Total** | **66 368** | |

Les dimensions annoncées satisfont les seuils du professeur (**au moins 10 000 lignes brutes et 10 attributs**). L'équipe vérifiera ces chiffres dès le téléchargement et le chargement du CSV original.

## 3. Expérience retenue : classification binaire unique

Nous réalisons **une seule expérience de classification**, avec deux classes :

| Statut d'origine | `target` | Classe interprétée |
|---|---:|---|
| `ipo` | **1** | Succès observé (introduction en bourse) |
| `acquired` | **1** | Succès observé (acquisition) |
| `closed` | **0** | Échec observé (fermeture) |
| `operating` | **Exclu** | Issue non déterminée selon notre définition |

**Pourquoi exclure `operating` ?** Une startup encore active n'est pas nécessairement rentable et n'a pas forcément connu une acquisition ou une IPO. La ranger dans la classe « succès » rendrait la cible ambiguë.

**Effectifs attendus avant nettoyage :** 7 096 startups en classe 1 et 6 238 en classe 0, soit **13 334 lignes**, à confirmer sur les données d'origine.

**Attention à l'interprétation :** dans ce projet, « succès » est une **convention de classification**, pas une preuve de rentabilité. L'exclusion des entreprises actives entraîne un **biais de sélection** : le modèle étudie uniquement les startups aux issues retenues. Il ne fournit donc pas la probabilité réelle de succès de toutes les nouvelles startups.

## 4. Entrées (X) et sortie (y) du modèle

### Entrées proposées

| Variable de modèle | Origine | Utilisation |
|---|---|---|
| `country_code` | Colonne brute | Pays |
| `state_code` | Colonne brute | État / province |
| `region` | Colonne brute | Région |
| `city` | Colonne brute | Ville |
| `primary_category` | Première catégorie de `category_list` | Secteur d'activité |
| `founding_year` | Année extraite de `founded_at` | Année de création |
| `` | Mois extrait de `founded_at` | Mois de création |

Ces variables sont une **liste initiale à vérifier**. L'équipe doit contrôler les valeurs manquantes, les dates aberrantes, les catégories rares et la disponibilité des informations au moment auquel la prédiction est censée être faite.

**Variables exclues des entrées :** `status`, `target`, `funding_total_usd`, `funding_rounds`, `first_funding_at`, `last_funding_at` et les variables dérivées du financement total ou des financements futurs. Elles peuvent divulguer des informations postérieures à la création.

### Sortie

- `1` : profil associé aux startups ayant connu une **IPO ou une acquisition** dans l'échantillon historique.
- `0` : profil associé aux startups **fermées** dans cet échantillon.

L'interface pourra afficher un score du modèle, mais **pas comme une probabilité garantie de réussite future**.

## 5. Protocole de traitement et d'apprentissage

1. **Charger et contrôler les données brutes :** types, doublons, valeurs manquantes, incohérences de dates et répartition de `status`.
2. **Construire la cible :** ne conserver que `ipo`, `acquired` et `closed`, puis créer `target` selon la table de la section 3.
3. **Préparer les caractéristiques :** extraction de la catégorie principale, de l'année et du mois de création ; examen des valeurs aberrantes.
4. **Séparer les observations :** 80 % pour entraînement/validation et 20 % pour test final (`random_state=42`, stratification par classe). Conserver le test hors de toute optimisation.
5. **Construire le prétraitement dans des `Pipeline` scikit-learn :** imputation, encodage `OneHotEncoder` et standardisation selon le modèle ; apprendre ces transformations sur les folds d'entraînement seulement.
6. **Entraîner les six modèles obligatoires :** Zero-R (`DummyClassifier`), KNN, Decision Tree, Random Forest, Naïve Bayes et SVM.
7. **Optimiser les modèles réglables :** `GridSearchCV` ou `RandomizedSearchCV` avec validation croisée **K >= 5**. Zero-R est exempté d'optimisation.
8. **Évaluer tous les modèles selon le même protocole :** Accuracy, Precision, Recall, F1, matrices de confusion et ROC-AUC ; utiliser si possible les métriques macro et les scores par classe.
9. **Comparer et interpréter les résultats :** tableau multicritère, visualisations, importance des caractéristiques lorsque pertinente et analyse des erreurs.
10. **Intégrer le pipeline retenu dans Streamlit :** formulaire de saisie, prédiction et dashboard EDA ; tester l'exécution locale.

## 6. EDA et tests statistiques obligatoires

Le dashboard EDA de l'application comprendra **six éléments** : statistiques descriptives ; taux de valeurs manquantes ; histogrammes et densités des variables numériques ; fréquences catégorielles ; matrice de corrélation ; au moins un filtre interactif.

**Deux tests statistiques prévus (à justifier et à valider) :**

- **Test du chi-deux (χ²) d'indépendance :** rechercher une association entre le secteur (`primary_category`) et la classe `target`. Vérifier les effectifs attendus et regrouper les catégories peu fréquentes si nécessaire.
- **Test de Mann-Whitney U :** comparer les distributions de `founding_year` entre les deux classes, en précisant les limites d'interprétation dues aux différentes durées d'observation.

Pour chaque test : définir H0/H1, justifier le choix, fournir statistique et p-value, puis interpréter le résultat. Un test statistique ne prouve pas une causalité.

## 7. Livrables attendus

| Livrable | Contenu minimal |
|---|---|
| `notebooks/` | EDA, prétraitement, entraînement et évaluation reproductibles |
| `src/` | Code de prétraitement, modèles et application Streamlit |
| Application locale | Dashboard EDA à six fonctions + formulaire de prédiction avec score/confiance |
| `docs/Rapport_MiniProject.pdf` | Problématique, méthodes, tests, résultats et limites |
| `README.md` | Objectifs, dataset, installation locale, rôles et résultats |
| GitHub public | Historique de commits personnels réguliers et arborescence demandée |

Le déploiement en ligne est **facultatif** ; le fonctionnement **local** est obligatoire. Les modèles de Deep Learning et les LLM ne peuvent pas servir de moteur principal de classification.

## 8. Limites et précautions de conclusion

- Le jeu de données est **historique** : les relations apprises ne représentent pas nécessairement le marché actuel.
- L'absence des dates fiables de toutes les issues ne permet pas de garantir une prédiction à horizon fixe (par exemple, « réussira dans cinq ans »).
- Les startups `operating` sont exclues : les scores du modèle sont **conditionnels à cette sélection** et ne doivent pas être interprétés comme des probabilités de gain financier.
- Les variables prédictives peuvent être corrélées au résultat sans en être la cause.

**Formulation recommandée pour la démonstration :** « Ce système classe le profil d'une startup par rapport aux issues observées dans un échantillon historique de Crunchbase. Il constitue une aide à l'analyse, pas une prédiction certaine de son avenir. »

---

**Remarque :** ce fichier présente la méthodologie prévue. Les données seront chargées et préparées, puis les modèles seront entraînés et évalués par notre équipe. Aucun résultat de performance n'est encore disponible.
