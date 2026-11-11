# Rapport de mini-projet Data Mining

## Prediction des issues observees des startups a partir de leurs caracteristiques initiales

| Champ | Valeur |
|---|---|
| Projet | Startup Success Prediction |
| Dataset | Crunchbase / Kaggle, fichier canonique `data/raw/startup_data.csv` |
| Tache | Classification binaire supervisee |
| Contribution EDA et statistique | CS30 (compte GitHub `nabilhhh`) |
| Date | 10 novembre 2026 |
| Depot | `github.com/saciAb-ai/startup-success-prediction` |

## 1. Problematique et objectif

La question etudiee est la suivante : parmi les startups historiques ayant connu une acquisition, une introduction en bourse ou une fermeture, peut-on distinguer leur issue a partir de caracteristiques disponibles au debut de leur activite ?

L'objectif est de fournir un outil exploratoire de comparaison de profils. Le modele indique si une startup ressemble davantage, dans cet echantillon historique, a une startup acquise ou cotee, ou a une startup fermee. Il ne calcule pas une probabilite garantie de succes futur et ne remplace pas une analyse financiere.

Deux precisions encadrent toute la lecture des resultats. D'une part, le mot "succes" designe ici une convention de classification (acquisition ou IPO), pas une preuve de rentabilite. D'autre part, les startups encore actives sont exclues : les scores sont donc conditionnels a cette selection et ne s'appliquent pas a l'ensemble des startups.

## 2. Dataset et preparation

### 2.1 Source et dimensions

Le fichier canonique de l'equipe contient 65 403 lignes et 14 colonnes brutes. Le chiffre de 66 368 lignes annonce dans la methodologie initiale etait une estimation : la valeur observee et verifiee est 65 403. La chaine officielle de preparation produit successivement :

| Etape | Lignes | Colonnes |
|---|---:|---:|
| Chargement brut (`data_loader.load_raw`) | 65 403 | 14 |
| Caracteristiques (`features.build_features`) | 65 403 | 17 |
| Cible (`targets.prepare_target`) | 13 129 | 18 |

La repartition des statuts bruts est la suivante :

| Statut brut | Lignes | Part |
|---|---:|---:|
| `operating` | 52 274 | 79,9 % |
| `closed` | 6 161 | 9,4 % |
| `acquired` | 5 463 | 8,4 % |
| `ipo` | 1 505 | 2,3 % |
| **Total** | **65 403** | **100 %** |

### 2.2 Definition de la cible

La cible binaire suit la table de la methodologie :

| Statut d'origine | `target` | Classe interpretee |
|---|---:|---|
| `ipo` | 1 | Succes observe (introduction en bourse) |
| `acquired` | 1 | Succes observe (acquisition) |
| `closed` | 0 | Echec observe (fermeture) |
| `operating` | exclu | Issue non determinee |

Le jeu labellise contient donc 13 129 lignes : 6 968 succes (53,07 %) et 6 161 fermetures (46,93 %). La classe 1 est legerement majoritaire, ce qui rend l'accuracy seule trompeuse ; les resultats sont donc commentes avec le F1 macro, l'accuracy equilibree et les metriques par classe.

L'exclusion de `operating` introduit un biais de selection : le modele n'apprend que sur des issues deja tranchees. Il ne repond pas a la question "cette startup va-t-elle reussir ?" pour une startup encore active.

### 2.3 Variables d'entree et exclusions

Le protocole retient six variables, toutes disponibles au debut de l'activite :

| Variable | Origine | Role |
|---|---|---|
| `country_code` | Colonne brute | Pays |
| `state_code` | Colonne brute | Etat ou province |
| `region` | Colonne brute | Region |
| `city` | Colonne brute | Ville |
| `primary_category` | Premiere categorie non vide de `category_list` | Secteur |
| `founding_year` | Annee extraite de `founded_at` | Annee de creation |

Sont exclues des entrees : `status` et `target` (la cible), ainsi que `funding_total_usd`, `funding_rounds`, `first_funding_at`, `last_funding_at` et les variables derivees du financement. Ces variables peuvent contenir une information posterieure a la creation et provoqueraient une fuite de donnees. Le module `splitting.py` refuse explicitement toute variable de la liste `FORBIDDEN_FEATURES`, et les tests ajoutes verifient ce refus pour chaque variable interdite.

`founding_month` est calculee pour l'analyse exploratoire mais volontairement exclue des entrees. La raison est documentee et mesuree : 42,6 % des dates valent exactement le 1er janvier et 56 % des mois valent janvier, car Crunchbase ne connait souvent que l'annee. Cette precision de date est liee a l'epoque du dossier (89,9 % de 01-01 avant 2000 contre 25,7 % apres 2012) et fortement associee a la cible (67,7 % de classe 1 contre 42,5 %). L'utiliser comme variable d'entree reviendrait a exploiter un artefact de saisie.

### 2.4 Valeurs manquantes et dates

Sur le jeu labellise, les manquants des variables du protocole sont les suivants :

| Variable | Manquants | Part |
|---|---:|---:|
| `founding_year` | 3 719 | 28,3 % |
| `state_code` | 2 190 | 16,7 % |
| `region` | 2 133 | 16,2 % |
| `city` | 2 133 | 16,2 % |
| `country_code` | 1 991 | 15,2 % |
| `primary_category` | 1 077 | 8,2 % |

Les dates sont controlees : une annee inferieure a 1900, superieure a l'annee de reference (2026) ou illisible est convertie en valeur manquante. Apres controle, `founding_year` varie de 1901 a 2016, avec une mediane a 2007. Aucune imputation n'est faite dans `features.build_features` : l'imputation et l'encodage sont appris dans les `Pipeline` scikit-learn, sur les seuls jeux d'entrainement, afin d'eviter toute fuite.

## 3. Analyse exploratoire et tests statistiques

Cette section resume la contribution EDA et statistique de CS30, realisee dans `notebooks/01_eda.ipynb` et documentee dans `docs/handoff_eda.md`. Toutes les valeurs ci-dessous sont issues de l'execution du notebook sur le fichier canonique.

### 3.1 Statistiques descriptives

| Variable | n | Manquants | Mediane | Moyenne | Ecart-type | Maximum | Asymetrie |
|---|---:|---:|---:|---:|---:|---:|---:|
| `funding_total_usd` (analyse) | 10 962 | 16,5 % | 5 762 380 | 36 147 146 | 371 764 439 | 30 079 503 000 | 59,84 |
| `funding_rounds` | 13 129 | 0 % | 1 | 1,86 | 1,52 | 18 | 2,81 |
| `founding_year` | 9 410 | 28,3 % | 2 007 | 2 004,80 | 9,22 | 2 016 | -3,67 |

Le financement total est extremement asymetrique : la moyenne est plus de six fois la mediane et l'asymetrie atteint 59,84. La mediane est donc un resume plus honnete que la moyenne. Le nombre de tours de financement est lui aussi fortement etale a droite (median 1, maximum 18). L'annee de creation est au contraire etalee a gauche, avec une masse concentree dans les annees 2000 et le debut des annees 2010. Ces variables de financement restent descriptives : elles ne sont pas des entrees de modele.

### 3.2 Frequences categoriques et cardinalite

| Variable | Niveaux | Niveau principal | Part du principal | Niveaux < 30 lignes | Part des lignes dans ces niveaux |
|---|---:|---|---:|---:|---:|
| `country_code` | 83 | USA | 73,4 % | 62 | 3,5 % |
| `city` | 1 671 | San Francisco | 7,8 % | 1 611 | 44,5 % |
| `primary_category` | 503 | Biotechnology | 9,0 % | 438 | 20,0 % |

La geographie est tres concentree sur les Etats-Unis, tandis que `city` est la variable la plus fragmentee : pres de la moitie des lignes appartiennent a des villes de moins de 30 observations. `primary_category` presente une longue trainee de petits secteurs. Cette cardinalite elevee est le principal risque de sur-apprentissage du projet et justifie le regroupement des modalites rares.

### 3.3 Correlations

Les correlations de rang de Spearman, calculees sur les lignes completes par paire, sont les suivantes :

| Paire | Spearman |
|---|---:|
| `funding_total_usd` vs `target` | 0,457 |
| `funding_rounds` vs `target` | 0,328 |
| `founding_year` vs `target` | -0,399 |

La correlation de Pearson entre `target` et `founding_year` vaut -0,299 sur les 9 410 lignes ou l'annee est observee. C'est la seule paire ou Pearson est defendable ici : la cible est binaire et l'annee est bornee, ce qui correspond a une correlation point-biseriale. Pour le financement, Pearson sous-estimerait fortement la relation (0,060 pour `funding_total_usd`) a cause de l'asymetrie et des valeurs extremes ; le rang de Spearman est plus adapte. Ces associations sont descriptives et ne prouvent aucune causalite.

### 3.4 Test du chi-deux : secteur et issue

Hypotheses : H0 affirme l'independance entre `primary_category` et `target` ; H1 affirme une association. Les modalites de moins de 30 lignes sont regroupees dans "Other", ce qui laisse 66 categories et 12 052 lignes exploitables.

| Indicateur | Valeur |
|---|---:|
| Effectif analyse | 12 052 |
| Categories apres regroupement | 66 |
| Statistique du chi-deux | 715,992 |
| Degres de liberte | 65 |
| p-value | 2,215 x 10^-110 |
| V de Cramer | 0,2437 |
| Effectif theorique minimal | 12,99 |
| Cellules avec effectif theorique < 5 | 0 |

L'hypothese d'independance est rejetee tres largement. Le V de Cramer de 0,244 indique toutefois une association d'intensite faible a moderee, pas une separation nette des classes. Une analyse de sensibilite avec un seuil de regroupement plus strict (moins de 100 lignes dans "Other") donne 27 categories, une statistique de 604,159 (ddl 26), une p-value de 8,078 x 10^-111 et un V de Cramer de 0,2239. La conclusion reste stable.

Les residus de Pearson les plus eleves concernent Biotechnology (+8,17), Enterprise Software (+5,12), Apps (-4,42) et Curated Web (-4,26). Certains secteurs sont donc sur-representes parmi les succes, d'autres parmi les fermetures, mais l'effet reste modeste.

### 3.5 Tests de Mann-Whitney

Le test de Mann-Whitney U compare deux groupes independants sans supposer la normalite. Il est adapte au financement, tres asymetrique, et a l'annee de creation, bornee et discrete. L'alternative est bilaterale.

| Variable | n succes | n fermetures | Mediane succes | Mediane fermeture | U | p-value | r rang-biserial | P(succes > fermeture) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `founding_year` | 5 392 | 4 018 | 2 004 | 2 009 | 5 795 749,5 | < 10^-300 | -0,4650 | 0,2675 |
| `funding_total_usd` | 6 111 | 4 851 | 12 500 000 | 1 240 000 | 22 697 642,5 | < 10^-300 | 0,5313 | 0,7657 |

Parmi les annees observees, les startups acquises ou cotees sont plus anciennes (mediane 2004 contre 2009). Le financement est lui aussi nettement plus eleve parmi les succes, mais cette variable est exclue des entrees pour cause de fuite temporelle. La difference d'annee doit etre lue avec prudence : les startups anciennes ont eu plus de temps pour connaitre une acquisition ou une IPO, et les manquants ne sont pas equilibres entre les classes (34,8 % des fermetures contre 22,6 % des succes n'ont pas d'annee exploitable).

### 3.6 Correction de Holm

Trois tests forment une famille. La correction de Holm controle le risque d'erreur de type I sur l'ensemble de la famille sans supposer l'independance des tests. Les trois p-values brutes sont deja extremement faibles ; apres correction, les trois hypotheses nulles restent rejetees au seuil de 0,05.

### 3.7 Limites de l'inference

Les tests etablissent des associations, pas des causes. Trois limites pesent sur l'interpretation : le biais de selection du a l'exclusion des startups actives ; la duree d'observation variable selon l'annee de creation ; et le desequilibre des valeurs manquantes entre succes et fermetures. Les conclusions restent donc conditionnelles a l'echantillon historique etudie.

## 4. Protocole de modelisation

### 4.1 Decoupage train/test

Le decoupage est stratifie, avec 80 % pour l'entrainement et 20 % pour le test final, et une graine fixee a 42. Sur le fichier canonique, il produit exactement 10 503 lignes d'entrainement et 2 626 lignes de test :

| Jeu | Lignes | Classe 0 (fermeture) | Classe 1 (succes) |
|---|---:|---:|---:|
| Entrainement | 10 503 | 4 929 | 5 574 |
| Test | 2 626 | 1 232 | 1 394 |

Le jeu de test est conserve hors de toute optimisation. Les hyperparametres sont choisis uniquement sur le train, par validation croisee stratifiee a 5 folds, avec le F1 macro comme critere de selection. Le test n'est utilise qu'une seule fois, apres la validation croisee. Le decoupage est reproductible : les tests ajoutes verifient que deux appels avec `random_state=42` donnent le meme partage, que les classes sont stratifiees, que les index sont disjoints et que les valeurs du test ne sont ni imputees ni transformees.

### 4.2 Pretraitement

Tout traitement qui apprend des parametres est place dans un `Pipeline` scikit-learn : imputation des variables categoriques par la modalite "Inconnu", imputation des variables numeriques par la mediane, encodage one-hot avec `handle_unknown="ignore"` et regroupement des modalites rares a `min_frequency=20`, puis standardisation pour les modeles sensibles a l'echelle (KNN et SVM). Le preprocesseur est re-appris a chaque fold de validation croisee, sur les seules donnees d'entrainement du fold.

### 4.3 Modeles et evaluation

Les six modeles imposes par la methodologie sont entraines : Zero-R (`DummyClassifier`), KNN, arbre de decision, Naive Bayes gaussien, Random Forest et SVM. Zero-R sert de plancher de reference : il predit toujours la classe majoritaire et n'a pas d'hyperparametre. Les autres modeles sont regles par recherche sur grille. Les metriques communes sont l'accuracy, l'accuracy equilibree, la precision, le rappel et le F1 macro, les metriques par classe, la matrice de confusion et le ROC-AUC.

## 5. Resultats experimentaux

Les resultats ci-dessous sont ceux des rapports valides du depot. Ils n'ont pas ete recalcules ni arrondis differemment pour ce rapport.

### 5.1 Comparaison globale sur le jeu de test

| Modele | F1 macro (CV) | F1 macro (test) | Accuracy equilibree (test) | Accuracy (test) | ROC-AUC (test) |
|---|---:|---:|---:|---:|---:|
| Zero-R | - | 0,3468 | 0,5000 | 0,5308 | 0,5000 |
| KNN | 0,6847 | 0,6806 | 0,6824 | 0,6908 | 0,7517 |
| Arbre de decision | 0,6856 | 0,6872 | 0,6868 | 0,6908 | 0,7548 |
| Naive Bayes | 0,6496 | 0,6372 | 0,6376 | 0,6379 | 0,6756 |
| Random Forest | 0,6984 | 0,6984 | 0,6979 | 0,7011 | 0,7764 |
| SVM | 0,6997 | 0,7012 | 0,7007 | 0,7049 | 0,7632 |

Les deux modeles avances arrivent en tete. Le Random Forest obtient un F1 macro de 0,6984 et un ROC-AUC de 0,7764 ; le SVM obtient un F1 macro de 0,7012 et un ROC-AUC de 0,7632. L'ecart de F1 entre les deux (0,0028) reste du meme ordre de grandeur que la variabilite observee entre graines de validation croisee (ecart-type de 0,0010 a 0,0021, etendue jusqu'a 0,0050) : il ne permet pas de declarer un modele significativement meilleur que l'autre. Zero-R confirme que la tache n'est pas triviale (F1 macro 0,3468), mais les scores restent moderes : environ 0,70 de F1 macro, soit une discrimination imparfaite.

### 5.2 Metriques par classe

| Modele | Classe | Precision | Rappel | F1 | Effectif |
|---|---|---:|---:|---:|---:|
| Random Forest | 0 (fermeture) | 0,6949 | 0,6469 | 0,6700 | 1 232 |
| Random Forest | 1 (succes) | 0,7059 | 0,7489 | 0,7268 | 1 394 |
| SVM | 0 (fermeture) | 0,7075 | 0,6323 | 0,6678 | 1 232 |
| SVM | 1 (succes) | 0,7030 | 0,7690 | 0,7345 | 1 394 |

Les deux modeles reperent mieux les succes que les fermetures : le rappel de la classe 1 atteint 0,7489 pour le Random Forest et 0,7690 pour le SVM, contre 0,6469 et 0,6323 pour la classe 0. Ce desequilibre est coherent avec la matrice de confusion et alimente l'analyse des erreurs.

### 5.3 Matrices de confusion

Lignes = classe reelle, colonnes = classe predite. TN = vraie fermeture predite fermeture, FP = fermeture predite succes, FN = succes predit fermeture, TP = succes predit succes.

| Modele | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Zero-R | 0 | 1 232 | 0 | 1 394 |
| KNN | 673 | 559 | 253 | 1 141 |
| Arbre de decision | 766 | 466 | 346 | 1 048 |
| Naive Bayes | 781 | 451 | 500 | 894 |
| Random Forest | 797 | 435 | 350 | 1 044 |
| SVM | 779 | 453 | 322 | 1 072 |

### 5.4 Stabilite de la validation croisee

La robustesse a ete mesuree en rejouant la validation croisee sur le train avec trois graines de folds differentes (0, 1 et 2), sans utiliser le test :

| Modele | F1 macro moyen | Ecart-type | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Random Forest | 0,6980 | 0,0021 | 0,6953 | 0,7003 |
| SVM | 0,6977 | 0,0010 | 0,6967 | 0,6991 |

Les scores sont stables : l'ecart-type entre graines est de 0,0021 pour le Random Forest et 0,0010 pour le SVM. Le F1 de validation croisee et le F1 de test sont presque identiques (0,6984 contre 0,6984 pour le Random Forest ; 0,6997 contre 0,7012 pour le SVM). Il n'y a donc pas de signe de surapprentissage severe au niveau agrege. En revanche, le plateau autour de 0,70 suggere une limite du signal disponible dans les six variables, plutot qu'un modele sur-ajuste.

### 5.5 Importance des variables

L'importance par impurete du Random Forest, agregee par variable d'origine, est la suivante :

| Variable | Importance cumulee |
|---|---:|
| `founding_year` | 0,2696 |
| `primary_category` | 0,2258 |
| `country_code` | 0,1334 |
| `city` | 0,1262 |
| `region` | 0,1244 |
| `state_code` | 0,1206 |

L'annee de creation et le secteur dominent. Les modalites liees a l'information manquante apparaissent aussi parmi les plus importantes (`city_Inconnu`, `country_code_Inconnu`, `region_Inconnu`). Cela signifie que l'absence d'information est elle-meme predictive dans cet echantillon, ce qui est un artefact de collecte et non une relation causale. Ces importances decrivent le modele ; elles ne prouvent pas qu'une variable cause l'issue.

## 6. Analyse des faux positifs et faux negatifs

Un faux positif (FP) est une fermeture predite comme succes : le profil est trop optimiste. Un faux negatif (FN) est un succes predit comme fermeture : le modele manque une reussite. Le tableau suivant exprime ces erreurs en part du jeu de test de 2 626 lignes.

| Modele | FP | FP (%) | FN | FN (%) |
|---|---:|---:|---:|---:|
| Zero-R | 1 232 | 46,9 % | 0 | 0,0 % |
| KNN | 559 | 21,3 % | 253 | 9,6 % |
| Arbre de decision | 466 | 17,7 % | 346 | 13,2 % |
| Naive Bayes | 451 | 17,2 % | 500 | 19,0 % |
| Random Forest | 435 | 16,6 % | 350 | 13,3 % |
| SVM | 453 | 17,3 % | 322 | 12,3 % |

Le Random Forest et le SVM produisent plus de faux positifs que de faux negatifs. Ils ont donc tendance a classer les profils ambigus du cote du succes, ce qui explique leur meilleur rappel sur la classe 1 et leur rappel plus faible sur la classe 0. Le KNN accentue ce biais (21,3 % de FP contre 9,6 % de FN). Naive Bayes est le seul modele dont les faux negatifs (19,0 %) depassent les faux positifs (17,2 %), ce qui degrade fortement son rappel sur la classe 1.

Sans matrice de couts metier, il n'est pas possible d'affirmer qu'un type d'erreur est plus grave qu'un autre. Un investisseur qui veut eviter les profils trop optimistes preferera un modele avec peu de FP ; un incubateur qui ne veut manquer aucun succes acceptera davantage de FP. Le choix final depend donc de l'usage. L'audit disponible ne permet pas non plus d'attribuer chaque erreur a une variable precise : les matrices de confusion donnent le volume des erreurs, pas leur explication individuelle.

## 7. Risques de sur-apprentissage et regularisation

Le principal risque vient de la cardinalite des variables categoriques. Avec 1 671 villes, 503 secteurs et 83 pays, un encodage one-hot complet produirait plusieurs milliers de colonnes, dont beaucoup ne contiennent que quelques observations. Un modele d'arbre pourrait memoriser ces modalites rares et obtenir un tres bon score sur le train sans generaliser. L'encodage one-hot aggrave aussi la dispersion : chaque ligne devient tres creuse, et les importances de variables deviennent instables.

Plusieurs choix limitent ce risque. Le regroupement `min_frequency=20` ramene l'encodage a environ 300 colonnes analysees dans le rapport d'importance, au lieu de plusieurs milliers. `handle_unknown="ignore"` evite qu'une modalite jamais vue fasse echouer la prediction. L'imputation et l'encodage sont ajustes dans chaque fold, jamais sur le test. Les modeles sont regularises ou volontairement simples : Random Forest avec 200 arbres et `min_samples_leaf=2` ; SVM a noyau RBF avec `C=1.0` et `gamma="scale"` ; arbre de decision limite a `max_depth=20` avec `min_samples_leaf=10` ; KNN avec 201 voisins, ce qui lisse fortement les predictions. Les configurations Random Forest, SVM et arbre utilisent aussi `class_weight="balanced"`.

La comparaison train / validation croisee / test ne montre pas d'ecart important : pour le Random Forest, le F1 macro est de 0,6984 en validation croisee et de 0,6984 sur le test ; pour le SVM, il passe de 0,6997 a 0,7012. Un surapprentissage severe produirait au contraire un score de validation croisee nettement superieur au score de test. Ici, le probleme semble plutot etre un plafond de performance du a la faible quantite d'information predictible dans les six variables. La stabilite entre graines de folds (ecart-type inferieur a 0,003) conforte ce diagnostic.

Deux precautions restent necessaires. D'abord, l'importance par impurete favorise les variables a forte cardinalite ; elle ne doit pas etre lue comme une preuve de causalite. Ensuite, le test n'a ete utilise qu'une fois : un seul decoupage ne permet pas de conclure a une stabilite parfaite sur d'autres echantillons. La validation croisee repetee sur le train est la seule mesure de robustesse disponible, et elle ne remplace pas un intervalle de confiance.

## 8. Limites et conclusion

Les limites principales sont les suivantes :

- Le dataset est historique : les relations apprises ne representent pas necessairement le marche actuel.
- Les startups `operating` sont exclues, ce qui biaise l'echantillon vers les issues deja tranchees.
- Les dates d'issue ne sont pas disponibles pour toutes les lignes : le modele ne fournit pas une prediction a horizon fixe.
- Les valeurs manquantes sont importantes (28,3 % pour l'annee de creation) et leur distribution differe entre succes et fermetures.
- Les associations statistiques ne sont pas des relations causales.
- Les scores ne sont pas des probabilites de succes futur ; ce sont des mesures de discrimination sur un echantillon donne.

Malgré ces limites, les resultats sont coherents et reproductibles. Le Random Forest et le SVM se detachent avec un F1 macro d'environ 0,70 et un ROC-AUC de 0,76 a 0,78, nettement au-dessus du plancher Zero-R (0,3468). Les modeles classiques (KNN, arbre, Naive Bayes) restent entre 0,64 et 0,69 de F1 macro. Le systeme constitue donc une aide exploratoire a la comparaison de profils, pas un outil de decision automatique.

## 9. Reproductibilite et contribution

Le notebook `notebooks/01_eda.ipynb` contient l'analyse exploratoire complete, les figures et les tests statistiques. Le document `docs/handoff_eda.md` decrit la transmission du dashboard. Les tests ajoutes par CS30 couvrent la construction de la cible, la construction des caracteristiques, le decoupage stratifie 80/20 avec `random_state=42`, la reproductibilite, l'absence de fuite et les metriques d'evaluation. La suite complete compte 144 tests, tous passes dans l'environnement du projet.

Commandes de verification :

```
python -m pytest -q
python src/data_loader.py data/raw/startup_data.csv
python src/train_advanced.py --run
python src/train_classical.py --run
```

Les resultats de modelisation de ce rapport proviennent des rapports valides du depot : `reports/advanced_models_results.md`, `reports/cs23_classical_report.md`, `reports/robustness.md` et `reports/feature_importance.md`. La contribution CS30 porte sur l'EDA, l'inference statistique, la qualite du protocole (tests) et la synthese de ce rapport ; les scores de modelisation sont ceux produits par les membres de l'equipe en charge des modeles.

## Annexe : provenance des chiffres

| Bloc du rapport | Source |
|---|---|
| Dimensions, statuts, cible, manquants, dates | `notebooks/01_eda.ipynb`, sections 1 a 9 |
| Statistiques descriptives, cardinalite, correlations | `notebooks/01_eda.ipynb`, sections 10 a 16 |
| Chi-deux, sensibilite, Mann-Whitney, Holm | `notebooks/01_eda.ipynb`, section 17 |
| Resultats Random Forest et SVM, metriques par classe, matrices | `reports/advanced_models_results.md` |
| Resultats Zero-R, KNN, arbre, Naive Bayes | `reports/cs23_classical_report.md` |
| Stabilite entre graines de folds | `reports/robustness.md` |
| Importance des variables | `reports/feature_importance.md` |
| Protocole et exigences | `docs/methodology.md` |
