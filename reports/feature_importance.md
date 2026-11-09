# Importance des variables - Random Forest

## Protocole

- Variables : country_code, state_code, region, city, primary_category, founding_year
- Exclues : founding_month (EDA uniquement)
- `min_frequency` : 20
- Colonnes one-hot analysees : 300
- Entrainement : 10503 lignes (le test n'est pas utilise)

## Importance par variable d'origine

| Variable | Importance cumulee |
|---|---|
| founding_year | 0.2696 |
| primary_category | 0.2258 |
| country_code | 0.1334 |
| city | 0.1262 |
| region | 0.1244 |
| state_code | 0.1206 |

## Principales modalites

| Rang | Colonne | Variable | Importance |
|---|---|---|---|
| 1 | `num__founding_year` | founding_year | 0.2696 |
| 2 | `cat__city_Inconnu` | city | 0.0516 |
| 3 | `cat__country_code_USA` | country_code | 0.0477 |
| 4 | `cat__primary_category_Inconnu` | primary_category | 0.0473 |
| 5 | `cat__country_code_Inconnu` | country_code | 0.0444 |
| 6 | `cat__region_Inconnu` | region | 0.0383 |
| 7 | `cat__state_code_Inconnu` | state_code | 0.0370 |
| 8 | `cat__primary_category_Biotechnology` | primary_category | 0.0234 |
| 9 | `cat__region_SF Bay Area` | region | 0.0136 |
| 10 | `cat__region_infrequent_sklearn` | region | 0.0126 |
| 11 | `cat__state_code_CA` | state_code | 0.0110 |
| 12 | `cat__city_infrequent_sklearn` | city | 0.0096 |
| 13 | `cat__country_code_RUS` | country_code | 0.0093 |
| 14 | `cat__primary_category_Curated Web` | primary_category | 0.0085 |
| 15 | `cat__primary_category_Enterprise Software` | primary_category | 0.0083 |
| 16 | `cat__primary_category_infrequent_sklearn` | primary_category | 0.0082 |
| 17 | `cat__state_code_infrequent_sklearn` | state_code | 0.0079 |
| 18 | `cat__primary_category_Software` | primary_category | 0.0071 |
| 19 | `cat__primary_category_Analytics` | primary_category | 0.0068 |
| 20 | `cat__country_code_infrequent_sklearn` | country_code | 0.0056 |

## Rappel methodologique

- Le jeu de test n'est pas utilise : analyse descriptive sur le train.
- `feature_importances_` mesure la baisse d'impurete dans les arbres ; elle favorise les variables a forte cardinalite (`city`, `region`).
- Ces importances decrivent le modele, elles ne prouvent pas une relation causale entre la variable et l'issue de la startup.
