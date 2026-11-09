# Robustesse des modeles avances (variation des folds)

## Protocole

- Variables : country_code, state_code, region, city, primary_category, founding_year
- `min_frequency` : 20
- Score : `f1_macro`
- Validation croisee : 5 folds stratifies
- Graines testees : 0, 1, 2
- Entrainement : 10503 lignes (le test n'est pas utilise)

## Resultats par graine

| Modele | Graine | F1 macro moyen | Ecart-type | Min | Max |
|---|---|---|---|---|---|
| random_forest | 0 | 0.6953 | 0.0052 | 0.6900 | 0.7025 |
| random_forest | 1 | 0.7003 | 0.0094 | 0.6893 | 0.7175 |
| random_forest | 2 | 0.6984 | 0.0047 | 0.6940 | 0.7073 |
| svm | 0 | 0.6967 | 0.0048 | 0.6873 | 0.7000 |
| svm | 1 | 0.6974 | 0.0108 | 0.6860 | 0.7174 |
| svm | 2 | 0.6991 | 0.0051 | 0.6934 | 0.7082 |

## Synthese

| Modele | Moyenne des graines | Ecart-type | Min | Max | Duree (s) |
|---|---|---|---|---|---|
| random_forest | 0.6980 | 0.0021 | 0.6953 | 0.7003 | 41.3 |
| svm | 0.6977 | 0.0010 | 0.6967 | 0.6991 | 349.0 |

## Rappel methodologique

- Seules les graines de decoupage changent ; les hyperparametres sont ceux choisis par `--run` sur le train.
- Le jeu de test n'est pas utilise : la robustesse se mesure sur le train.
- L'ecart observe indique la stabilite du score, ce n'est pas un intervalle de confiance statistique.
