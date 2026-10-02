# Churn Prediction : Telco

## Problème business
Quels clients vont quitter l'entreprise, pourquoi, et que faire pour les retenir ?

## Données
Telco Customer Churn (IBM, Kaggle) : 7 043 clients, 21 variables.
Nettoyage : 11 lignes supprimées (TotalCharges vide, clients à tenure = 0).
Taux de churn : 26,5 % (classes déséquilibrées).

## Méthode
EDA → one-hot encoding → Régression Logistique, Random Forest, XGBoost → SHAP.
Split 80/20 stratifié (5 625 train / 1 407 test).

## Résultats
| Modèle | Recall | Precision | AUC |
|---|---|---|---|
| Régression Logistique | 0.80 | 0.49 | 0.835 |
| Random Forest | 0.66 | 0.56 | 0.821 |
| XGBoost | 0.78 | 0.49 | ... |

## Segmentation du risque (XGBoost, jeu de test)
| Segment | Clients | Churn réel |
|---|---|---|
| Faible | 646 | 6,8 % |
| Moyen | 277 | 22,4 % |
| Élevé | 484 | 55,4 % |

Le segment « Élevé » (34 % des clients) concentre ~72 % des churners.

## Insights clés
1. Contrat mensuel : 42,7 % de churn vs 11,3 % (1 an) et 2,8 % (2 ans).
2. Le churn se concentre dans les premiers mois (médiane 10 mois chez les partants vs 38).
3. Les partants paient plus cher (médiane ~80 vs ~65 par mois).
4. ... (ajoute ce que tu vois dans SHAP)

## Recommandations
...

## Impact estimé
Hypothèses : perte client 500 DH, offre de rétention 50 DH, taux de succès ... %.
Au seuil de ..., le coût passe de ... DH (contacter tout le monde) à ... DH.

## Visuels
![SHAP summary](images/shap_summary.png)
![SHAP waterfall](images/shap_waterfall.png)

## Reproduire
pip install -r requirements.txt, télécharger le CSV dans data/, lancer les notebooks.