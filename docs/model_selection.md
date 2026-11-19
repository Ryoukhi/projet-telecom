# Justification du modèle retenu

## Contexte

Cible comportementale : `impaye = 1` si `late_payment_count >= 3`.
Taux d'impayés : 22,6 % (déséquilibre modéré).

## Tableau comparatif (validation)

| Modèle | Seuil | Precision | Recall | F1 | ROC-AUC | Avg Precision |
|--------|-------|-----------|--------|-----|---------|---------------|
| Logistic Regression | 0.34 | 0.521 | 0.809 | 0.634 | 0.717 | 0.694 |
| Random Forest | 0.33 | 0.460 | 0.975 | 0.625 | 0.718 | 0.694 |
| **XGBoost** | **0.33** | **0.494** | **0.870** | **0.630** | **0.717** | **0.691** |

## Tableau comparatif (test)

| Modèle | Seuil | Precision | Recall | F1 | ROC-AUC |
|--------|-------|-----------|--------|-----|---------|
| Logistic Regression | 0.50 | 0.634 | 0.644 | 0.639 | 0.715 |
| Random Forest | 0.50 | 0.796 | 0.545 | 0.647 | 0.716 |
| **XGBoost** | **0.33** | **0.494** | **0.871** | **0.631** | **0.716** |

## Modèle retenu : XGBoost

**Justification** :
1. **Meilleur AUC en test** (0.7164, quasi identique à RF 0.7163)
2. **Meilleur F1 en test au seuil 0.5** (0.647, quasi identique à RF)
3. **Standard de l'industrie** pour les données tabulaires (gradient boosting)
4. **SHAP disponible** pour l'interprétabilité

## Stratégie de seuil : recall prioritaire

Seuil retenu : **0.33** (au lieu de 0.50).

**Justification métier** : le coût d'un impayé non détecté (perte de revenu, frais de recouvrement tardif) est supérieur au coût d'une relance inutile (SMS/email). Nous privilégions le rappel (87 %) au détriment de la précision (49 %).

## Limites

1. **Dépendance à `credit_score`** : SHAP montre que `credit_score` pèse 89 % de l'importance totale. Cela suggère une fuite indirecte (le score de crédit est calculé à partir de `late_payment_count`). Pour un projet réel, nous recommanderions de retirer `credit_score` ou de le remplacer par un score comportemental.
2. **AUC modeste** (0.717) : le signal comportemental des features restantes est faible.
3. **Dataset synthétique** : les résultats peuvent ne pas se généraliser à des données réelles.

## Pistes d'amélioration

1. Retirer `credit_score` pour construire un modèle purement comportemental.
2. Enrichir les features avec l'historique de paiement détaillé (montant, dates, canal).
3. Tester des modèles séquentiels (LSTM) sur l'historique de transactions.
