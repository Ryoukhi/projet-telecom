# Data Card — Dataset Impayés Télécom (Nigeria)

## 1. Source
- **Origine** : [à préciser : Kaggle / formateur / synthétique]
- **Période** : assessment_date = 2025-09-30
- **Taille** : X lignes × 16 colonnes
- **Opérateurs** : MTN, 9mobile, Airtel, Glo
- **Zone** : Nigeria (montants en NGN)

## 2. Colonnes
| Colonne | Type | Description |
|---|---|---|
| customer_id | str | Identifiant (exclu des features) |
| phone_number | str | **PII — exclu** |
| operator | cat | Opérateur télécom |
| assessment_date | date | **Exclu** (constante) |
| credit_score | int | Score crédit 300–1000 |
| credit_rating | cat | poor/fair/good/excellent |
| tenure_months | int | Ancienneté |
| avg_monthly_spend_ngn | float | Dépense mensuelle |
| recharge_frequency_per_month | int | Fréquence recharge |
| payment_consistency_score | float | Consistance paiement (0–1) |
| late_payment_count | int | Nb retards |
| account_type | cat | prepaid/postpaid |
| loan_eligibility | bool | Éligibilité prêt |
| max_loan_amount_ngn | float | Montant max |
| default_risk | cat | **Cible source** : low/medium/high |

## 3. Recodage de la cible
`impaye = 1` si `default_risk ∈ {medium, high}`, sinon `0`.
Justification : medium+high = risque d'impayé avéré.

## 4. Prétraitements
1. Suppression `phone_number`, `assessment_date`
2. Dédoublonnage `customer_id`
3. Recodage cible binaire
4. Features dérivées : `spend_per_tenure`, `log_max_loan`, `log_avg_spend`, `late_per_tenure`, `consistency_x_late`
5. Imputation médiane (num) / mode (cat)
6. OneHotEncoder (cat), StandardScaler (num)

## 5. Split
- Train 70% / Val 15% / Test 15%, stratifié, graine 42

## 6. Déséquilibre
- Taux d'impayés : X% (à compléter après exécution)

## 7. Limites
- Dataset mono-date (2025-09-30) → pas de dimension temporelle
- Pas d'historique de paiement détaillé (seulement agrégats)
- Biais potentiel : surreprésentation de certains opérateurs
- Montants en NGN → conversion si déploiement FCFA

## 8. Conformité
- PII supprimées (`phone_number`)
- Identifiants conservés uniquement pour traçabilité
