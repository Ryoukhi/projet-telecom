# Fiche courte du dataset

## Présentation

Le projet utilise `telecom_credit_scoring_nigeria.csv`, un jeu de données de scoring client télécom au Nigeria. Le fichier présent dans le dépôt contient **500 000 enregistrements et 15 colonnes**. Les montants sont exprimés en nairas nigérians (NGN) et les opérateurs représentés sont MTN, Airtel, Glo et 9mobile.

La documentation du projet décrit les données comme synthétiques, mais la source exacte et la licence ne sont pas encore renseignées. Les résultats doivent donc être présentés comme ceux d’un prototype, sans supposer qu’ils se généralisent à de vrais clients.

## Ce que décrivent les colonnes

| Groupe | Colonnes | Rôle |
|---|---|---|
| Identifiants et contexte | `customer_id`, `phone_number`, `assessment_date` | Identifiant client, numéro de téléphone et date d’évaluation. Le numéro est une donnée personnelle et doit être retiré avant l’analyse. |
| Profil télécom | `operator`, `tenure_months`, `account_type` | Opérateur, ancienneté et type de compte prépayé ou postpayé. |
| Profil financier | `credit_score`, `credit_rating`, `avg_monthly_spend_ngn`, `max_loan_amount_ngn`, `loan_eligibility` | Score et catégorie de crédit, dépenses mensuelles, éligibilité et montant de prêt maximal. |
| Historique de paiement agrégé | `recharge_frequency_per_month`, `payment_consistency_score`, `late_payment_count` | Fréquence de recharge, régularité des paiements et nombre de retards. |
| Cible source | `default_risk` | Niveau de risque fourni dans le fichier (`low`, `medium` ou `high`). C’est l’information que le modèle cherche à estimer après recodage. |

## Préparation pour le modèle

Le code d’ingestion supprime `phone_number` et `assessment_date`, dédoublonne les lignes par `customer_id`, puis crée une cible binaire `impaye` : `1` pour les niveaux `medium` ou `high` de `default_risk`, `0` pour `low`. Les variables catégorielles sont encodées et les valeurs manquantes sont imputées dans les étapes de préparation.

## Points de vigilance

- **Définition de la cible à harmoniser :** le code d’ingestion utilise `default_risk` (`medium` ou `high`), tandis que `docs/model_selection.md` décrit `impaye = 1` si `late_payment_count >= 3`. Ces définitions ne sont pas équivalentes ; il faut en confirmer une seule avant de présenter les métriques comme définitives.
- **Fuite indirecte possible :** le projet indique que `credit_score` est fortement lié aux retards de paiement. Si ce score est calculé à partir de retards déjà observés, il peut donner au modèle une information trop proche de la cible.
- **Instantané unique :** les données décrivent une date d’évaluation et fournissent surtout des agrégats, pas un historique détaillé des paiements.
- **Provenance et licence :** à documenter avant toute réutilisation ou diffusion des données.

Les données brutes et les fichiers préparés sont gérés localement/DVC ; ils ne doivent pas être publiés avec les données personnelles.
