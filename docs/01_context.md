# Contexte métier et cadrage ML

## Problème

Le projet étudie l'estimation du risque d'impayé d'un client télécom afin d'aider à prioriser des actions de suivi. Le jeu de données indiqué dans le dossier est **Telecom Credit Scoring Data (Nigeria)**; la cible annoncée est `default_risk`.

## Question ML

À partir des variables disponibles au moment de l'évaluation d'un client, peut-on estimer s'il présente le risque de défaut défini par la cible `default_risk` ? La définition exacte de cette cible, son horizon temporel et les valeurs possibles doivent être confirmés à M1 à partir de la documentation et des données.

## Type de problème

- Classification supervisée; binaire seulement si l'inspection confirme que `default_risk` ne contient que deux classes.
- La fréquence des classes n'est pas encore connue; mesurer le déséquilibre à M1.
- Le dossier annonce un jeu de données synthétique d'environ 500 000 lignes et 15 colonnes. Valider ces caractéristiques après téléchargement.

## Indicateurs

Rapporter au minimum précision, rappel, F1-score, ROC-AUC et matrice de confusion, conformément au livrable technique. Ajouter PR-AUC si la classe positive est rare. Le seuil de décision doit être discuté selon le coût des faux négatifs (impayés manqués) et des faux positifs (relances inutiles).

Les valeurs F1 ≥ 0,80 et rappel ≥ 0,85 ainsi que la latence p95 < 500 ms figurant dans le guide initial sont des **objectifs de travail proposés**, pas des seuils imposés par le dossier d'évaluation. Les confirmer ou les ajuster après baseline et mesure.

## Contraintes et gouvernance

- Utiliser uniquement des variables disponibles au moment où une prédiction serait faite; vérifier la fuite de cible et la temporalité des features.
- Documenter la provenance, licence, définition de la cible et les limites du jeu synthétique.
- Limiter les données enregistrées dans les logs; ne jamais exposer des attributs sensibles dans les réponses de l'API sans justification.
- Le dossier demande une gouvernance et l'auditabilité des décisions; définir les mesures concrètes après inspection du schéma.
- Mesurer la latence lorsque le serving sera implémenté. Le seuil de 500 ms du guide est indicatif tant qu'il n'a pas été validé comme exigence.

## Critères de succès du cadrage

À la sortie de M1, l'équipe connaît la définition opérationnelle de la cible, l'accès et les conditions d'utilisation des données, la qualité des colonnes et la baseline de classification. Les critères de qualité du modèle seront alors fixés à partir des résultats, du déséquilibre et du cas d'usage.
