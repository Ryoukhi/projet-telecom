1. Problème métier

    Un opérateur télécom souhaite anticiper les impayés de ses clients afin de déclencher des actions de recouvrement proactives (relances ciblées, échelonnement, offres adaptées) plutôt que réactives.

2. Question ML

    « Étant donné l'historique d'un client (factures, paiements, ancienneté, profil), quelle est la probabilité qu'il devienne impayé au prochain cycle de facturation ? »

3. Type de problème

    Classification binaire : impayé (1) / non impayé (0)

    Déséquilibre de classes attendu (les impayés sont minoritaires → attention au rappel)

4. KPIs métier à optimiser
KPI	Description	Lien avec la métrique ML
Taux de recouvrement	% d'impayés récupérés	Rappel élevé (ne pas rater d'impayés)
Coût de relance	Coût des actions de recouvrement	Précision (éviter les faux positifs coûteux)
Taux de faux négatifs	Impayés non détectés	Priorité métier n°1

5. Contraintes

    Latence de prédiction < 500 ms (usage en batch quotidien ou API temps réel)

    Explicabilité requise (RGPD, justification d'une relance)

    Données sensibles → gouvernance à prévoir

6. Critères de succès du projet

    F1-score ≥ 0.80 en production

    Rappel ≥ 0.85 sur la classe "impayé"

    Latence p95 < 500 ms