# Feuille de route du projet MLOps

## But et périmètre

Construire une chaîne reproductible de prédiction du risque d'impayé (`default_risk`) à partir du jeu Telecom Credit Scoring Data (Nigeria), annoncé dans le dossier comme comprenant environ 500 000 lignes et 15 colonnes. Le résultat attendu est un prototype MLOps démontrable : données versionnées, entraînement traçable, modèle évalué et servi par API, conteneur déployable, automatisation et surveillance.

Les seuils de performance et de latence seront fixés après exploration des données et mesure d'une baseline. Les seuils F1/rappel et latence du guide M0 sont des objectifs proposés, pas des exigences chiffrées du dossier officiel; ils doivent être justifiés par les résultats et le cas d'usage.

## Découpage en milestones

| Milestone | Objectif | Livrables principaux | Porte de sortie |
|---|---|---|---|
| **M0 — Cadrage et setup** | Clarifier le problème, l'architecture, les responsabilités et l'environnement. | Dépôt structuré, contexte, architecture, guide de setup, CI de qualité, DVC initialisé ou décision documentée. | Une autre personne peut cloner le dépôt et reproduire l'environnement; limites et données d'accès sont documentées. |
| **M1 — Données et compréhension** | Récupérer et auditer les données sans fuite d'information. | Script d'ingestion, version/snapshot des données, data card, notebook EDA, rapport qualité et dictionnaire des colonnes. | Source, licence/accès, cible, types, valeurs manquantes, déséquilibre et risques de fuite sont expliqués. |
| **M2 — Features et baseline** | Construire une préparation reproductible et établir une référence mesurable. | Pipeline de prétraitement/features, split adapté, baseline scikit-learn, métriques et matrice de confusion. | Toutes les transformations sont apprises uniquement sur le train; un baseline peut être relancé depuis le code. |
| **M3 — Expériences et modèle retenu** | Comparer des modèles et justifier le choix selon l'usage métier. | Runs MLflow, comparaison des métriques, analyse d'erreurs, calibration/seuil documentés, modèle sérialisé. | Modèle évalué sur un jeu tenu à l'écart; précision, rappel, F1 et ROC-AUC (et PR-AUC si classes rares) sont rapportés. |
| **M4 — API et conteneur** | Rendre le modèle utilisable et reproductible. | API FastAPI avec validation d'entrée, prédiction et santé, documentation OpenAPI, Dockerfile et instructions de lancement. | Contrat d'entrée/sortie et erreurs sont documentés; image exécutable avec modèle/configuration identifiés. |
| **M5 — CI/CD/CT et gouvernance** | Automatiser les contrôles et sécuriser le cycle de vie du modèle. | Workflow CI (lint/tests), pipeline de réentraînement/validation, registre ou équivalent, règles de versioning et audit. | Un modèle ne passe en production qu'après validations automatiques et revue/approbation prévues. |
| **M6 — Monitoring et stratégie de déploiement** | Détecter les problèmes en production simulée et pouvoir revenir en arrière. | Dashboard, métriques techniques et métier, rapport de drift, seuils/alertes, plan canari/A-B et rollback. | Scénarios de latence, data drift et baisse de performance ont une interprétation et une action définies; le drift seul ne déclenche pas un remplacement automatique. |
| **M7 — Dossier final et soutenance** | Assembler une démonstration claire, vérifiable et cohérente. | Rapport PDF/DOCX répondant au dossier, diagramme d'architecture, résultats, limites, présentation PowerPoint et scénario de démo. | Chaque affirmation chiffrée renvoie à un artefact; démo répétée, rôles de parole et temps sont préparés. |

## Ordre de travail recommandé

1. Terminer M0 en premier, en résolvant l'environnement avant d'ajouter des dépendances supplémentaires.
2. Commencer M1 par l'accès réel au jeu de données, sa licence et son schéma; adapter les hypothèses métier à la définition effective de `default_risk`.
3. Geler une baseline avant toute optimisation. Garder un jeu de test final non utilisé pour le choix du modèle.
4. Construire API, conteneur et monitoring à partir du même pipeline de features et du même artefact versionné.
5. Préparer rapport et slides en parallèle dès M1 avec les décisions, résultats et figures, puis finaliser en M7.

## Livrables finaux à cocher

- [ ] Dépôt Git : code, scripts, notebooks, tests et documentation.
- [ ] Données ou échantillon versionné avec DVC ou équivalent; prétraitements documentés.
- [ ] Modèle entraîné, évalué et sérialisé; métriques et rapport d'expériences.
- [ ] API avec endpoint de prédiction et documentation OpenAPI.
- [ ] Image Docker reproductible.
- [ ] Pipeline CI/CD/CT documenté et automatisé.
- [ ] Registre de versions/statuts du modèle, ou mécanisme équivalent clairement démontré.
- [ ] Monitoring technique et métier, drift et alertes.
- [ ] Rapport PDF/DOCX : architecture, pipeline, monitoring et gouvernance.
- [ ] Présentation PowerPoint et démonstration préparées pour la soutenance.

## Risques et décisions à tenir à jour

- Définition exacte de `default_risk`, disponibilité des labels dans le temps et éventuelle fuite de cible.
- Accès au dataset, licence, modalités de téléchargement et stratégie de versionnement d'un gros fichier.
- Déséquilibre des classes et arbitrage entre faux négatifs et coût des relances.
- Gouvernance : données personnelles/sensibles, minimisation, durée de conservation et audit des décisions.
- Périmètre réaliste : privilégier une chaîne de bout en bout fonctionnelle plutôt que multiplier les outils sans rôle démontré.

## Statut actuel de M0

- Présent : structure de dossiers, `docs/01_context.md`, `docs/02_architecture.md`, feuille de route, setup Git Bash, `.gitignore`, `.env.example`, `pyproject.toml` avec extras, README, configuration pre-commit et CI.
- Environnement : `.venv-m0` a été créé sans supprimer `.venv`. L'installation `pip install -e ".[dev]"` n'est pas encore validée : pip reste bloqué pendant le téléchargement de setuptools dans l'environnement observé. Une tentative sans isolation a confirmé que l'editable metadata se construit, puis a échoué parce que `wheel` manquait; le backend de build déclare maintenant setuptools et wheel.
- À vérifier sur la machine de l'équipe : terminer l'installation de `.[dev]`, exécuter pre-commit, puis confirmer le workflow CI.
- En attente : DVC n'est pas initialisé; initialiser DVC et son remote local après installation de l'extra `pipeline` dans un environnement fonctionnel.
- Git : au début de la reprise, `pyproject.toml`, `requirements.lock`, cette feuille de route et le guide de setup étaient non suivis. `requirements.lock` est vide; ne pas l'ajouter comme s'il était un lockfile valide.
- Fin du M0 : confirmer que l'environnement est reproductible sur une machine propre, DVC fonctionne avec un remote documenté, et CI passe; créer alors le tag `v0.0.1` si l'équipe utilise les tags de jalon.
