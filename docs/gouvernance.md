# Gouvernance et Auditabilité du Modèle MLOps

## 1. Politique de Versioning
Pour garantir la reproductibilité, nous versionnons l'ensemble des artefacts du projet :
- **Code** : Versionné via Git (tags sémantiques v0.1.0 à v0.5.0). Chaque entraînement est lié à un commit Git précis.
- **Données** : Versionnées via DVC (`data/processed/features.parquet.dvc`). Le hash DVC est logué dans MLflow lors de l'entraînement.
- **Modèles** : Versionnés dans le MLflow Model Registry (nom : `impayes-xgboost`).
  - `Staging` : Modèle candidat, testé en pré-production.
  - `Production` : Modèle validé, utilisé par l'API.
  - `Archived` : Anciens modèles, conservés pour audit.
- **API** : Versionnée via Docker (tag d'image = hash du commit Git).

## 2. Gouvernance des Données (RGPD & Sécurité)
- **Anonymisation** : Les données PII (numéro de téléphone, nom) sont supprimées dès l'ingestion (`load_data.py`). Seul le `customer_id` (hashé) est conservé.
- **Minimisation** : Seules les features nécessaires à la prédiction sont conservées (11 features).
- **Traçabilité** : Chaque prédiction de l'API logue dans un fichier d'audit :
  - L'ID du client
  - Le timestamp
  - Les features d'entrée
  - La probabilité prédite
  - La version du modèle utilisée
- **Accès** : Les données brutes et les modèles sont stockés sur des serveurs sécurisés (S3 avec chiffrement au repos en production).

## 3. Auditabilité et Explicabilité
- **Explicabilité globale** : Les graphiques SHAP (générés au M2) sont conservés dans MLflow pour chaque version du modèle. Ils permettent d'expliquer quelles features influencent le plus les prédictions.
- **Explicabilité locale** : L'API pourrait exposer un endpoint `/explain` (via SHAP) pour justifier une décision individuelle (ex: "pourquoi ce client est prédit comme impayé ?").
- **Logs d'audit** : Les logs de l'API sont centralisés (dans un futur déploiement, via Prometheus/Grafana) et conservés 1 an pour répondre aux exigences réglementaires.

## 4. Gestion des Incidents et Réentraînement
- **Détection de dérive** : Un rapport Evidently est généré quotidiennement. Si plus de 30% des features dérivent, une alerte est envoyée à l'équipe MLOps.
- **Réentraînement** : Le réentraînement est **semi-automatique**. Le pipeline CI/CD/CT propose un nouveau modèle en `Staging`. Un Data Scientist doit valider manuellement la promotion en `Production` après avoir vérifié les métriques et le rapport SHAP.
- **Rollback** : En cas de régression en production, le modèle précédent est immédiatement restauré via MLflow Registry (`transition_model_version_stage` vers `Production` de la version N-1).
