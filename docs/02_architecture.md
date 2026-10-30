## Composants retenus

| Composant | Choix | Rôle dans ce projet |
|---|---|---|
| Langage | Python 3.10/3.11 | Un environnement accepté par les dépendances et CI. |
| Données/versioning | DVC (remote local au début) | Suivre les versions des données et éviter de stocker les gros fichiers dans Git. |
| Expériences/registre | MLflow | Conserver paramètres, métriques, artefacts et versions de modèle. |
| Orchestration | Prefect | Exécuter les étapes du pipeline et enregistrer leur état. |
| API | FastAPI + Uvicorn | Exposer une prédiction documentée par OpenAPI. |
| Conteneur | Docker | Reproduire l'environnement d'exécution de l'API. |
| Monitoring | Evidently; tableau de bord à cadrer au M6 | Produire des rapports de dérive et suivre la qualité du modèle. Grafana est une option, à retenir seulement si une source de métriques adaptée est ajoutée. |
| CI et qualité | GitHub Actions, pytest, Ruff, Black, pre-commit | Vérifier le code à chaque changement et faciliter les contrôles locaux. |

Les dépendances sont séparées dans `pyproject.toml` : `dev` ne contient que les outils de qualité; `data` (DVC), `ml`, `pipeline`, `serving` et `notebook` s'installent aux milestones correspondants. Le support DVC/S3 est optionnel et n'est pas requis pour le remote local.

## Architecture cible (à affiner après l'EDA)

```text
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Données     │───▶│ Pipeline     │───▶│ MLflow      │
│ brutes (DVC)│    │ Prefect      │    │ Tracking    │
└─────────────┘    └──────────────┘    └─────────────┘
                          │                    │
                          ▼                    ▼
                   ┌──────────────┐    ┌─────────────┐
                   │ Modèle .pkl  │───▶│ Registre    │
                   └──────────────┘    │ (Staging/   │
                          │            │  Production)│
                          ▼            └─────────────┘
                   ┌──────────────┐
                   │ API FastAPI  │
                   │ (Docker)     │
                   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐    ┌─────────────┐
                   │ Evidently    │───▶│ Grafana     │
                   │ (drift)      │    │ (alertes)   │
                   └──────────────┘    └─────────────┘
```
