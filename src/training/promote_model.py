"""Compare le modèle en Staging avec celui en Production et promeut le meilleur."""

import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

MODEL_NAME = "impayes-xgboost"
METRIC = "val_opt_f1"  # La métrique qu'on veut maximiser

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()


def get_stage_version(stage: str):
    """Récupère la version d'un modèle à un stage donné."""
    versions = client.get_latest_versions(MODEL_NAME, stages=[stage])
    return versions[0] if versions else None


def get_metric(run_id: str, metric_name: str):
    """Récupère une métrique d'un run MLflow."""
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_name, 0.0)


def promote():
    staging = get_stage_version("Staging")
    production = get_stage_version("Production")

    if not staging:
        logger.warning("Aucun modèle en Staging.")
        return

    staging_f1 = get_metric(staging.run_id, METRIC)
    logger.info(f"Staging v{staging.version} - {METRIC}: {staging_f1:.4f}")

    if not production:
        logger.info("Aucun modèle en Production. Promotion du Staging.")
        promote_to_production(staging)
        return

    production_f1 = get_metric(production.run_id, METRIC)
    logger.info(f"Production v{production.version} - {METRIC}: {production_f1:.4f}")

    if staging_f1 > production_f1:
        logger.info(
            f"Le Staging ({staging_f1:.4f}) est meilleur que la Production ({production_f1:.4f}). Promotion !"
        )
        promote_to_production(staging)
    else:
        logger.info("Le Staging n'est pas meilleur. Aucune promotion.")


def promote_to_production(version):
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version.version,
        stage="Production",
        archive_existing_versions=True,  # Archive l'ancien modèle en Production
    )
    logger.info(f" Modèle v{version.version} promu en Production.")


if __name__ == "__main__":
    promote()
