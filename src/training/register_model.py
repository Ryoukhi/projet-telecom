"""Enregistre le modèle XGBoost local dans le Model Registry MLflow."""

import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger
import joblib

MODEL_NAME = "impayes-xgboost"
MODEL_PATH = "models/model.pkl"

# On se connecte au serveur MLflow local avec SQLite
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()


def register_model():
    """Enregistre le modèle local et le passe en Staging."""
    logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # 1. Créer un nouveau run "factice" pour héberger l'artefact proprement
    logger.info("Création d'un nouveau run dans MLflow pour héberger le modèle...")
    with mlflow.start_run(run_name="model_registration") as run:
        # On logue quelques métriques pour retrouver le run facilement
        mlflow.log_metric("val_opt_f1", 0.6309)

        # On logue le modèle dans ce run
        mlflow.xgboost.log_model(model, "model")
        run_id = run.info.run_id
        logger.info(f"Modèle logué dans le run : {run_id}")

    # 2. Enregistrement dans le Model Registry
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Enregistrement au registre sous le nom : {MODEL_NAME}")
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    # 3. Transition vers Staging
    logger.info(f"Transition du modèle version {mv.version} vers Staging")
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    logger.info(f" Modèle enregistré : {MODEL_NAME} v{mv.version} (Staging)")


if __name__ == "__main__":
    register_model()
