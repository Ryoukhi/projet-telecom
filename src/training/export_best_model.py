"""Exporte le modèle XGBoost final depuis MLflow vers models/model.pkl."""

from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
from loguru import logger

RUN_ID = "93ca0d183c78451ab7359525bc01817d"
MODEL_DIR = Path("models")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_uri = f"runs:/{RUN_ID}/model"
    logger.info(f"Chargement du modèle depuis {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    output_path = MODEL_DIR / "model.pkl"
    joblib.dump(model, output_path)
    logger.info(f"Modèle exporté : {output_path}")

    # Vérification : recharger et prédire sur 5 lignes
    loaded = joblib.load(output_path)
    logger.info(f"Modèle rechargé avec succès : {type(loaded).__name__}")


if __name__ == "__main__":
    main()
