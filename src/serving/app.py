"""API FastAPI pour la prédiction des impayés télécom."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.serving.schemas import PredictionInput, PredictionOutput

MODEL_PATH = Path("models/model.pkl")
THRESHOLD = 0.33  # Seuil validé au M2 (stratégie recall)

# Variable globale pour le modèle (chargé au démarrage)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, le libère à l'arrêt."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}. Lance d'abord le M2.")

    logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
    ml_models["model"] = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé avec succès.")
    yield
    ml_models.clear()
    logger.info("Modèle déchargé.")


app = FastAPI(
    title="API Prédiction Impayés Télécom",
    description="Prédit la probabilité d'impayé d'un client à partir de ses features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Liveness/readiness probe."""
    model_loaded = "model" in ml_models
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    """Prédit la probabilité d'impayé pour un client."""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    model = ml_models["model"]

    # 1. Récupérer les données envoyées
    data = payload.model_dump()

    # 2. Filtrer uniquement les 11 features attendues par le modèle
    n_expected = model.n_features_in_

    # Liste des 11 features attendues (basée sur l'entraînement)
    expected_cols = [
        "cat__account_type_postpaid",
        "cat__account_type_prepaid",
        "cat__operator_9mobile",
        "cat__operator_Airtel",
        "cat__operator_Glo",
        "cat__operator_MTN",
        "num__avg_monthly_spend_ngn",
        "num__credit_score",
        "num__log_avg_spend",
        "num__log_max_loan",
        "num__max_loan_amount_ngn",
    ]

    # Si le modèle attend 11 features, on garde ces 11 colonnes.
    if n_expected == 11:
        df = pd.DataFrame([data])[expected_cols]
    else:
        df = pd.DataFrame([data])

    try:
        proba = float(model.predict_proba(df)[0, 1])
    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {e}")

    prediction = int(proba >= THRESHOLD)

    # Niveau de risque (métier)
    if proba < 0.2:
        risk = "low"
    elif proba < 0.5:
        risk = "medium"
    else:
        risk = "high"

    return PredictionOutput(
        probability_impaye=round(proba, 4),
        prediction=prediction,
        threshold=THRESHOLD,
        risk_level=risk,
    )
