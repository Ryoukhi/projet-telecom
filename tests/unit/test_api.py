"""Tests unitaires de l'API FastAPI."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    """TestClient avec lifespan (charge le modèle)."""
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_low_risk(client):
    """Un client avec un bon profil ne devrait pas être impayé."""
    payload = {
        "num__credit_score": 800,
        "num__tenure_months": 48,
        "num__avg_monthly_spend_ngn": 50000,
        "num__recharge_frequency_per_month": 15,
        "num__max_loan_amount_ngn": 100000,
        "num__spend_per_tenure": 1000,
        "num__log_max_loan": 11.5,
        "num__log_avg_spend": 10.8,
        "cat__operator_MTN": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["probability_impaye"] <= 1
    assert data["prediction"] in [0, 1]
    assert data["risk_level"] in ["low", "medium", "high"]


def test_predict_invalid_input(client):
    """Un payload invalide doit renvoyer 422."""
    response = client.post("/predict", json={"num__credit_score": "pas_un_chiffre"})
    assert response.status_code == 422
