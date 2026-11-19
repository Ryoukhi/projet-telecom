"""Schémas Pydantic pour l'API de prédiction."""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Features d'entrée (déjà transformées par le pipeline M1)."""

    num__credit_score: float = Field(..., ge=0, le=1000, description="Score de crédit")
    num__tenure_months: float = Field(..., ge=0, description="Ancienneté en mois")
    num__avg_monthly_spend_ngn: float = Field(..., ge=0, description="Dépense mensuelle moyenne")
    num__recharge_frequency_per_month: float = Field(..., ge=0, description="Fréquence de recharge")
    num__max_loan_amount_ngn: float = Field(..., ge=0, description="Montant max du prêt")
    num__spend_per_tenure: float = Field(..., ge=0, description="Dépense par mois d'ancienneté")
    num__log_max_loan: float = Field(..., description="Log du prêt max")
    num__log_avg_spend: float = Field(..., description="Log de la dépense moyenne")

    # Catégorielles (one-hot encodées)
    cat__operator_9mobile: int = Field(0, ge=0, le=1)
    cat__operator_Airtel: int = Field(0, ge=0, le=1)
    cat__operator_Glo: int = Field(0, ge=0, le=1)
    cat__operator_MTN: int = Field(0, ge=0, le=1)
    cat__account_type_postpaid: int = Field(0, ge=0, le=1)
    cat__account_type_prepaid: int = Field(0, ge=0, le=1)


class PredictionOutput(BaseModel):
    """Résultat de la prédiction."""

    probability_impaye: float = Field(..., ge=0, le=1, description="Probabilité d'impayé")
    prediction: int = Field(..., ge=0, le=1, description="0 = non impayé, 1 = impayé")
    threshold: float = Field(..., description="Seuil de décision utilisé")
    risk_level: str = Field(..., description="low / medium / high")
