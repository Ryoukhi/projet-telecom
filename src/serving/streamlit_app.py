"""Interface Streamlit autonome pour la prédiction du risque d'impayé."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.pkl"
THRESHOLD = 0.33
FEATURE_COLUMNS = [
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


@st.cache_resource
def load_model():
    """Charge le modèle une seule fois par instance Streamlit."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Prédiction Impayés Télécom", page_icon="📱")
st.title("Prédiction d'impayés télécom")
st.write("Renseignez le profil du client pour évaluer son risque d'impayé.")

try:
    model = load_model()
except Exception as exc:
    st.error(
        f"Impossible de charger le modèle. Vérifiez que `models/model.pkl` est présent. Détail : {exc}"
    )
    st.stop()

with st.form("prediction_form"):
    st.subheader("Caractéristiques du client")
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Score de crédit", min_value=0, max_value=1000, value=600)
        tenure_months = st.number_input("Ancienneté (mois)", min_value=0, value=12)
        avg_spend = st.number_input("Dépense mensuelle (NGN)", min_value=0, value=20000)
        recharge_freq = st.number_input("Recharges / mois", min_value=0, value=5)

    with col2:
        max_loan = st.number_input("Montant max du prêt (NGN)", min_value=0, value=50000)
        operator = st.selectbox("Opérateur", ["MTN", "Airtel", "Glo", "9mobile"])
        account_type = st.selectbox("Type de compte", ["Prepaid", "Postpaid"])

    submitted = st.form_submit_button("Prédire le risque")

if submitted:
    import numpy as np

    features = {
        "cat__account_type_postpaid": int(account_type == "Postpaid"),
        "cat__account_type_prepaid": int(account_type == "Prepaid"),
        "cat__operator_9mobile": int(operator == "9mobile"),
        "cat__operator_Airtel": int(operator == "Airtel"),
        "cat__operator_Glo": int(operator == "Glo"),
        "cat__operator_MTN": int(operator == "MTN"),
        "num__avg_monthly_spend_ngn": float(avg_spend),
        "num__credit_score": float(credit_score),
        "num__log_avg_spend": float(np.log1p(avg_spend)),
        "num__log_max_loan": float(np.log1p(max_loan)),
        "num__max_loan_amount_ngn": float(max_loan),
    }

    # Ces champs sont demandés dans le formulaire pour décrire le client,
    # même s'ils ne figurent pas parmi les 11 variables retenues du modèle.
    _ = tenure_months, recharge_freq
    try:
        probability = float(
            model.predict_proba(pd.DataFrame([features], columns=FEATURE_COLUMNS))[0, 1]
        )
        prediction = int(probability >= THRESHOLD)
        risk = "low" if probability < 0.2 else "medium" if probability < 0.5 else "high"

        st.divider()
        st.subheader("Résultat de la prédiction")
        risk_label = {"low": "Faible", "medium": "Moyen", "high": "Élevé"}[risk]
        col_probability, col_risk = st.columns(2)
        col_probability.metric("Probabilité d'impayé", f"{probability:.2%}")
        col_risk.metric("Niveau de risque", risk_label)
        st.caption("Niveau de risque : faible (< 20 %), moyen (20 à 50 %), élevé (≥ 50 %).")
        if prediction:
            st.warning(
                f"⚠️ Le seuil de décision de {THRESHOLD:.0%} est atteint. "
                f"Le modèle recommande une relance. Le niveau de risque estimé reste « {risk_label.lower()} » selon les catégories ci-dessus."
            )
        else:
            st.success(
                f"✅ La probabilité est inférieure au seuil de décision de {THRESHOLD:.0%}. "
                "Aucune relance n'est recommandée par cette règle."
            )
    except Exception as exc:
        st.error(f"La prédiction a échoué : {exc}")
