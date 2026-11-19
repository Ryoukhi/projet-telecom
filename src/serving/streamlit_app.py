"""Application Streamlit de prédiction à partir d'un client sélectionné."""

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
LOCAL_DATA_PATHS = [
    PROJECT_ROOT / "data" / "serving" / "clients.csv",
    PROJECT_ROOT / "data" / "raw" / "telecom_credit_scoring_nigeria.csv",
]
THRESHOLD = 0.33

CLIENT_COLUMNS = [
    "customer_id",
    "operator",
    "credit_score",
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "max_loan_amount_ngn",
    "account_type",
]
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


@st.cache_data(show_spinner="Lecture du fichier clients…")
def load_clients(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Lit un fichier clients CSV ou Excel et vérifie les colonnes nécessaires."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".csv":
        clients = pd.read_csv(BytesIO(file_bytes))
    elif suffix in {".xlsx", ".xls"}:
        clients = pd.read_excel(BytesIO(file_bytes))
    else:
        raise ValueError(
            "Format non pris en charge. Utilisez un fichier CSV ou Excel (.xlsx, .xls)."
        )

    missing = sorted(set(CLIENT_COLUMNS) - set(clients.columns))
    if missing:
        raise ValueError("Colonnes requises absentes : " + ", ".join(missing))

    clients = clients.drop_duplicates(subset=["customer_id"], keep="first").copy()
    clients["customer_id"] = clients["customer_id"].astype("string").str.strip()
    clients = clients[clients["customer_id"].notna() & clients["customer_id"].ne("")]
    return clients.reset_index(drop=True)


def make_features(client: pd.Series) -> pd.DataFrame:
    """Construit les 11 variables attendues par le modèle servi."""
    operator = str(client["operator"]).strip().lower()
    account_type = str(client["account_type"]).strip().lower()
    if operator not in {"mtn", "airtel", "glo", "9mobile"}:
        raise ValueError(f"Opérateur non reconnu pour ce client : {client['operator']}")
    if account_type not in {"prepaid", "postpaid"}:
        raise ValueError(f"Type de compte non reconnu pour ce client : {client['account_type']}")

    try:
        credit_score = float(pd.to_numeric(client["credit_score"], errors="raise"))
        avg_spend = float(pd.to_numeric(client["avg_monthly_spend_ngn"], errors="raise"))
        max_loan = float(pd.to_numeric(client["max_loan_amount_ngn"], errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Score de crédit, dépense ou montant de prêt manquant/invalide.") from exc

    if not all(np.isfinite(value) and value >= 0 for value in (credit_score, avg_spend, max_loan)):
        raise ValueError(
            "Le score et les montants du client doivent être des nombres positifs valides."
        )

    features = {
        "cat__account_type_postpaid": int(account_type == "postpaid"),
        "cat__account_type_prepaid": int(account_type == "prepaid"),
        "cat__operator_9mobile": int(operator == "9mobile"),
        "cat__operator_Airtel": int(operator == "airtel"),
        "cat__operator_Glo": int(operator == "glo"),
        "cat__operator_MTN": int(operator == "mtn"),
        "num__avg_monthly_spend_ngn": avg_spend,
        "num__credit_score": credit_score,
        "num__log_avg_spend": float(np.log1p(avg_spend)),
        "num__log_max_loan": float(np.log1p(max_loan)),
        "num__max_loan_amount_ngn": max_loan,
    }
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


def display_number(value, suffix="") -> str:
    """Formate une valeur numérique du profil sans faire échouer l'interface."""
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "Non renseigné"
    return f"{number:,.0f}{suffix}"


st.set_page_config(page_title="Risque d'impayé télécom", page_icon="📱", layout="centered")
st.title("Prédiction du risque d'impayé")
st.write("Chargez la base clients, recherchez un client et consultez sa prédiction.")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Impossible de charger le modèle `models/model.pkl` : {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Base clients (CSV ou Excel)",
    type=["csv", "xlsx", "xls"],
    help="Le fichier doit contenir un identifiant client, son opérateur, son type de compte et ses données financières.",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif next((path for path in LOCAL_DATA_PATHS if path.is_file()), None) is not None:
    local_data_path = next(path for path in LOCAL_DATA_PATHS if path.is_file())
    file_bytes = local_data_path.read_bytes()
    file_name = local_data_path.name
    st.caption("Base locale du projet chargée automatiquement.")
else:
    st.info("Chargez le fichier CSV ou Excel de vos clients pour commencer.")
    st.stop()

try:
    clients = load_clients(file_bytes, file_name)
except Exception as exc:
    st.error(f"La base clients ne peut pas être utilisée : {exc}")
    st.stop()

st.caption(f"{len(clients):,} clients disponibles · fichier : {file_name}")
search = st.text_input("Rechercher un client par identifiant", placeholder="Ex. 51426327")
if search.strip():
    matches = clients[
        clients["customer_id"].str.contains(search.strip(), case=False, regex=False, na=False)
    ].head(100)
else:
    matches = clients.head(50)
    st.caption("Saisissez quelques caractères de l’identifiant pour filtrer les résultats.")

if matches.empty:
    st.warning("Aucun client ne correspond à cette recherche.")
    st.stop()

client_ids = matches["customer_id"].tolist()
selected_id = st.selectbox("Sélectionner le client", client_ids)
client = clients.loc[clients["customer_id"].eq(selected_id)].iloc[0]

st.subheader(f"Profil client · {selected_id}")
profile_1, profile_2, profile_3 = st.columns(3)
profile_1.metric("Opérateur", str(client["operator"]))
profile_2.metric("Type de compte", str(client["account_type"]))
profile_3.metric("Score de crédit", display_number(client["credit_score"]))
profile_4, profile_5, profile_6 = st.columns(3)
profile_4.metric("Ancienneté", display_number(client["tenure_months"], " mois"))
profile_5.metric("Dépense mensuelle", display_number(client["avg_monthly_spend_ngn"], " NGN"))
profile_6.metric("Recharges / mois", display_number(client["recharge_frequency_per_month"]))
st.caption(
    "Montant maximal de prêt : "
    f"{display_number(client['max_loan_amount_ngn'], ' NGN')}. "
    "L'ancienneté et les recharges sont affichées à titre informatif ; le modèle ne les utilise pas dans cette version."
)

try:
    features = make_features(client)
    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= THRESHOLD)
    risk = "Faible" if probability < 0.2 else "Moyen" if probability < 0.5 else "Élevé"

    st.divider()
    st.subheader("Résultat de la prédiction")
    result_probability, result_risk = st.columns(2)
    result_probability.metric("Probabilité estimée d'impayé", f"{probability:.2%}")
    risk_colors = {
        "Faible": ("#166534", "#dcfce7", "#86efac"),
        "Moyen": ("#92400e", "#fef3c7", "#fcd34d"),
        "Élevé": ("#991b1b", "#fee2e2", "#fca5a5"),
    }
    text_color, background_color, border_color = risk_colors[risk]
    result_risk.markdown("**Niveau de risque**")
    result_risk.markdown(
        f'<div style="display:inline-block;padding:0.45rem 0.8rem;border-radius:999px;'
        f"background:{background_color};color:{text_color};border:1px solid {border_color};"
        f'font-weight:700;font-size:1.05rem">● {risk}</div>',
        unsafe_allow_html=True,
    )
    st.caption("Niveau : faible (< 20 %), moyen (20 à 50 %), élevé (≥ 50 %).")
    if prediction:
        st.warning(
            f"Le seuil de décision de {THRESHOLD:.0%} est atteint. Le modèle recommande une relance. "
            f"Le niveau de risque estimé reste « {risk.lower()} » selon les catégories ci-dessus."
        )
    else:
        st.success(
            f"La probabilité est sous le seuil de décision de {THRESHOLD:.0%}. "
            "Aucune relance n'est recommandée par cette règle."
        )
except Exception as exc:
    st.error(f"La prédiction n'a pas pu être calculée : {exc}")

st.caption(
    "La prédiction est une estimation issue du modèle et doit aider à prioriser le suivi, "
    "pas remplacer une vérification du dossier client."
)
