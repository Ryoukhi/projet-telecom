"""Module d'ingestion des données brutes.

Charge le CSV brut, effectue des vérifications de base et sauvegarde
une version nettoyée dans data/interim/.
"""

from pathlib import Path

import pandas as pd
from loguru import logger

RAW_PATH = Path("data/raw/telecom_credit_scoring_nigeria.csv")
INTERIM_PATH = Path("data/interim/impayes_loaded.parquet")

REQUIRED_COLUMNS = [
    "customer_id",
    "operator",
    "credit_score",
    "credit_rating",
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
    "payment_consistency_score",
    "late_payment_count",
    "account_type",
    "loan_eligibility",
    "max_loan_amount_ngn",
    "default_risk",
]

# Colonnes à exclure (PII ou non pertinentes pour le ML)
DROP_COLUMNS = ["phone_number", "assessment_date"]


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Charge le CSV brut et vérifie le schéma."""
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    df = pd.read_csv(path)
    logger.info(f"Chargé {len(df)} lignes, {len(df.columns)} colonnes")

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    return df


def basic_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage de base et recodage de la cible."""
    n_before = len(df)

    # Doublons sur customer_id
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    logger.info(f"Doublons supprimés : {n_before - len(df)}")

    # Exclusion PII / colonnes inutiles
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Recodage de la cible : medium/high → 1, low → 0
    df["impaye"] = df["default_risk"].isin(["medium", "high"]).astype(int)
    df = df.drop(columns=["default_risk"])

    # Booléen → int
    df["loan_eligibility"] = df["loan_eligibility"].astype(int)

    # Log du déséquilibre
    taux = df["impaye"].mean()
    logger.info(f"Taux d'impayés (medium+high) : {taux:.2%}")
    logger.info(f"Distribution : {df['impaye'].value_counts().to_dict()}")

    return df


def save_interim(df: pd.DataFrame, path: Path = INTERIM_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Sauvegardé {len(df)} lignes dans {path}")


def main() -> None:
    df = load_raw_data()
    df = basic_checks(df)
    save_interim(df)


if __name__ == "__main__":
    main()
