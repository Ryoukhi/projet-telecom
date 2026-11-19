"""Pipeline de feature engineering adapté au dataset télécom Nigeria."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INTERIM_PATH = Path("data/interim/impayes_loaded.parquet")
PROCESSED_PATH = Path("data/processed/features.parquet")

# Features brutes
NUMERIC_COLS = [
    "credit_score",
    "tenure_months",
    "avg_monthly_spend_ngn",
    "recharge_frequency_per_month",
]
CATEGORICAL_COLS = ["operator", "account_type"]

TARGET = "impaye"
ID_COL = "customer_id"


def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des features métier dérivées."""
    df = df.copy()

    # Dépense mensuelle par mois d'ancienneté
    df["spend_per_tenure"] = df["avg_monthly_spend_ngn"] / (df["tenure_months"] + 1)

    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_COLS),
        ]
    )


def main() -> None:
    df = pd.read_parquet(INTERIM_PATH)
    logger.info(f"Chargé {len(df)} lignes")

    df = create_business_features(df)

    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df)
    feature_names = preprocessor.get_feature_names_out()
    logger.info(f"Features transformées : {X.shape}")

    df_out = pd.DataFrame(X, columns=feature_names)
    df_out[TARGET] = df[TARGET].values
    df_out[ID_COL] = df[ID_COL].values

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(PROCESSED_PATH, index=False)
    logger.info(f"Sauvegardé dans {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
