"""Simule un jeu de données de production avec du Data Drift."""

import pandas as pd
from loguru import logger
from pathlib import Path

TEST_PATH = Path("data/processed/splits/test.parquet")
DRIFTED_PATH = Path("data/processed/splits/drifted_production.parquet")


def simulate():
    logger.info("Chargement des données de test...")
    df = pd.read_parquet(TEST_PATH)

    # On crée une copie pour simuler la production
    df_drifted = df.copy()

    # --- SIMULATION DU DRIFT ---
    # 1. Inflation : les dépenses moyennes augmentent de 50%
    df_drifted["num__avg_monthly_spend_ngn"] = df_drifted["num__avg_monthly_spend_ngn"] * 1.5

    # 2. Nouveaux clients : l'ancienneté baisse de 80% (beaucoup de nouveaux clients)
    df_drifted["num__tenure_months"] = df_drifted["num__tenure_months"] * 0.2

    # 3. Changement technologique : le score de crédit est plus bas en moyenne
    df_drifted["num__credit_score"] = df_drifted["num__credit_score"] - 50

    logger.info("Simulation de drift terminée (inflation + nouveaux clients).")

    DRIFTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_drifted.to_parquet(DRIFTED_PATH, index=False)
    logger.info(f"Données 'de production' simulées sauvegardées : {DRIFTED_PATH}")


if __name__ == "__main__":
    simulate()
