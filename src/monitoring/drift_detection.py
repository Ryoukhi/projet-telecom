"""Détection de Data Drift avec Evidently AI."""

import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from loguru import logger
from pathlib import Path

REFERENCE_DATA_PATH = Path("data/processed/splits/train.parquet")
CURRENT_DATA_PATH = Path("data/processed/splits/test.parquet")
REPORT_PATH = Path("reports/drift_report.html")


def run_drift_detection():
    """Compare les données d'entraînement et de production pour détecter le drift."""
    logger.info("Chargement des données de référence (train) et actuelles (test)...")
    reference = pd.read_parquet(REFERENCE_DATA_PATH)
    current = pd.read_parquet(CURRENT_DATA_PATH)

    # On retire la cible et l'ID pour ne garder que les features
    cols_to_drop = ["impaye", "customer_id"]
    reference = reference.drop(columns=[c for c in cols_to_drop if c in reference.columns])
    current = current.drop(columns=[c for c in cols_to_drop if c in current.columns])

    logger.info(f"Référence : {reference.shape} | Actuel : {current.shape}")

    # Configuration du rapport Evidently
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    logger.info("Calcul du Data Drift en cours...")
    report.run(reference_data=reference, current_data=current)

    # Sauvegarde du rapport HTML
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    logger.info(f"✅ Rapport de drift sauvegardé : {REPORT_PATH}")

    # Affichage d'un résumé dans la console
    result = report.as_dict()
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]
    logger.info(f"Part des colonnes en dérive (Data Drift) : {drift_share:.2%}")


if __name__ == "__main__":
    # Si un argument est passé (ex: drifted_production.parquet), on l'utilise
    if len(sys.argv) > 1:
        CURRENT_DATA_PATH = Path(sys.argv[1])

    run_drift_detection()
