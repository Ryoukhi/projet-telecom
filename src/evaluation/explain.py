"""Analyse SHAP pour l'interprétabilité du modèle XGBoost final.

Charge le modèle depuis MLflow (run_id), calcule les valeurs SHAP
sur le jeu de validation, et sauvegarde les graphiques dans reports/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from loguru import logger

from src.training.utils import load_splits, split_xy

EXPERIMENT_NAME = "impayes-telecom"
OUTPUT_DIR = Path("reports/figures")
N_SAMPLES = 2000  # échantillon pour SHAP (plus rapide sur 73k lignes)


def load_model_from_mlflow(run_id: str):
    """Charge le modèle XGBoost depuis MLflow."""
    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Chargement du modèle depuis {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)
    return model


def main(run_id: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger les données de validation
    _, val, _ = load_splits()
    X_val, y_val = split_xy(val)
    logger.info(f"Validation : {X_val.shape}, taux impayés={y_val.mean():.3f}")

    # 2. Échantillonner pour SHAP (plus rapide)
    if len(X_val) > N_SAMPLES:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_val), size=N_SAMPLES, replace=False)
        X_sample = X_val.iloc[idx].reset_index(drop=True)
        y_sample = y_val.iloc[idx].reset_index(drop=True)
    else:
        X_sample, y_sample = X_val, y_val
    logger.info(f"Échantillon SHAP : {X_sample.shape}")

    # 3. Charger le modèle
    model = load_model_from_mlflow(run_id)

    # 4. Calculer les valeurs SHAP
    logger.info("Calcul des valeurs SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Pour XGBoost binaire, shap_values est un array 2D (n_samples, n_features)
    # Pour certains cas, c'est une liste [classe_0, classe_1] → on prend la classe 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    logger.info(f"SHAP values shape : {shap_values.shape}")

    # 5. Summary plot (beeswarm) — impact global + direction
    logger.info("Génération du summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    summary_path = OUTPUT_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sauvegardé : {summary_path}")

    # 6. Bar plot — importance globale moyenne
    logger.info("Génération du bar plot (importance globale)...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    bar_path = OUTPUT_DIR / "shap_importance.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Sauvegardé : {bar_path}")

    # 7. Waterfall plot pour un client impayé (exemple individuel)
    logger.info("Génération du waterfall plot (exemple individuel)...")
    # Trouver un vrai impayé dans l'échantillon
    impaye_indices = np.where(y_sample.values == 1)[0]
    if len(impaye_indices) > 0:
        idx_example = impaye_indices[0]
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx_example],
                base_values=explainer.expected_value,
                data=X_sample.iloc[idx_example].values,
                feature_names=X_sample.columns.tolist(),
            ),
            show=False,
            max_display=10,
        )
        plt.tight_layout()
        waterfall_path = OUTPUT_DIR / "shap_waterfall_example.png"
        plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Sauvegardé : {waterfall_path}")

    # 8. Loguer les figures dans MLflow (artefacts du run)
    logger.info("Log des figures dans MLflow...")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(summary_path), artifact_path="shap")
        mlflow.log_artifact(str(bar_path), artifact_path="shap")
        if len(impaye_indices) > 0:
            mlflow.log_artifact(str(waterfall_path), artifact_path="shap")

    # 9. Top 10 features par importance SHAP moyenne
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": X_sample.columns, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)
    logger.info("Top 10 features (importance SHAP moyenne) :")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    # Sauvegarder le CSV d'importance
    importance_path = OUTPUT_DIR / "shap_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"Sauvegardé : {importance_path}")

    logger.info("✅ Analyse SHAP terminée.")


if __name__ == "__main__":
    # Run ID du modèle XGBoost final
    RUN_ID = "93ca0d183c78451ab7359525bc01817d"
    main(RUN_ID)
