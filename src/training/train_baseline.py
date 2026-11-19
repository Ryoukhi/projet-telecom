"""Entraînement du baseline : régression logistique.

Interprétable, sert de référence pour évaluer les modèles plus complexes.
"""

import mlflow
import mlflow.sklearn
from loguru import logger
from sklearn.linear_model import LogisticRegression

from src.training.utils import (
    evaluate_model,
    find_threshold_for_recall,
    load_splits,
    log_confusion_matrix,
    log_metrics,
    split_xy,
)

EXPERIMENT_NAME = "impayes-telecom"
RANDOM_STATE = 42
TARGET_RECALL = 0.80


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    train, val, test = load_splits()
    x_train, y_train = split_xy(train)
    x_val, y_val = split_xy(val)
    x_test, y_test = split_xy(test)

    with mlflow.start_run(run_name="baseline_logistic") as run:
        # Paramètres
        params = {
            "model": "LogisticRegression",
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
            "target_recall": TARGET_RECALL,
        }
        mlflow.log_params(params)
        logger.info(f"Run ID : {run.info.run_id}")

        # Entraînement
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        model.fit(x_train, y_train)

        # Prédictions sur validation
        y_proba = model.predict_proba(x_val)[:, 1]

        # --- Métriques au seuil par défaut 0.5 (référence) ---
        y_pred_default = (y_proba >= 0.5).astype(int)
        metrics_default = evaluate_model(y_val, y_pred_default, y_proba)
        log_metrics(metrics_default, prefix="val_default_")
        logger.info(f"Métriques (seuil 0.5) : {metrics_default}")

        # --- Seuil pour recall cible (stratégie métier) ---
        best_thr = find_threshold_for_recall(y_val, y_proba, target_recall=TARGET_RECALL)
        y_pred_opt = (y_proba >= best_thr).astype(int)
        metrics_opt = evaluate_model(y_val, y_pred_opt, y_proba)
        log_metrics(metrics_opt, prefix="val_opt_")
        mlflow.log_param("optimal_threshold", best_thr)
        logger.info(f"Métriques (seuil {best_thr:.2f}) : {metrics_opt}")

        # --- Évaluation sur le test set ---
        y_test_proba = model.predict_proba(x_test)[:, 1]
        test_default = evaluate_model(y_test, (y_test_proba >= 0.5).astype(int), y_test_proba)
        test_opt = evaluate_model(y_test, (y_test_proba >= best_thr).astype(int), y_test_proba)
        log_metrics(test_default, prefix="test_default_")
        log_metrics(test_opt, prefix="test_opt_")
        logger.info(f"Test (seuil 0.50) : {test_default}")
        logger.info(f"Test (seuil {best_thr:.2f}) : {test_opt}")

        # Artefacts
        log_confusion_matrix(y_val, y_pred_opt)
        mlflow.sklearn.log_model(model, "model")

        # Coefficients (interprétabilité)
        coefs = dict(zip(x_train.columns, model.coef_[0]))
        logger.info("Top 5 coefficients positifs (risque↑) :")
        for name, coef in sorted(coefs.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"  {name}: {coef:+.3f}")


if __name__ == "__main__":
    main()
