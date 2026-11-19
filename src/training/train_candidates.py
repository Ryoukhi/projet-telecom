"""Entraînement des modèles candidats : Random Forest et XGBoost.

Chaque modèle est logué dans MLflow avec métriques et artefacts.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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


def train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    """Random Forest avec class_weight balanced."""
    with mlflow.start_run(run_name="random_forest"):
        params = {
            "model": "RandomForest",
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "target_recall": TARGET_RECALL,
        }
        mlflow.log_params(params)
        logger.info("Entraînement Random Forest...")

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]

        # Métriques au seuil 0.5 (référence)
        y_pred_default = (y_proba >= 0.5).astype(int)
        metrics_default = evaluate_model(y_val, y_pred_default, y_proba)
        log_metrics(metrics_default, prefix="val_default_")

        # Seuil pour recall cible
        best_thr = find_threshold_for_recall(y_val, y_proba, target_recall=TARGET_RECALL)
        y_pred_opt = (y_proba >= best_thr).astype(int)
        metrics_opt = evaluate_model(y_val, y_pred_opt, y_proba)
        log_metrics(metrics_opt, prefix="val_opt_")
        mlflow.log_param("optimal_threshold", best_thr)

        log_confusion_matrix(y_val, y_pred_opt)
        mlflow.sklearn.log_model(model, "model")
        logger.info(f"RF metrics (seuil {best_thr:.2f}) : {metrics_opt}")

        # Feature importance
        importances = dict(zip(X_train.columns, model.feature_importances_))
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
        logger.info(f"Top 5 features : {top5}")

        # Évaluation sur le test set
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_default = evaluate_model(y_test, (y_test_proba >= 0.5).astype(int), y_test_proba)
        test_opt = evaluate_model(y_test, (y_test_proba >= best_thr).astype(int), y_test_proba)
        log_metrics(test_default, prefix="test_default_")
        log_metrics(test_opt, prefix="test_opt_")
        logger.info(f"RF test (seuil 0.50) : {test_default}")
        logger.info(f"RF test (seuil {best_thr:.2f}) : {test_opt}")


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test) -> None:
    """XGBoost avec scale_pos_weight pour le déséquilibre."""
    with mlflow.start_run(run_name="xgboost"):
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos
        logger.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

        params = {
            "model": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "random_state": RANDOM_STATE,
            "target_recall": TARGET_RECALL,
        }
        mlflow.log_params({k: v for k, v in params.items() if k not in ("model", "target_recall")})

        model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_proba = model.predict_proba(X_val)[:, 1]

        # Métriques au seuil 0.5 (référence)
        y_pred_default = (y_proba >= 0.5).astype(int)
        metrics_default = evaluate_model(y_val, y_pred_default, y_proba)
        log_metrics(metrics_default, prefix="val_default_")

        # Seuil pour recall cible
        best_thr = find_threshold_for_recall(y_val, y_proba, target_recall=TARGET_RECALL)
        y_pred_opt = (y_proba >= best_thr).astype(int)
        metrics_opt = evaluate_model(y_val, y_pred_opt, y_proba)
        log_metrics(metrics_opt, prefix="val_opt_")
        mlflow.log_param("optimal_threshold", best_thr)

        log_confusion_matrix(y_val, y_pred_opt)
        mlflow.xgboost.log_model(model, "model")
        logger.info(f"XGB metrics (seuil {best_thr:.2f}) : {metrics_opt}")

        # Évaluation sur le test set
        y_test_proba = model.predict_proba(X_test)[:, 1]
        test_default = evaluate_model(y_test, (y_test_proba >= 0.5).astype(int), y_test_proba)
        test_opt = evaluate_model(y_test, (y_test_proba >= best_thr).astype(int), y_test_proba)
        log_metrics(test_default, prefix="test_default_")
        log_metrics(test_opt, prefix="test_opt_")
        logger.info(f"XGB test (seuil 0.50) : {test_default}")
        logger.info(f"XGB test (seuil {best_thr:.2f}) : {test_opt}")


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    train, val, test = load_splits()
    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    train_random_forest(X_train, y_train, X_val, y_val, X_test, y_test)
    train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()
