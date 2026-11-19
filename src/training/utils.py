"""Fonctions utilitaires partagées pour l'entraînement et l'évaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "impaye"
ID_COL = "customer_id"
SPLIT_DIR = Path("data/processed/splits")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les splits train/val/test."""
    train = pd.read_parquet(SPLIT_DIR / "train.parquet")
    val = pd.read_parquet(SPLIT_DIR / "val.parquet")
    test = pd.read_parquet(SPLIT_DIR / "test.parquet")
    logger.info(f"Train={len(train)} | Val={len(val)} | Test={len(test)}")
    return train, val, test


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare features et cible (exclut customer_id et les colonnes de construction de la cible)."""
    drop_cols = [TARGET]
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)

    # Ceinture de sécurité anti-fuite
    leakage_cols = [
        "default_risk",
        "late_payment_count",
        "payment_consistency_score",
        "credit_rating",
        "loan_eligibility",
        "max_loan_amount_ngn",
        "num__late_payment_count",
        "num__payment_consistency_score",
        "num__credit_rating",
        "cat__credit_rating",
        "num__loan_eligibility",
        "cat__loan_eligibility",
        "num__max_loan_amount_ngn",
        "num__log_max_loan",
        "num__consistency_x_late",
        "num__late_per_tenure",
    ]
    drop_cols.extend([c for c in leakage_cols if c in df.columns])
    # OneHotEncoder crée plusieurs colonnes commençant par ces préfixes.
    drop_cols.extend(
        c
        for c in df.columns
        if c.startswith("cat__credit_rating_") or c.startswith("cat__loan_eligibility_")
    )

    X = df.drop(columns=drop_cols)
    y = df[TARGET]
    return X, y


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Calcule toutes les métriques d'évaluation."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "avg_precision": average_precision_score(y_true, y_proba),
    }


def log_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, artifact_path: str = "confusion_matrix"
) -> None:
    """Génère et logue la matrice de confusion dans MLflow."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Non impayé", "Impayé"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Matrice de confusion")
    plt.tight_layout()
    mlflow.log_figure(fig, f"{artifact_path}.png")
    plt.close(fig)


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """Logue un dict de métriques dans MLflow."""
    for name, value in metrics.items():
        mlflow.log_metric(f"{prefix}{name}", value)


def find_threshold_for_recall(
    y_true: pd.Series,
    y_proba: np.ndarray,
    target_recall: float = 0.80,
) -> float:
    """Choisit le seuil atteignant le rappel cible avec la meilleure précision."""
    candidates = []

    thresholds = np.arange(0.05, 0.96, 0.01)
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        candidates.append((threshold, precision, recall, f1))

    achievable = [c for c in candidates if c[2] >= target_recall]
    if achievable:
        # Parmi les seuils qui respectent le rappel métier, privilégier précision puis F1.
        best_thr, _, recall, f1 = max(achievable, key=lambda c: (c[1], c[3], -c[0]))
    else:
        # Aucun seuil n'atteint la cible : retenir le meilleur rappel disponible.
        best_thr, _, recall, f1 = max(candidates, key=lambda c: (c[2], c[3], c[1]))

    precision = precision_score(y_true, (y_proba >= best_thr).astype(int), zero_division=0)
    logger.info(
        f"Seuil pour recall≥{target_recall:.2f} : {best_thr:.2f} | "
        f"precision={precision:.4f} | recall={recall:.4f} | F1={f1:.4f}"
    )
    return float(best_thr)
