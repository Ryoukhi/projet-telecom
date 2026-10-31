"""Split train/val/test reproductible avec stratification."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

PROCESSED_PATH = Path("data/processed/features.parquet")
SPLIT_DIR = Path("data/processed/splits")
TARGET = "impaye"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def main() -> None:
    df = pd.read_parquet(PROCESSED_PATH)
    logger.info(f"Chargé {len(df)} lignes")

    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[TARGET], random_state=RANDOM_STATE
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split.to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)
        logger.info(f"{name}: {len(split)} lignes, taux impayés {split[TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
