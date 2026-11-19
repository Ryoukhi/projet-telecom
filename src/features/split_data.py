"""Split train/val/test reproductible avec stratification."""

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.features.build_features import build_preprocessor, create_business_features

INTERIM_PATH = Path("data/interim/impayes_loaded.parquet")
SPLIT_DIR = Path("data/processed/splits")
TARGET = "impaye"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15


def main() -> None:
    df = create_business_features(pd.read_parquet(INTERIM_PATH))
    logger.info(f"Chargé {len(df)} lignes")

    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=RANDOM_STATE
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_ratio, stratify=train_val[TARGET], random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train)
    x_val = preprocessor.transform(val)
    x_test = preprocessor.transform(test)
    feature_names = preprocessor.get_feature_names_out()

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split, x in (
        ("train", train, x_train),
        ("val", val, x_val),
        ("test", test, x_test),
    ):
        split_out = pd.DataFrame(x, columns=feature_names)
        split_out[TARGET] = split[TARGET].values
        split_out["customer_id"] = split["customer_id"].values
        split_out.to_parquet(SPLIT_DIR / f"{name}.parquet", index=False)
        logger.info(f"{name}: {len(split_out)} lignes, taux impayés {split_out[TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
