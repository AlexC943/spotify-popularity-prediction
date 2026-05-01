"""Data loading, cleaning, splitting, and preprocessing.

The preprocessor is fit only on the training split and reused for
validation and test, which prevents information leakage.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.RAW_CSV)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    drop = [c for c in config.LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=drop)
    # A few rows have NaN audio features; drop them (very small fraction).
    df = df.dropna(subset=config.NUMERIC_FEATURES + [config.TARGET])
    # explicit is bool → int for safe one-hot.
    df["explicit"] = df["explicit"].astype(int)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, seed: int = config.SEED) -> SplitData:
    feature_cols = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df[config.TARGET].to_numpy(dtype=np.float32)

    # First carve out the test set, then split remainder into train/val.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=seed
    )
    val_relative = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=seed
    )
    return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)


def build_preprocessor() -> ColumnTransformer:
    """Standard-scale numerics, one-hot categoricals (sparse output off for compat)."""
    numeric_pipe = Pipeline([("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", cat_pipe, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def prepare() -> tuple[SplitData, ColumnTransformer]:
    df = clean(load_raw())
    splits = split(df)
    pre = build_preprocessor()
    pre.fit(splits.X_train)
    return splits, pre
