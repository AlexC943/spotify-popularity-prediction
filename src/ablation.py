"""Genre-ablation experiment.

The headline pipeline includes `track_genre` as a one-hot categorical
feature. Genre is partly a proxy for the kind of artist that records in
that genre (i.e., a soft form of the identity-leakage problem we tried
to remove by dropping `artists`). To quantify the strict audio-only
ceiling — what we could predict from the waveform alone — we refit
LightGBM with the same hyperparameters but with `track_genre` removed
from the feature set, and compare the test metrics against the headline
LightGBM run.

The output (`outputs/tables/ablation_genre.csv`) gives the with-genre
vs. without-genre comparison; `compare.py` references it in the report.
"""
from __future__ import annotations

from . import _plotting  # noqa: F401

import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import lightgbm as lgb

from . import config
from .data import clean, load_raw, split as split_data
from .metrics import regression_metrics


def _build_no_genre_preprocessor() -> ColumnTransformer:
    cats = [c for c in config.CATEGORICAL_FEATURES if c != "track_genre"]
    numeric_pipe = Pipeline([("scale", StandardScaler())])
    cat_pipe = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, config.NUMERIC_FEATURES),
            ("cat", cat_pipe, cats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def main() -> dict:
    df = clean(load_raw())
    df = df.drop(columns=["track_genre"])  # remove the column entirely

    # Reuse the same split mechanism / seed as the headline pipeline.
    feature_cols = config.NUMERIC_FEATURES + [
        c for c in config.CATEGORICAL_FEATURES if c != "track_genre"
    ]
    X = df[feature_cols].copy()
    y = df[config.TARGET].to_numpy(dtype=np.float32)

    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED
    )
    val_relative = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=config.SEED
    )

    pre = _build_no_genre_preprocessor()
    pre.fit(X_train)
    feat_names = list(pre.get_feature_names_out())
    X_tr = pd.DataFrame(pre.transform(X_train), columns=feat_names)
    X_va = pd.DataFrame(pre.transform(X_val), columns=feat_names)
    X_te = pd.DataFrame(pre.transform(X_test), columns=feat_names)

    t0 = time.perf_counter()
    gbm = lgb.LGBMRegressor(
        n_estimators=5000, learning_rate=0.05, num_leaves=127,
        min_child_samples=20, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=5,
        random_state=config.SEED, n_jobs=-1, verbose=-1,
    )
    gbm.fit(
        X_tr, y_train,
        eval_set=[(X_va, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    train_seconds = time.perf_counter() - t0
    no_genre = regression_metrics(y_test, gbm.predict(X_te))

    # Read the headline (with-genre) numbers from the main results CSV.
    main_df = pd.read_csv(config.RESULTS_CSV)
    with_genre = main_df.loc[main_df["model"] == "LightGBM", ["rmse", "mae", "r2"]].iloc[0]
    out = pd.DataFrame([
        {"variant": "with_genre", "rmse": float(with_genre["rmse"]),
         "mae": float(with_genre["mae"]), "r2": float(with_genre["r2"]),
         "train_seconds": np.nan, "n_features": np.nan},
        {"variant": "no_genre", "rmse": no_genre["rmse"],
         "mae": no_genre["mae"], "r2": no_genre["r2"],
         "train_seconds": train_seconds, "n_features": X_tr.shape[1]},
    ])
    out["delta_r2_vs_with_genre"] = out["r2"] - float(with_genre["r2"])
    out.to_csv(config.TABLES_DIR / "ablation_genre.csv", index=False)

    print("[ablation] LightGBM with vs without track_genre:")
    print(out.to_string(index=False))
    return no_genre


if __name__ == "__main__":
    main()
