"""Train all non-NN models: mean baseline, Ridge, Lasso, Random Forest, LightGBM.

Hyperparameters that need tuning are searched on the validation split.
The test split is consumed only for the final metrics that are written
to outputs/tables/model_results.csv.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge

import lightgbm as lgb

from . import config
from .data import get_feature_names, prepare
from .metrics import regression_metrics


def _record(rows: list[dict], name: str, family: str, y_test, y_pred,
            train_time: float, params: dict | None = None) -> None:
    m = regression_metrics(y_test, y_pred)
    rows.append({
        "model": name,
        "family": family,
        "rmse": m["rmse"],
        "mae": m["mae"],
        "r2": m["r2"],
        "train_seconds": train_time,
        "params": json.dumps(params or {}),
    })


def _tune_alpha(model_cls, alphas, X_tr, y_tr, X_val, y_val):
    best_alpha, best_rmse, best_model = None, np.inf, None
    for a in alphas:
        m = model_cls(alpha=a, random_state=config.SEED, max_iter=20000)
        m.fit(X_tr, y_tr)
        rmse = regression_metrics(y_val, m.predict(X_val))["rmse"]
        if rmse < best_rmse:
            best_alpha, best_rmse, best_model = a, rmse, m
    return best_model, best_alpha, best_rmse


def _tune_random_forest(X_tr, y_tr, X_val, y_val) -> tuple[RandomForestRegressor, dict]:
    """Small validation grid: a fast 100-tree search picks the configuration,
    then we refit at full size (300 trees) on the same training data."""
    grid = [
        {"min_samples_leaf": 1, "max_features": "sqrt"},
        {"min_samples_leaf": 1, "max_features": 1.0},
        {"min_samples_leaf": 2, "max_features": "sqrt"},
        {"min_samples_leaf": 2, "max_features": 1.0},
        {"min_samples_leaf": 5, "max_features": "sqrt"},
        {"min_samples_leaf": 5, "max_features": 1.0},
    ]
    best_cfg, best_rmse = None, np.inf
    sweep_results = []
    for cfg in grid:
        m = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=config.SEED, **cfg
        )
        m.fit(X_tr, y_tr)
        rmse = regression_metrics(y_val, m.predict(X_val))["rmse"]
        sweep_results.append({**cfg, "val_rmse": rmse})
        if rmse < best_rmse:
            best_rmse, best_cfg = rmse, cfg
    pd.DataFrame(sweep_results).to_csv(
        config.TABLES_DIR / "rf_tuning.csv", index=False
    )
    full = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=config.SEED, **best_cfg
    )
    full.fit(X_tr, y_tr)
    return full, {"n_estimators": 300, **best_cfg, "tuning_val_rmse": best_rmse}


def main() -> pd.DataFrame:
    splits, pre = prepare()
    feat_names = get_feature_names(pre)

    # Wrap the preprocessor output in DataFrames with explicit feature
    # names. LightGBM's sklearn wrapper auto-generates "Column_N" names
    # for unnamed input, which then triggers a feature-name mismatch
    # warning at predict time; explicit names make fit and predict
    # consistent.
    X_tr = pd.DataFrame(pre.transform(splits.X_train), columns=feat_names)
    X_val = pd.DataFrame(pre.transform(splits.X_val), columns=feat_names)
    X_te = pd.DataFrame(pre.transform(splits.X_test), columns=feat_names)
    y_tr, y_val, y_te = splits.y_train, splits.y_val, splits.y_test

    rows: list[dict] = []

    # 1) Mean baseline
    t0 = time.perf_counter()
    mean_y = float(np.mean(y_tr))
    y_pred = np.full_like(y_te, mean_y, dtype=np.float64)
    _record(rows, "MeanBaseline", "baseline", y_te, y_pred,
            time.perf_counter() - t0, {"mean": mean_y})

    # 2) Ridge
    t0 = time.perf_counter()
    ridge_alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
    ridge, alpha_r, _ = _tune_alpha(Ridge, ridge_alphas, X_tr, y_tr, X_val, y_val)
    _record(rows, "Ridge", "linear", y_te, ridge.predict(X_te),
            time.perf_counter() - t0, {"alpha": alpha_r})

    # 3) Lasso
    t0 = time.perf_counter()
    lasso_alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    lasso, alpha_l, _ = _tune_alpha(Lasso, lasso_alphas, X_tr, y_tr, X_val, y_val)
    _record(rows, "Lasso", "linear", y_te, lasso.predict(X_te),
            time.perf_counter() - t0, {"alpha": alpha_l})

    # 4) Random Forest (validation-tuned on a small grid, refit at 300 trees)
    t0 = time.perf_counter()
    rf, rf_params = _tune_random_forest(X_tr, y_tr, X_val, y_val)
    _record(rows, "RandomForest", "tree_ensemble", y_te, rf.predict(X_te),
            time.perf_counter() - t0, rf_params)

    # 5) LightGBM with early stopping on validation
    t0 = time.perf_counter()
    gbm = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=config.SEED,
        n_jobs=-1,
        verbose=-1,
    )
    gbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    gbm_params = {
        "learning_rate": 0.05, "num_leaves": 127, "n_estimators_cap": 5000,
        "best_iteration": int(gbm.best_iteration_ or gbm.n_estimators_),
    }
    _record(rows, "LightGBM", "gradient_boosting", y_te, gbm.predict(X_te),
            time.perf_counter() - t0, gbm_params)

    # --- Save feature importances from RF and LightGBM ---
    fi = pd.DataFrame({
        "feature": feat_names,
        "rf_importance": rf.feature_importances_,
        "lgbm_importance": gbm.feature_importances_,
    })
    fi.to_csv(config.TABLES_DIR / "feature_importances.csv", index=False)

    # --- Persist results ---
    df_res = pd.DataFrame(rows)
    if config.RESULTS_CSV.exists():
        prev = pd.read_csv(config.RESULTS_CSV)
        prev = prev[~prev["model"].isin(df_res["model"])]
        df_res = pd.concat([prev, df_res], ignore_index=True)
    df_res.to_csv(config.RESULTS_CSV, index=False)

    joblib.dump(pre, config.MODELS_DIR / "preprocessor.joblib")
    joblib.dump(ridge, config.MODELS_DIR / "ridge.joblib")
    joblib.dump(lasso, config.MODELS_DIR / "lasso.joblib")
    joblib.dump(rf, config.MODELS_DIR / "rf.joblib")
    gbm.booster_.save_model(str(config.MODELS_DIR / "lgbm.txt"))

    print("[classical] results:")
    print(df_res.sort_values("rmse")[["model", "rmse", "mae", "r2"]].to_string(index=False))
    return df_res


if __name__ == "__main__":
    main()
