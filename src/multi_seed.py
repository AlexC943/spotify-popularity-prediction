"""Multi-seed stability check for the three non-trivial models.

After the main pipeline has selected the best hyperparameters on the
validation split (`outputs/tables/rf_tuning.csv`, `outputs/tables/mlp_tuning.csv`,
the LightGBM defaults, and the early-stopping point), we refit those exact
configurations with three random seeds and evaluate each on the held-out
test set. The seeds vary every source of stochasticity that the model
exposes (sklearn `random_state`, numpy/torch RNGs, DataLoader generator,
LightGBM bagging). The point estimate reported in `model_results.csv`
remains the seed=42 run; this module adds the mean and standard deviation
across seeds so the reader can see how stable the headline metrics are.

We do NOT rerun hyperparameter tuning per seed — selection is already
locked in upstream. Only the final fit + test evaluation moves.
"""
from __future__ import annotations

from . import _plotting  # noqa: F401

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

from . import config
from .data import get_feature_names, prepare
from .metrics import regression_metrics
from .train_nn import MLP, _to_tensor, set_seed


SEEDS = [42, 7, 13]


def _read_best_rf_config() -> dict:
    df = pd.read_csv(config.TABLES_DIR / "rf_tuning.csv")
    best = df.sort_values("val_rmse").iloc[0]
    cfg = {"min_samples_leaf": int(best["min_samples_leaf"])}
    mf = best["max_features"]
    cfg["max_features"] = mf if isinstance(mf, str) else float(mf)
    return cfg


def _read_best_mlp_config() -> dict:
    df = pd.read_csv(config.TABLES_DIR / "mlp_tuning.csv")
    best = df.sort_values("val_rmse").iloc[0]
    return {
        "hidden": tuple(int(s) for s in str(best["hidden"]).split("x")),
        "dropout": float(best["dropout"]),
        "lr": float(best["lr"]),
    }


def _run_rf(X_tr, y_tr, X_te, y_te, cfg, seed) -> dict:
    t0 = time.perf_counter()
    rf = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=seed, **cfg
    )
    rf.fit(X_tr, y_tr)
    pred = rf.predict(X_te)
    m = regression_metrics(y_te, pred)
    return {"model": "RandomForest", "seed": seed,
            "train_seconds": time.perf_counter() - t0, **m}


def _run_lgbm(X_tr, y_tr, X_val, y_val, X_te, y_te, seed) -> dict:
    t0 = time.perf_counter()
    gbm = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    gbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    pred = gbm.predict(X_te)
    m = regression_metrics(y_te, pred)
    return {"model": "LightGBM", "seed": seed,
            "train_seconds": time.perf_counter() - t0, **m}


def _run_mlp(X_tr_t, X_val_t, X_te_t, y_tr_raw, y_val_raw, y_te_raw,
             cfg, seed, device) -> dict:
    """Refit the chosen MLP config under a different seed and report test metrics."""
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)
    y_mean = y_tr_raw.mean().item()
    y_std = y_tr_raw.std().item() or 1.0
    y_tr_std = (y_tr_raw - y_mean) / y_std

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_std),
        batch_size=512,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = MLP(in_dim=X_tr_t.shape[1], hidden=cfg["hidden"],
                dropout=cfg["dropout"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    X_val_dev = X_val_t.to(device)
    best_state, best_val_rmse, bad = None, float("inf"), 0
    t0 = time.perf_counter()
    for epoch in range(1, 81):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            v_pred = model(X_val_dev).cpu().numpy() * y_std + y_mean
        val_rmse = float(np.sqrt(np.mean((v_pred - y_val_raw.numpy()) ** 2)))
        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_te_t.to(device)).cpu().numpy() * y_std + y_mean
    m = regression_metrics(y_te_raw.numpy(), test_pred)
    return {"model": "MLP_PyTorch", "seed": seed,
            "train_seconds": time.perf_counter() - t0, **m}


def main() -> pd.DataFrame:
    splits, pre = prepare()
    feat_names = get_feature_names(pre)
    X_tr_df = pd.DataFrame(pre.transform(splits.X_train), columns=feat_names)
    X_val_df = pd.DataFrame(pre.transform(splits.X_val), columns=feat_names)
    X_te_df = pd.DataFrame(pre.transform(splits.X_test), columns=feat_names)

    rf_cfg = _read_best_rf_config()
    mlp_cfg = _read_best_mlp_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t = _to_tensor(X_tr_df.to_numpy())
    X_val_t = _to_tensor(X_val_df.to_numpy())
    X_te_t = _to_tensor(X_te_df.to_numpy())
    y_tr_raw_t = torch.from_numpy(splits.y_train.astype(np.float32))
    y_val_raw_t = torch.from_numpy(splits.y_val.astype(np.float32))
    y_te_raw_t = torch.from_numpy(splits.y_test.astype(np.float32))

    rows: list[dict] = []
    for seed in SEEDS:
        rows.append(_run_rf(X_tr_df, splits.y_train, X_te_df, splits.y_test,
                            rf_cfg, seed))
        rows.append(_run_lgbm(X_tr_df, splits.y_train, X_val_df, splits.y_val,
                              X_te_df, splits.y_test, seed))
        rows.append(_run_mlp(X_tr_t, X_val_t, X_te_t,
                             y_tr_raw_t, y_val_raw_t, y_te_raw_t,
                             mlp_cfg, seed, device))

    df = pd.DataFrame(rows)
    df.to_csv(config.TABLES_DIR / "multi_seed_results.csv", index=False)

    summary = (df.groupby("model")[["rmse", "mae", "r2"]]
                 .agg(["mean", "std"]))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary["seeds"] = json.dumps(SEEDS)
    summary["n_seeds"] = len(SEEDS)
    summary.to_csv(config.TABLES_DIR / "multi_seed_summary.csv", index=False)

    print("[multi-seed] summary:")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    main()
