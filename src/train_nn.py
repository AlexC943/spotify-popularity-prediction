"""PyTorch feed-forward MLP for tabular regression.

Training uses MSE loss on standardized targets, Adam with weight decay,
dropout, and early stopping on validation RMSE. We sweep a small grid of
(architecture, dropout, learning rate) on the validation split, keep the
best configuration, and report a single test-set evaluation.
"""
from __future__ import annotations

from . import _plotting  # noqa: F401 — forces Agg backend before pyplot

import json
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .data import prepare
from .metrics import regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA / cuDNN as deterministic as the kernels allow.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _to_tensor(arr) -> torch.Tensor:
    if hasattr(arr, "toarray"):
        arr = arr.toarray()
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


@dataclass
class TrainResult:
    val_rmse: float
    history: dict
    state_dict: dict
    train_seconds: float
    epochs_trained: int


def _train_one(
    X_tr: torch.Tensor, y_tr_std: torch.Tensor,
    X_val: torch.Tensor, y_val_raw_np: np.ndarray,
    y_mean: float, y_std: float,
    hidden: tuple[int, ...], dropout: float, lr: float,
    weight_decay: float, batch_size: int, max_epochs: int, patience: int,
    device: torch.device, seed: int = config.SEED,
) -> TrainResult:
    """Train one MLP configuration. Returns the best validation RMSE and the
    state_dict at that point. **Does not look at the test set** — the test
    set is consumed exactly once, in `main()`, on the configuration that
    wins the validation sweep.
    """
    set_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr_std),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    model = MLP(in_dim=X_tr.shape[1], hidden=hidden, dropout=dropout).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    history = {"epoch": [], "train_loss": [], "val_rmse": []}
    best_state, best_val_rmse, bad_epochs = None, float("inf"), 0
    X_val_dev = X_val.to(device)

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = running / n

        model.eval()
        with torch.no_grad():
            v_std = model(X_val_dev).cpu().numpy()
        v_pred = v_std * y_std + y_mean
        val_rmse = float(np.sqrt(np.mean((v_pred - y_val_raw_np) ** 2)))

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    train_seconds = time.perf_counter() - t0
    return TrainResult(
        val_rmse=best_val_rmse,
        history=history,
        state_dict=best_state or {},
        train_seconds=train_seconds,
        epochs_trained=history["epoch"][-1],
    )


def main() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.SEED)

    splits, pre = prepare()
    X_tr = _to_tensor(pre.transform(splits.X_train))
    X_val = _to_tensor(pre.transform(splits.X_val))
    # X_te is held until after the validation sweep finishes; the sweep
    # never sees it, so model selection is purely on validation RMSE.
    X_te = _to_tensor(pre.transform(splits.X_test))
    y_tr_raw = torch.from_numpy(splits.y_train.astype(np.float32))

    y_mean = y_tr_raw.mean().item()
    y_std = y_tr_raw.std().item() or 1.0
    y_tr = (y_tr_raw - y_mean) / y_std

    # --- Validation sweep over a small grid ---
    grid = [
        {"hidden": (256, 128, 64), "dropout": 0.1, "lr": 1e-3},
        {"hidden": (256, 128, 64), "dropout": 0.2, "lr": 1e-3},
        {"hidden": (256, 128, 64), "dropout": 0.2, "lr": 5e-4},
        {"hidden": (512, 256, 128), "dropout": 0.2, "lr": 1e-3},
        {"hidden": (512, 256, 128), "dropout": 0.3, "lr": 1e-3},
    ]
    sweep_rows = []
    best, best_cfg = None, None
    for cfg in grid:
        result = _train_one(
            X_tr=X_tr, y_tr_std=y_tr, X_val=X_val, y_val_raw_np=splits.y_val,
            y_mean=y_mean, y_std=y_std,
            hidden=cfg["hidden"], dropout=cfg["dropout"], lr=cfg["lr"],
            weight_decay=1e-4, batch_size=512, max_epochs=80, patience=8,
            device=device,
        )
        sweep_rows.append({
            "hidden": "x".join(str(h) for h in cfg["hidden"]),
            "dropout": cfg["dropout"],
            "lr": cfg["lr"],
            "val_rmse": result.val_rmse,
            "epochs": result.epochs_trained,
            "train_seconds": result.train_seconds,
        })
        if best is None or result.val_rmse < best.val_rmse:
            best, best_cfg = result, cfg

    pd.DataFrame(sweep_rows).to_csv(config.TABLES_DIR / "mlp_tuning.csv", index=False)

    # --- Test-set evaluation: ONE forward pass on the winner only ---
    final_model = MLP(in_dim=X_tr.shape[1], hidden=best_cfg["hidden"],
                       dropout=best_cfg["dropout"]).to(device)
    final_model.load_state_dict(best.state_dict)
    final_model.eval()
    with torch.no_grad():
        test_pred = final_model(X_te.to(device)).cpu().numpy() * y_std + y_mean
    test_metrics = regression_metrics(splits.y_test, test_pred)

    # --- Save learning curve of the BEST configuration ---
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(best.history["epoch"], best.history["train_loss"],
             color="#3b82f6", label="train MSE (standardized)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train MSE (standardized target)", color="#3b82f6")
    ax1.tick_params(axis="y", labelcolor="#3b82f6")
    ax2 = ax1.twinx()
    ax2.plot(best.history["epoch"], best.history["val_rmse"],
             color="#dc2626", linestyle="--", label="val RMSE")
    ax2.set_ylabel("Validation RMSE (original scale)", color="#dc2626")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    fig.suptitle("MLP learning curve (best validation configuration)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "06_mlp_learning_curve.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    torch.save(best.state_dict, config.MODELS_DIR / "mlp.pt")
    pd.DataFrame(best.history).to_csv(config.TABLES_DIR / "mlp_history.csv", index=False)

    # --- Append to results ---
    row = {
        "model": "MLP_PyTorch",
        "family": "neural_network",
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "r2": test_metrics["r2"],
        "train_seconds": best.train_seconds,
        "params": json.dumps({
            "hidden": list(best_cfg["hidden"]),
            "dropout": best_cfg["dropout"],
            "lr": best_cfg["lr"],
            "weight_decay": 1e-4,
            "batch_size": 512,
            "early_stopped_epoch": best.epochs_trained,
            "best_val_rmse": best.val_rmse,
            "device": str(device),
            "tuned_grid_size": len(grid),
        }),
    }
    if config.RESULTS_CSV.exists():
        df = pd.read_csv(config.RESULTS_CSV)
        df = df[df["model"] != "MLP_PyTorch"]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(config.RESULTS_CSV, index=False)

    print("[nn] best cfg:", best_cfg, "test:", test_metrics)
    return test_metrics


if __name__ == "__main__":
    main()
