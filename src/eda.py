"""Exploratory data analysis. Generates figures and a small summary table."""
from __future__ import annotations

from . import _plotting  # noqa: F401 — forces Agg backend before pyplot

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config
from .data import clean, load_raw

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    raw = load_raw()
    df = clean(raw)

    # --- 1. Summary table ---
    summary = pd.DataFrame({
        "n_rows_raw": [len(raw)],
        "n_rows_clean": [len(df)],
        "n_cols_clean": [df.shape[1]],
        "n_genres": [df["track_genre"].nunique()],
        "missing_total_raw": [int(raw.isna().sum().sum())],
        "popularity_mean": [df[config.TARGET].mean()],
        "popularity_std": [df[config.TARGET].std()],
        "popularity_zero_share": [float((df[config.TARGET] == 0).mean())],
    })
    summary.to_csv(config.TABLES_DIR / "eda_summary.csv", index=False)

    # Per-split balance check (uses the same seeded split as training).
    from .data import split as _split  # local to avoid circular imports
    splits = _split(df)
    split_balance = pd.DataFrame({
        "split": ["train", "val", "test"],
        "n": [len(splits.y_train), len(splits.y_val), len(splits.y_test)],
        "popularity_mean": [splits.y_train.mean(), splits.y_val.mean(), splits.y_test.mean()],
        "popularity_std": [splits.y_train.std(), splits.y_val.std(), splits.y_test.std()],
        "zero_share": [
            float((splits.y_train == 0).mean()),
            float((splits.y_val == 0).mean()),
            float((splits.y_test == 0).mean()),
        ],
    })
    split_balance.to_csv(config.TABLES_DIR / "split_balance.csv", index=False)

    # --- 2. Popularity histogram ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[config.TARGET], bins=50, color="#3b82f6", edgecolor="white")
    ax.set_xlabel("Popularity (0–100)")
    ax.set_ylabel("Track count")
    ax.set_title("Distribution of track popularity")
    _save(fig, "01_popularity_hist.png")

    # --- 3. Correlation heatmap (numeric features + target) ---
    corr_cols = config.NUMERIC_FEATURES + [config.TARGET]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation among numeric features and target")
    _save(fig, "02_correlation_heatmap.png")

    # --- 4. Top/bottom genres by mean popularity ---
    by_genre = (df.groupby("track_genre")[config.TARGET]
                  .agg(["mean", "count"])
                  .sort_values("mean", ascending=False))
    top = pd.concat([by_genre.head(15), by_genre.tail(15)])
    fig, ax = plt.subplots(figsize=(8, 9))
    colors = ["#2563eb"] * 15 + ["#dc2626"] * 15
    ax.barh(top.index[::-1], top["mean"].values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean popularity")
    ax.set_title("Top-15 and bottom-15 genres by mean popularity")
    _save(fig, "03_genre_popularity.png")
    by_genre.to_csv(config.TABLES_DIR / "popularity_by_genre.csv")

    # --- 5. Feature-target relationships (binned) ---
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    for ax, feat in zip(axes.ravel(), config.NUMERIC_FEATURES):
        bins = pd.qcut(df[feat], q=20, duplicates="drop")
        means = df.groupby(bins, observed=True)[config.TARGET].mean()
        centers = [interval.mid for interval in means.index]
        ax.plot(centers, means.values, marker="o", color="#2563eb")
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel("")
    axes[0, 0].set_ylabel("Mean popularity")
    axes[1, 0].set_ylabel("Mean popularity")
    fig.suptitle("Mean popularity vs. binned audio features (20 quantile bins)",
                 fontsize=12)
    _save(fig, "04_feature_target_relationships.png")

    # --- 6. Boxplot of popularity by explicit flag ---
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(data=df, x="explicit", y=config.TARGET, hue="explicit",
                palette=["#94a3b8", "#f97316"], legend=False, ax=ax)
    ax.set_title("Popularity by explicit flag")
    _save(fig, "05_popularity_by_explicit.png")

    print("[EDA] saved figures to", config.FIGURES_DIR)
    print(summary.T.to_string())


if __name__ == "__main__":
    run()
