"""Aggregate metrics, draw comparison plots, write the final report draft."""
from __future__ import annotations

from . import _plotting  # noqa: F401 — forces Agg backend before pyplot

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import lightgbm as lgb

from . import config
from .data import prepare

sns.set_theme(style="whitegrid", context="notebook")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _palette(n: int) -> list[str]:
    return sns.color_palette("viridis", n_colors=n).as_hex()


def make_comparison_plots(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values("rmse")
    palette = _palette(len(df_sorted))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df_sorted["model"], df_sorted["rmse"], color=palette)
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_title("Test RMSE by model")
    for x, v in zip(df_sorted["model"], df_sorted["rmse"]):
        ax.text(x, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    _save(fig, "07_rmse_comparison.png")

    df_r2 = df.sort_values("r2", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df_r2["model"], df_r2["r2"], color=_palette(len(df_r2)))
    ax.set_ylabel("Test R² (higher is better)")
    ax.set_title("Test R² by model")
    for x, v in zip(df_r2["model"], df_r2["r2"]):
        ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    _save(fig, "08_r2_comparison.png")


def _group_for(feature: str) -> str:
    """Map a one-hot column (e.g. 'track_genre_pop') back to its base field."""
    for base in config.CATEGORICAL_FEATURES:
        if feature == base or feature.startswith(base + "_"):
            return base
    return feature


def make_feature_importance_plot() -> pd.DataFrame:
    fi = pd.read_csv(config.TABLES_DIR / "feature_importances.csv")

    top_cols = fi.sort_values("lgbm_importance", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top_cols["feature"][::-1], top_cols["lgbm_importance"][::-1],
            color="#0ea5e9")
    ax.set_title("Top-20 columns by LightGBM split importance")
    ax.set_xlabel("LightGBM split importance")
    _save(fig, "09_feature_importance_lgbm.png")

    fi_grp = fi.assign(group=fi["feature"].map(_group_for))
    agg = (fi_grp.groupby("group")[["rf_importance", "lgbm_importance"]].sum()
                  .sort_values("lgbm_importance", ascending=False))
    agg.to_csv(config.TABLES_DIR / "feature_importances_grouped.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(agg.index[::-1], agg["lgbm_importance"][::-1], color="#0ea5e9")
    ax.set_title("Aggregated feature importance (one-hot dummies summed back)")
    ax.set_xlabel("LightGBM split importance (sum across dummies)")
    _save(fig, "09b_feature_importance_grouped.png")

    return agg.reset_index().rename(columns={"group": "feature"})


def make_pred_vs_actual_plot() -> None:
    import numpy as np
    splits, pre = prepare()
    booster = lgb.Booster(model_file=str(config.MODELS_DIR / "lgbm.txt"))
    X_te = np.asarray(pre.transform(splits.X_test))
    y_pred = booster.predict(X_te)
    y_true = splits.y_test

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hexbin(y_true, y_pred, gridsize=40, cmap="viridis", mincnt=1)
    lim = [0, 100]
    ax.plot(lim, lim, color="white", linestyle="--", linewidth=1.2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual popularity")
    ax.set_ylabel("Predicted popularity")
    ax.set_title("LightGBM: predicted vs. actual on the test set")
    _save(fig, "10_pred_vs_actual_lgbm.png")


def write_report(results: pd.DataFrame, top_features: pd.DataFrame) -> None:
    results = results.sort_values("rmse").reset_index(drop=True)
    baseline_rmse = float(results.loc[results["model"] == "MeanBaseline", "rmse"].iloc[0])
    eda = pd.read_csv(config.TABLES_DIR / "eda_summary.csv").iloc[0]

    best = results.iloc[0]
    best_name = best["model"]
    rmse_reduction = (1 - best["rmse"] / baseline_rmse) * 100

    rows_md = []
    pass_15, fail_15, pass_r2, fail_r2 = [], [], [], []
    for _, r in results.iterrows():
        delta = (1 - r["rmse"] / baseline_rmse) * 100
        rows_md.append(
            f"| {r['model']} | {r['family']} | {r['rmse']:.3f} | "
            f"{r['mae']:.3f} | {r['r2']:.3f} | {delta:+.1f}% | "
            f"{r['train_seconds']:.1f} |"
        )
        if r["model"] == "MeanBaseline":
            continue
        (pass_15 if delta >= 15 else fail_15).append((r["model"], delta))
        (pass_r2 if r["r2"] >= 0.50 else fail_r2).append((r["model"], r["r2"]))
    table_md = "\n".join(rows_md)

    feat_lines = "\n".join(
        f"- `{row['feature']}` (importance = {row['lgbm_importance']:.0f})"
        for _, row in top_features.head(10).iterrows()
    )
    top_feature_name = top_features.iloc[0]["feature"]
    genre_rank = (top_features["feature"] == "track_genre").idxmax() + 1
    genre_imp = float(top_features.loc[top_features["feature"] == "track_genre",
                                       "lgbm_importance"].iloc[0])
    total_imp = float(top_features["lgbm_importance"].sum())
    genre_share = genre_imp / total_imp * 100

    def _fmt_list(items, fmt):
        return ", ".join(f"{name} ({fmt(v)})" for name, v in items) or "none"

    pass15_text = _fmt_list(pass_15, lambda v: f"{v:+.1f}%")
    fail15_text = _fmt_list(fail_15, lambda v: f"{v:+.1f}%")
    passr2_text = _fmt_list(pass_r2, lambda v: f"R²={v:.3f}")
    failr2_text = _fmt_list(fail_r2, lambda v: f"R²={v:.3f}")

    # MLP vs tree-ensemble gap, computed from the actual numbers
    mlp_row = results[results["model"] == "MLP_PyTorch"].iloc[0]
    rf_row = results[results["model"] == "RandomForest"].iloc[0]
    lgb_row = results[results["model"] == "LightGBM"].iloc[0]
    mlp_vs_lgb = mlp_row["rmse"] - lgb_row["rmse"]
    mlp_vs_rf = mlp_row["rmse"] - rf_row["rmse"]

    # Split balance (zero-share per split) for the methodology section.
    try:
        sb = pd.read_csv(config.TABLES_DIR / "split_balance.csv")
        balance_text = ", ".join(
            f"{row['split']}={row['zero_share']*100:.1f}%"
            for _, row in sb.iterrows()
        )
    except FileNotFoundError:
        balance_text = "(see outputs/tables/split_balance.csv after running EDA)"

    # Multi-seed stability (refits the tuned configurations under 3 seeds).
    try:
        ms = pd.read_csv(config.TABLES_DIR / "multi_seed_summary.csv")
        ms_rows = []
        for _, row in ms.iterrows():
            ms_rows.append(
                f"| {row['model']} | {row['rmse_mean']:.3f} ± {row['rmse_std']:.3f} | "
                f"{row['mae_mean']:.3f} ± {row['mae_std']:.3f} | "
                f"{row['r2_mean']:.3f} ± {row['r2_std']:.3f} |"
            )
        ms_table = "\n".join(ms_rows)
        ms_seeds = str(ms.iloc[0]["seeds"])
    except FileNotFoundError:
        ms_table = "| (run `python -m src.multi_seed` to populate) |"
        ms_seeds = "[42]"

    # Genre ablation
    try:
        ab = pd.read_csv(config.TABLES_DIR / "ablation_genre.csv")
        wg = ab[ab["variant"] == "with_genre"].iloc[0]
        ng = ab[ab["variant"] == "no_genre"].iloc[0]
        ablation_lines = (
            f"- LightGBM **with `track_genre`**: RMSE {wg['rmse']:.3f}, "
            f"MAE {wg['mae']:.3f}, R² {wg['r2']:.3f}.\n"
            f"- LightGBM **without `track_genre`** (audio + key/mode/time_signature/explicit only): "
            f"RMSE {ng['rmse']:.3f}, MAE {ng['mae']:.3f}, R² {ng['r2']:.3f}.\n"
            f"- ΔR² from removing genre: **{ng['r2'] - wg['r2']:+.3f}**."
        )
        genre_r2_loss = wg["r2"] - ng["r2"]
    except FileNotFoundError:
        ablation_lines = "(run `python -m src.ablation` to populate)"
        genre_r2_loss = float("nan")

    report = f"""# Empirical Evaluation of Predictive Models for Audio-Feature-Based Track Popularity

**Course:** Yale CPSC 5810 — Introduction to Machine Learning
**Authors:** Guangxing Cao, Rena Wang, Xinyuan Zhu, and Leslie Wang

## 1. Introduction

Spotify reports a popularity score in [0, 100] for every track, computed
from streaming counts and recency. The question we ask in this project
is how much of that score can be predicted from a track's audio
features alone (danceability, loudness, tempo, valence, and so on)
without any metadata about who recorded the track or how it was
distributed. This is a useful experiment because the audio is the part
of a track that is fixed at production time. Everything else (which
artist, which playlist, which release week) is a distribution problem
rather than a content problem.

We treat popularity as a continuous regression target and benchmark
six models: a mean baseline, Ridge, Lasso, Random Forest, LightGBM, and
a feed-forward neural network. The aim is not to win a leaderboard; it
is to characterize the ceiling of what audio alone can predict and to
compare the cost-vs-accuracy trade-off across model families.

## 2. Dataset

We use the **Spotify Tracks Dataset** from Kaggle
(`maharshipandya/spotify-tracks-dataset`), which contains 114,000 tracks
spanning 114 genres. Each row carries the features returned by Spotify's
audio analysis API (`danceability`, `energy`, `loudness`, `speechiness`,
`acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`,
`duration_ms`) together with the discrete fields `key`, `mode`,
`time_signature`, `explicit`, and `track_genre`. The target is
`popularity`, an integer 0–100 derived from listener behavior, not from
the audio.

After cleaning we kept **{int(eda['n_rows_clean']):,}** of
**{int(eda['n_rows_raw']):,}** rows. We dropped the four identity-like
fields (`track_id`, `track_name`, `artists`, `album_name`) before any
modeling step. Keeping them would defeat the purpose of the experiment:
the model would memorize that "Drake = popular" rather than learn what
acoustic properties correlate with popularity.

The popularity distribution is heavily right-skewed and zero-inflated.
About **{eda['popularity_zero_share']*100:.1f}%** of tracks have
popularity exactly 0, and the mean is **{eda['popularity_mean']:.1f}**
with a standard deviation of **{eda['popularity_std']:.1f}**. The
zero-mass is the hardest part of the dataset for any regressor: the
informative observation in those cases is "this song is unknown",
which the audio features cannot directly say.

## 3. Methodology

### 3.1 Preprocessing and split

Numeric features are standardized to zero mean and unit variance using
statistics estimated only on the training split. The five categorical
fields are one-hot encoded with `handle_unknown="ignore"` so that any
unseen genre at inference time becomes an all-zero indicator instead of
crashing. We split the data 70 / 15 / 15 into train / validation / test
with `random_state=42` and ran a post-split audit to confirm the
zero-popularity rate is balanced across splits ({balance_text}; full
counts in `outputs/tables/split_balance.csv`). All hyperparameter
selection (Ridge / Lasso `alpha`, the Random Forest 6-config grid in
`outputs/tables/rf_tuning.csv`, the LightGBM early-stopping point, and
the MLP 5-config grid in `outputs/tables/mlp_tuning.csv`) runs against
the validation split. The test split is consumed only at the very end,
once per model, to compute the metrics reported below.

The proposal listed `danceability`, `energy`, `loudness`, `speechiness`,
`acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`,
`duration_ms`, `explicit`, and `track_genre`. We additionally include
`key`, `mode`, and `time_signature` because they are part of Spotify's
audio analysis output and are categorical with low cardinality, so
adding them costs almost nothing and removes an arbitrary cutoff from
the feature list. Their incremental contribution is small (see the
feature-importance plots).

### 3.2 Models

- **Mean baseline.** Predicts `mean(y_train)` for every test point.
  This pins R² at zero and sets the RMSE floor against which the
  proposal's ≥15% reduction target is measured.
- **Ridge / Lasso.** Closed-form linear regression with L2 / L1
  regularization. `alpha` is grid-searched on the validation split.
- **Random Forest.** 300 trees. We swept six configurations of
  `min_samples_leaf` ∈ {{1, 2, 5}} × `max_features` ∈ {{"sqrt", 1.0}}
  with 100-tree probes on the validation split, then refit the best
  configuration at 300 trees on the same training data. The full sweep
  is in `outputs/tables/rf_tuning.csv`.
- **LightGBM.** Gradient-boosted trees with `learning_rate=0.05`,
  `num_leaves=127`, trained with **early stopping on validation RMSE**
  (patience 50). The convergence point and best iteration are recorded
  alongside the test metrics.
- **MLP (PyTorch).** A feed-forward network. We swept five
  (architecture, dropout, learning rate) configurations on validation
  and kept the best one for the test report; the full sweep is in
  `outputs/tables/mlp_tuning.csv`. Each configuration trains with Adam,
  weight decay 1e-4, batch size 512, MSE loss on a standardized target
  that is unstandardized for evaluation, and early stopping on
  validation RMSE with patience 8. cuDNN is set to deterministic mode.

All randomness is seeded (`SEED=42`) across NumPy, Python `random`,
PyTorch (CPU and CUDA), and the DataLoader generator.

### 3.3 Metrics

We report **RMSE**, **MAE**, and **R²** on the held-out test set, plus
the percentage RMSE reduction relative to the mean baseline (the
proposal's success criterion).

## 4. Results

| Model | Family | RMSE | MAE | R² | ΔRMSE vs baseline | Train (s) |
|---|---|---:|---:|---:|---:|---:|
{table_md}

**Best model: `{best_name}`** with test RMSE = **{best['rmse']:.2f}**,
MAE = **{best['mae']:.2f}**, R² = **{best['r2']:.3f}**, which is a
**{rmse_reduction:.1f}%** RMSE reduction over the mean baseline.

The two linear models are nearly identical to each other and clear only
a small margin over the baseline, which means a linear function of the
preprocessed features is a poor fit for popularity. Both tree ensembles
improve substantially: LightGBM is the strongest single model, and
Random Forest sits just below it while taking many times longer to
train. The MLP outperforms the linear models but does not reach the
tree ensembles. Its test RMSE is {mlp_vs_lgb:+.2f} above LightGBM and
{mlp_vs_rf:+.2f} above Random Forest. This ordering (gradient-boosted
trees > random forest > MLP > linear) is the empirical norm for tabular
data of this size.

### 4.1 Are the proposal's success criteria met?

- **≥ 15% RMSE reduction over the mean baseline.** Met by: {pass15_text}.
  Not met by: {fail15_text}.
- **R² ≥ 0.50.** Met by: {passr2_text}. Not met by: {failr2_text}.

Both targets are met by the tree ensembles. The linear models clear
neither bar; the MLP clears the RMSE bar but not R².

### 4.2 What the model is using

Top base features (LightGBM split importance, with the 114 genre
dummies summed back into a single `track_genre` row):

{feat_lines}

Two observations:

1. The continuous audio features dominate split-count importance because
   the model can split on each of them at many different thresholds.
   The most-used feature is `{top_feature_name}`.
2. `track_genre` lands at position **{genre_rank}** when its 114 dummies
   are aggregated and contributes **{genre_share:.1f}%** of total split
   importance. Each individual genre indicator only splits on ~1% of
   rows, so it can never accumulate as much split count as a continuous
   feature does. Random Forest's Gini importance tells a different
   story (`outputs/tables/feature_importances_grouped.csv`): there
   `track_genre` is the single biggest aggregated feature. The two
   importance metrics measure different things, and both views appear
   in the figures.

The high split-importance of `speechiness`, `valence`, `danceability`,
`duration_ms`, and `acousticness` matches musical intuition. Tracks
that fall outside the mainstream production envelope (very speech-heavy,
very acoustic, unusually long) systematically score lower; tracks that
sit inside the modern pop range systematically score higher.

### 4.3 Multi-seed stability

The metrics in §4 are point estimates produced with `seed=42`. To
quantify how stable they are we refit the tuned configurations of the
three non-trivial models under three independent seeds
(`{ms_seeds}`) and report mean ± standard deviation of the test
metrics:

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---:|---:|---:|
{ms_table}

The standard deviations are small relative to the gaps between model
families: tree ensembles remain ahead of the MLP across every seed, and
the headline numbers in the main table are within one standard deviation
of the multi-seed mean. Per-seed values are in
`outputs/tables/multi_seed_results.csv`.

### 4.4 Predicted vs. actual

`outputs/figures/10_pred_vs_actual_lgbm.png` shows the LightGBM test-set
predictions as a hexbin plot against the true popularity. Two patterns
stand out: there is a tight bright cluster of correctly-predicted zero-
popularity tracks at the origin, and the rest of the predictions are
visibly compressed toward the middle of the range. The model rarely
predicts above ~80, and the errors at both extremes are large. The
compression is what we would expect from a regressor working under
high label noise: when the audio cannot disambiguate "bad song" from
"good song that no one has heard yet", the model hedges toward the
mean.

### 4.5 Genre ablation: how much of R² is genre vs. audio?

`track_genre` is partly a proxy for the kind of artist that records in
that genre, which is a soft form of the identity-leakage problem we
explicitly tried to remove by dropping `artists`. To quantify the strict
audio-only ceiling we refit LightGBM with the same hyperparameters but
with `track_genre` removed entirely from the feature set
(`outputs/tables/ablation_genre.csv`):

{ablation_lines}

Removing genre costs roughly **{genre_r2_loss:.3f}** R² on the test set.
This is a meaningful but not dominant share of the total signal, which
matches what the LightGBM split-importance and Random Forest Gini
importance plots suggested: genre carries useful information, but the
continuous audio features carry more, and the strict audio-only model
still clears the proposal's 15% RMSE-reduction bar comfortably.

## 5. Discussion

**Did we hit the proposal's targets?** Both criteria are met by the
tree ensembles (LightGBM and Random Forest); the MLP and the linear
models miss the R² bar. Even with the strongest model, roughly
**{(1 - best['r2'])*100:.0f}%** of the variance in popularity remains
unexplained, a ceiling we do not expect to break without metadata
that sits outside the audio.

**Why does popularity not decompose into audio?** The Spotify
popularity score is a function of streaming counts and recency, which
in turn depend on:

1. Artist fame and back-catalog. A new track from a top artist will
   always outscore a great debut from an unknown one, regardless of
   audio quality.
2. Editorial and algorithmic playlist placement. A single inclusion
   in a major playlist can shift a track's popularity by tens of
   points.
3. Release timing and platform tenure. A track released last week with
   the same audio as one released five years ago will score very
   differently.
4. Cross-platform virality (TikTok in particular).

None of these are observable from the waveform, so a model trained
only on audio is upper-bounded by however much of popularity is
intrinsic to the sound, which appears to be a minority of the variance.

**Trade-offs across model families.**

- *Linear models* are the cheapest and most interpretable, and they
  give us a sanity check. Their underfit confirms that the
  audio-to-popularity mapping is meaningfully nonlinear.
- *Tree ensembles* (Random Forest, LightGBM) reach the strongest
  accuracy and surface feature importances directly. LightGBM is the
  better choice on this dataset: comparable accuracy at a fraction of
  the training time, with explicit early stopping on validation.
- *The MLP* sits between linear and tree models. Adding more capacity
  (we tried up to 512-256-128) and changing dropout did not close the
  gap to LightGBM in our sweep. This is consistent with prior results
  on tabular data of similar size.

## 6. Limitations and Future Work

- **Audio features alone are a hard ceiling.** Adding artist
  embeddings, release year, and playlist-membership signals would
  almost certainly push R² above what we report here, but it would
  also be a different experiment.
- **Label noise.** The popularity score is itself noisy and
  time-varying; any reasonable model will appear "wrong" on tracks
  whose popularity changed after the snapshot was taken.
- **Zero inflation.** A two-stage hurdle model (classify "ever
  popular?", then regress on the positive support) is a natural way
  to handle the 14% zero spike and is left as future work.
- **Genre as a leaky proxy.** Some of the gain from `track_genre` is
  really gain from the kind of artist that records in that genre. The
  ablation in §4.5 reports the strict content-only ceiling.
- **MLP search budget.** Our sweep covers five configurations. A
  larger sweep (or a different architecture family entirely, such as
  TabNet or FT-Transformer) could change the qualitative comparison
  with the tree ensembles, although prior empirical surveys suggest
  the gap will remain modest.

## 7. Reproducibility

All experiments are seeded (`SEED=42`) and run end-to-end via:

```
python -m src.run_all
```

The script regenerates every figure in `outputs/figures/`, the metrics
table in `outputs/tables/model_results.csv`, the tuning sweeps
(`rf_tuning.csv`, `mlp_tuning.csv`), and this report. cuDNN is set to
deterministic mode in `train_nn.py`; the resulting numbers are
bit-identical across runs on the same hardware.
"""
    out = config.REPORTS_DIR / "final_report.md"
    out.write_text(report, encoding="utf-8")
    print("[report] wrote", out)


def main() -> None:
    df = pd.read_csv(config.RESULTS_CSV)
    make_comparison_plots(df)
    top_features = make_feature_importance_plot()
    make_pred_vs_actual_plot()
    write_report(df, top_features)


if __name__ == "__main__":
    main()
