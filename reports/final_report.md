# Empirical Evaluation of Predictive Models for Audio-Feature-Based Track Popularity

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

After cleaning we kept **114,000** of
**114,000** rows. We dropped the four identity-like
fields (`track_id`, `track_name`, `artists`, `album_name`) before any
modeling step. Keeping them would defeat the purpose of the experiment:
the model would memorize that "Drake = popular" rather than learn what
acoustic properties correlate with popularity.

The popularity distribution is heavily right-skewed and zero-inflated.
About **14.1%** of tracks have
popularity exactly 0, and the mean is **33.2**
with a standard deviation of **22.3**. The
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
zero-popularity rate is balanced across splits (train=14.0%, val=14.1%, test=14.1%; full
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
  `min_samples_leaf` ∈ {1, 2, 5} × `max_features` ∈ {"sqrt", 1.0}
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
| LightGBM | gradient_boosting | 14.563 | 9.772 | 0.570 | +34.4% | 8.4 |
| RandomForest | tree_ensemble | 15.312 | 10.785 | 0.524 | +31.0% | 88.9 |
| MLP_PyTorch | neural_network | 16.900 | 11.201 | 0.421 | +23.9% | 32.7 |
| Ridge | linear | 19.109 | 14.118 | 0.259 | +14.0% | 0.5 |
| Lasso | linear | 19.109 | 14.108 | 0.259 | +14.0% | 6.2 |
| MeanBaseline | baseline | 22.207 | 18.803 | -0.000 | +0.0% | 0.0 |

**Best model: `LightGBM`** with test RMSE = **14.56**,
MAE = **9.77**, R² = **0.570**, which is a
**34.4%** RMSE reduction over the mean baseline.

The two linear models are nearly identical to each other and clear only
a small margin over the baseline, which means a linear function of the
preprocessed features is a poor fit for popularity. Both tree ensembles
improve substantially: LightGBM is the strongest single model, and
Random Forest sits just below it while taking many times longer to
train. The MLP outperforms the linear models but does not reach the
tree ensembles. Its test RMSE is +2.34 above LightGBM and
+1.59 above Random Forest. This ordering (gradient-boosted
trees > random forest > MLP > linear) is the empirical norm for tabular
data of this size.

### 4.1 Are the proposal's success criteria met?

- **≥ 15% RMSE reduction over the mean baseline.** Met by: LightGBM (+34.4%), RandomForest (+31.0%), MLP_PyTorch (+23.9%).
  Not met by: Ridge (+14.0%), Lasso (+14.0%).
- **R² ≥ 0.50.** Met by: LightGBM (R²=0.570), RandomForest (R²=0.524). Not met by: MLP_PyTorch (R²=0.421), Ridge (R²=0.259), Lasso (R²=0.259).

Both targets are met by the tree ensembles. The linear models clear
neither bar; the MLP clears the RMSE bar but not R².

### 4.2 What the model is using

Top base features (LightGBM split importance, with the 114 genre
dummies summed back into a single `track_genre` row):

- `speechiness` (importance = 35155)
- `valence` (importance = 34374)
- `danceability` (importance = 34361)
- `duration_ms` (importance = 34356)
- `liveness` (importance = 33874)
- `acousticness` (importance = 32686)
- `tempo` (importance = 32128)
- `loudness` (importance = 31829)
- `energy` (importance = 31407)
- `instrumentalness` (importance = 25744)

Two observations:

1. The continuous audio features dominate split-count importance because
   the model can split on each of them at many different thresholds.
   The most-used feature is `speechiness`.
2. `track_genre` lands at position **11** when its 114 dummies
   are aggregated and contributes **6.5%** of total split
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
(`[42, 7, 13]`) and report mean ± standard deviation of the test
metrics:

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---:|---:|---:|
| LightGBM | 14.529 ± 0.032 | 9.722 ± 0.052 | 0.572 ± 0.002 |
| MLP_PyTorch | 17.016 ± 0.127 | 11.506 ± 0.279 | 0.413 ± 0.009 |
| RandomForest | 15.292 ± 0.019 | 10.769 ± 0.014 | 0.526 ± 0.001 |

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

- LightGBM **with `track_genre`**: RMSE 14.563, MAE 9.772, R² 0.570.
- LightGBM **without `track_genre`** (audio + key/mode/time_signature/explicit only): RMSE 15.279, MAE 10.312, R² 0.527.
- ΔR² from removing genre: **-0.043**.

Removing genre costs roughly **0.043** R² on the test set.
This is a meaningful but not dominant share of the total signal, which
matches what the LightGBM split-importance and Random Forest Gini
importance plots suggested: genre carries useful information, but the
continuous audio features carry more, and the strict audio-only model
still clears the proposal's 15% RMSE-reduction bar comfortably.

## 5. Discussion

**Did we hit the proposal's targets?** Both criteria are met by the
tree ensembles (LightGBM and Random Forest); the MLP and the linear
models miss the R² bar. Even with the strongest model, roughly
**43%** of the variance in popularity remains
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
