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

The main challenges we faced were (i) **zero-inflation** — about 14% of
tracks have popularity exactly 0, which a single-stage regressor cannot
explain from the audio alone; (ii) **label noise** — the popularity
score is driven by streaming counts, playlist placement, and release
recency, none of which are observable from the waveform, so any
audio-only model has a hard upper bound on attainable R²; (iii)
**leakage avoidance** — naive use of `track_name` / `artists` would let
the model memorize "Drake = popular" rather than learn audio↔popularity
structure, so we drop those fields before any modeling step; and
(iv) **environment portability** — running PyTorch and LightGBM in the
same process can deadlock on macOS Anaconda due to two competing
OpenMP runtimes, which we work around with the documented Intel flag
(see README).

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

For every model the **measure of fit** at evaluation time is RMSE on
the held-out test set; the per-model **training objective** is given
below.

- **Mean baseline.** Predicts `mean(y_train)` for every test point.
  No training objective. This pins R² at zero and sets the RMSE floor
  against which the proposal's ≥15% reduction target is measured.
- **Ridge.** Linear regression with L2 regularization. Objective:
  minimize ½‖y − Xβ‖² + α‖β‖²₂, solved in closed form.
- **Lasso.** Linear regression with L1 regularization. Objective:
  minimize ½‖y − Xβ‖² + α‖β‖₁, solved by coordinate descent.
- **Random Forest.** Bagged ensemble of regression trees. Each tree
  greedily chooses splits that maximize variance reduction
  (equivalently, minimize within-node MSE). 300 trees. We swept six
  configurations of `min_samples_leaf` ∈ {1, 2, 5} × `max_features` ∈
  {"sqrt", 1.0} with 100-tree probes on the validation split, then
  refit the best configuration at 300 trees on the same training data.
  The full sweep is in `outputs/tables/rf_tuning.csv`.
- **LightGBM.** Gradient-boosted trees minimizing squared error
  (`objective="regression"`, equivalently L2 loss on the residuals).
  Trained with `learning_rate=0.05`, `num_leaves=127`, and **early
  stopping on validation RMSE** (patience 50). The convergence point
  and best iteration are recorded alongside the test metrics.
- **MLP (PyTorch).** A feed-forward fully-connected network with three
  hidden layers (best config: 512 → 256 → 128), ReLU activations, and
  dropout after each hidden layer. The output is a single linear unit.
  Objective: mean squared error on a standardized target,
  L = (1/N) Σᵢ (ŷᵢ − yᵢ)², optimized by Adam (PyTorch defaults β₁=0.9,
  β₂=0.999, ε=1e-8) with weight decay 1e-4 and batch size 512.
  Predictions are unstandardized for evaluation. We swept five
  (architecture, dropout, learning rate) configurations on validation
  and kept the best one for the test report; the full sweep is in
  `outputs/tables/mlp_tuning.csv`. Each configuration uses early
  stopping on validation RMSE with patience 8. cuDNN is set to
  deterministic mode.

All randomness is seeded (`SEED=42`) across NumPy, Python `random`,
PyTorch (CPU and CUDA), and the DataLoader generator.

### 3.3 Metrics

We report **RMSE**, **MAE**, and **R²** on the held-out test set, plus
the percentage RMSE reduction relative to the mean baseline (the
proposal's success criterion).

## 4. Implementation Details

### 4.1 Preprocessing

- **Numeric features** (`danceability`, `energy`, `loudness`,
  `speechiness`, `acousticness`, `instrumentalness`, `liveness`,
  `valence`, `tempo`, `duration_ms`): `StandardScaler` (zero mean, unit
  variance) fit on the training split only.
- **Categorical features** (`track_genre`, `key`, `mode`,
  `time_signature`, `explicit`): `OneHotEncoder` with
  `handle_unknown="ignore"` so unseen categories at inference become
  all-zero indicator vectors instead of crashing. After encoding the
  feature matrix has 145 columns (10 numeric + 135 one-hot).
- **Target** (`popularity`): kept on its native 0–100 scale for tree
  models and linear models; standardized to zero mean / unit variance
  for the MLP and unstandardized after prediction for evaluation.
- **Identity-like fields dropped before modeling**: `track_id`,
  `track_name`, `artists`, `album_name`, `Unnamed: 0`.
- **Rows dropped**: only those with NaN in any numeric feature or in
  the target. Final usable rows: 114,000 / 114,000.
- **Split**: 70 / 15 / 15 train / val / test, `random_state=42`. The
  preprocessor is fit on the training split only.

### 4.2 Final hyperparameters per model

| Model | Final hyperparameters (after validation tuning) |
|---|---|
| Mean baseline | `mean(y_train) ≈ 33.24` |
| Ridge | `α = 10.0` (selected from {0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0}), `max_iter=20000` |
| Lasso | `α = 0.001` (selected from {0.001, 0.01, 0.05, 0.1, 0.5, 1.0}), `max_iter=20000` |
| Random Forest | `n_estimators=300`, `min_samples_leaf=1`, `max_features="sqrt"` (winner of 6-config validation sweep at 100 trees, refit at 300) |
| LightGBM | `n_estimators_cap=5000`, `learning_rate=0.05`, `num_leaves=127`, `min_child_samples=20`, `feature_fraction=0.9`, `bagging_fraction=0.9`, `bagging_freq=5`. Early stopping on validation RMSE with patience 50 → **best_iteration = 2905** |
| MLP (PyTorch) | hidden=(512x256x128), ReLU, dropout=0.3 after each hidden layer; optimizer = Adam (lr=0.001, β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=1e-4); batch_size=512; loss=MSE on standardized target; max_epochs=80; early stopping on validation RMSE with patience 8 → stopped at epoch 37 |

### 4.3 Validation tuning sweeps

- **Random Forest** (6 configs at 100 trees, ranked by val RMSE):

  | min_samples_leaf | max_features | val RMSE |
  |---:|---|---:|
  | 1 | sqrt | 15.660 |
  | 2 | 1.0 | 15.789 |
  | 1 | 1.0 | 15.825 |
  | 5 | 1.0 | 16.137 |
  | 2 | sqrt | 16.337 |
  | 5 | sqrt | 17.312 |

- **MLP** (5 configs, ranked by val RMSE):

  | hidden | dropout | lr | epochs (early-stop) | val RMSE |
  |---|---:|---:|---:|---:|
  | 512x256x128 | 0.3 | 0.001 | 37 | 17.223 |
  | 256x128x64 | 0.2 | 0.001 | 47 | 17.277 |
  | 512x256x128 | 0.2 | 0.001 | 34 | 17.280 |
  | 256x128x64 | 0.2 | 0.0005 | 79 | 17.358 |
  | 256x128x64 | 0.1 | 0.001 | 37 | 17.610 |

Full sweeps in `outputs/tables/rf_tuning.csv` and `outputs/tables/mlp_tuning.csv`.

### 4.4 Reproducibility

`SEED = 42` is set across NumPy, Python `random`, PyTorch (CPU + CUDA),
the DataLoader generator, sklearn `random_state`, and LightGBM
`random_state`. cuDNN is set to deterministic mode in
`src/train_nn.py`. End-to-end execution: `python -m src.run_all`.

## 5. Results

| Model | Family | RMSE | MAE | R² | ΔRMSE vs baseline | Train (s) |
|---|---|---:|---:|---:|---:|---:|
| LightGBM | gradient_boosting | 14.563 | 9.772 | 0.570 | +34.4% | 31.3 |
| RandomForest | tree_ensemble | 15.312 | 10.785 | 0.524 | +31.0% | 835.5 |
| MLP_PyTorch | neural_network | 17.084 | 11.633 | 0.408 | +23.1% | 189.9 |
| Ridge | linear | 19.109 | 14.118 | 0.259 | +14.0% | 0.6 |
| Lasso | linear | 19.109 | 14.108 | 0.259 | +14.0% | 3.0 |
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
tree ensembles. Its test RMSE is +2.52 above LightGBM and
+1.77 above Random Forest. This ordering (gradient-boosted
trees > random forest > MLP > linear) is the empirical norm for tabular
data of this size.

### 5.1 Are the proposal's success criteria met?

- **≥ 15% RMSE reduction over the mean baseline.** Met by: LightGBM (+34.4%), RandomForest (+31.0%), MLP_PyTorch (+23.1%).
  Not met by: Ridge (+14.0%), Lasso (+14.0%).
- **R² ≥ 0.50.** Met by: LightGBM (R²=0.570), RandomForest (R²=0.524). Not met by: MLP_PyTorch (R²=0.408), Ridge (R²=0.259), Lasso (R²=0.259).

Both targets are met by the tree ensembles. The linear models clear
neither bar; the MLP clears the RMSE bar but not R².

### 5.2 What the model is using

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

### 5.3 Multi-seed stability

The metrics in §5 are point estimates produced with `seed=42`. To
quantify how stable they are we refit the tuned configurations of the
three non-trivial models under three independent seeds
(`[42, 7, 13]`) and report mean ± standard deviation of the test
metrics:

| Model | RMSE (mean ± std) | MAE (mean ± std) | R² (mean ± std) |
|---|---:|---:|---:|
| LightGBM | 14.529 ± 0.032 | 9.722 ± 0.052 | 0.572 ± 0.002 |
| MLP_PyTorch | 17.073 ± 0.077 | 11.641 ± 0.022 | 0.409 ± 0.005 |
| RandomForest | 15.292 ± 0.019 | 10.769 ± 0.014 | 0.526 ± 0.001 |

The standard deviations are small relative to the gaps between model
families: tree ensembles remain ahead of the MLP across every seed, and
the headline numbers in the main table are within one standard deviation
of the multi-seed mean. Per-seed values are in
`outputs/tables/multi_seed_results.csv`.

### 5.4 Predicted vs. actual

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

### 5.5 Genre ablation: how much of R² is genre vs. audio?

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

## 6. Discussion

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

## 7. Limitations and Future Work

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
  ablation in §5.5 reports the strict content-only ceiling.
- **MLP search budget.** Our sweep covers five configurations. A
  larger sweep (or a different architecture family entirely, such as
  TabNet or FT-Transformer) could change the qualitative comparison
  with the tree ensembles, although prior empirical surveys suggest
  the gap will remain modest.


## 8. Reproducibility

Please follow the instructions in `README.md` to set up the environment, install the required packages, download the dataset, and run the project.

All results reported in this paper were reproduced on the Yale HPC Code Server with the following resource settings:

```text
Number of CPU cores per node: 3
Memory per CPU core: 50 GiB
Partition: devel
```

All experiments are seeded with `SEED = 42` across NumPy, Python `random`, PyTorch, scikit-learn, and LightGBM whenever applicable. The full pipeline can be run end-to-end from the project root directory using:

```bash
python -m src.run_all
```

This command regenerates all major project outputs, including the figures in `outputs/figures/`, the evaluation tables in `outputs/tables/`, the model comparison results in `outputs/tables/model_results.csv`, the tuning results in `outputs/tables/rf_tuning.csv` and `outputs/tables/mlp_tuning.csv`, the ablation results in `outputs/tables/ablation_genre.csv`, the multi-seed results, and the final report.

For the neural network model, cuDNN is set to deterministic mode in `src/train_nn.py` to improve reproducibility. Under the same hardware and software environment, the results should be reproducible across repeated runs.


### 8.1 HPC Setup and Run Instructions


This project was tested with Python 3.9–3.11. CPU-only execution works, although GPU can speed up the neural network part.

#### 1. Open Code Server on HPC

Launch **Code Server** from the HPC OnDemand portal. Set the working directory to your project folder, for example:

```bash
 /home/cpsc3810_netid/project_cpsc3810/cpsc3810_netid/spotify-popularity-prediction
```

#### 2. Create and activate a Python environment

From the project root directory:
```bash
cd /home/cpsc3810_netid/project_cpsc3810/cpsc3810_netid/spotify-popularity-prediction

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 3. Download the Dataset


The pipeline expects the dataset at:
```
data/spotify_tracks.csv
```
Download it with:
```
mkdir -p data
python -c "from datasets import load_dataset; load_dataset('maharshipandya/spotify-tracks-dataset', split='train').to_csv('data/spotify_tracks.csv', index=False)"
```
The README specifies that the dataset is not included in the repo and must be placed at data/spotify_tracks.csv

#### 4. Run the full pipeline
For a quick direct run:
```
python -m src.run_all
```
This command regenerates the figures, result tables, and final report.


#### 5. Check outputs

After the job finishes, check:
```
ls outputs/figures
ls outputs/tables
ls reports
```
Expected important files include:
```
outputs/tables/model_results.csv
outputs/tables/rf_tuning.csv
outputs/tables/mlp_tuning.csv
outputs/tables/ablation_genre.csv
outputs/figures/
reports/final_report.md
```
