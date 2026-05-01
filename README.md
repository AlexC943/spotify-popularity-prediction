# Empirical Evaluation of Predictive Models for Audio-Feature-Based Track Popularity

Yale CPSC 5810 (Introduction to Machine Learning) — final project.
Predicts Spotify track popularity (0–100) from audio features using the
public Kaggle Spotify Tracks Dataset.

## What is in this repo

```
data/                           # raw CSV (~20 MB; not committed if you add .gitignore)
src/
  config.py                     # paths, seeds, feature lists
  data.py                       # load + clean + split + preprocessor
  eda.py                        # exploratory plots & summary table
  metrics.py                    # RMSE / MAE / R²
  train_classical.py            # mean baseline, Ridge, Lasso, RF, LightGBM
  train_nn.py                   # PyTorch MLP with early stopping
  compare.py                    # comparison plots + final report draft
  run_all.py                    # end-to-end orchestrator
outputs/
  figures/                      # 11 PNGs (EDA + learning curve + comparisons)
  tables/                       # model_results.csv, feature_importances.csv, eda_summary.csv,
                                # split_balance.csv, rf_tuning.csv, mlp_tuning.csv, ...
  models/                       # serialized preprocessor + every trained model
reports/
  final_report.md               # auto-generated, populated with real numbers
```

## Setup

Tested on Windows 11, Python 3.11, CUDA 12.x. CPU also works.

```bash
python -m pip install -r requirements.txt
```

## Get the dataset

The pipeline expects `data/spotify_tracks.csv`. There are two equivalent
ways to obtain it:

**Option A — HuggingFace mirror (no Kaggle account needed):**
```python
from datasets import load_dataset
ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
ds.to_csv("data/spotify_tracks.csv", index=False)
```

**Option B — Kaggle CLI:**
```bash
# requires kaggle.json with API credentials
kaggle datasets download -d maharshipandya/-spotify-tracks-dataset -p data --unzip
mv data/dataset.csv data/spotify_tracks.csv
```

## Reproduce all results

```bash
python -m src.run_all
```

This runs EDA, trains every model (with validation tuning sweeps for RF
and the MLP), regenerates every figure, and rewrites
`reports/final_report.md` with the actual numbers.

The pipeline forces the Matplotlib `Agg` backend internally
(`src/_plotting.py`), so no display server / Tk is required. If your
Matplotlib install is misconfigured anyway you can also export
`MPLBACKEND=Agg` before running.

Approximate wall-clock on this machine (RTX-class GPU + CPU): ~9 min
(RF tuning ~5 min, LightGBM ~10 s, MLP sweep ~2 min). On CPU only,
~12 min.

## Run individual stages

```bash
python -m src.eda                # EDA + figures 01–05
python -m src.train_classical    # writes outputs/tables/model_results.csv
python -m src.train_nn           # appends MLP row + writes learning curve
python -m src.compare            # comparison plots + report
```

## Reproducibility notes

- All random seeds are set to `42` (NumPy, Python `random`, PyTorch CPU + CUDA).
  cuDNN is forced into deterministic mode in `src/train_nn.py`, so two
  consecutive runs produce bit-identical metrics.
- The 70 / 15 / 15 train / val / test split uses the same seed.
- The preprocessor (StandardScaler + OneHotEncoder) is **fit only on the
  training split**. Validation and test data are passed through `transform`
  alone, so no statistics from val / test leak into training.
- Hyperparameters are tuned on the **validation** split (Ridge / Lasso
  `alpha`, the RF 6-config grid in `outputs/tables/rf_tuning.csv`, the
  LightGBM early-stopping point, and the MLP 5-config grid in
  `outputs/tables/mlp_tuning.csv`). The test set is consumed only at
  the very end, once per model, to compute the metrics in
  `outputs/tables/model_results.csv`. EDA computes the per-split
  `zero_share` summary in `outputs/tables/split_balance.csv` as a
  post-split audit, but does not feed test labels into any model.
- Identity-like fields (`track_id`, `track_name`, `artists`, `album_name`)
  are dropped before any modeling step.
- Multi-seed stability (`outputs/tables/multi_seed_results.csv`) and a
  genre-ablation experiment (`outputs/tables/ablation_genre.csv`) are
  produced as additional rigor checks; both are referenced inline in the
  final report.

## Headline numbers

See `outputs/tables/model_results.csv` and `reports/final_report.md` for
the full table. The proposal's success criteria are evaluated for each
model in §4.1 of the report; in summary:

- R² ≥ 0.50 — met by both tree ensembles (LightGBM, Random Forest).
- ≥ 15% RMSE reduction over the mean baseline — met by both tree
  ensembles and the MLP. The two linear models fall just below the
  threshold (~14%).
