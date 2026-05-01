"""Project-wide configuration: paths, seeds, hyperparameters."""
from __future__ import annotations
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "spotify_tracks.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for d in (FIGURES_DIR, TABLES_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
SEED = 42

# --- Columns ---
TARGET = "popularity"
LEAKAGE_COLS = ["Unnamed: 0", "track_id", "track_name", "artists", "album_name"]

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
]
CATEGORICAL_FEATURES = ["track_genre", "key", "mode", "time_signature", "explicit"]

# --- Splits ---
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# --- Output table ---
RESULTS_CSV = TABLES_DIR / "model_results.csv"
