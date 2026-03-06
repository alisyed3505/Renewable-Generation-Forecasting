# config/gru.py
"""
Configuration for Baseline GRU (Single-Site Only)

Mirrors config/baseline.py but for GRU model.
"""
# ==============================
# AUTO-VERSIONING
# ==============================
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.auto_version import get_versioned_paths

# Auto-detect next version or use environment variable override
# Usage: set MODEL_VERSION=v1 to force a specific version
_MODEL_VERSION = os.getenv("MODEL_VERSION")
if _MODEL_VERSION:
    _paths = get_versioned_paths("baseline_gru", version=_MODEL_VERSION)
else:
    _paths = get_versioned_paths("baseline_gru")

MODEL_VERSION = _paths["version"]

# ==============================
# DATA
# ==============================

DATA_FILE = "data/raw/pv_01.csv"

# ==============================
# TIME SERIES
# ==============================

TIME_STEPS = 8           # 8 timesteps (3-hour resolution -> 24h history)
TEST_SPLIT = 0.2         # final test split
VAL_SPLIT = 0.2          # validation split during training

# ==============================
# TRAINING
# ==============================

# v1 defaults (matching LSTM v1 for fair comparison)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# ==============================
# MODEL ARCHITECTURE
# ==============================

# v1 defaults (matching LSTM v1 for fair comparison)
GRU_UNITS_1 = 64
GRU_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

# v2 Optuna-optimized hyperparameters (50 trials, Trial 23, val_loss: 0.026967)
if MODEL_VERSION == "v2":
    GRU_UNITS_1 = 55
    GRU_UNITS_2 = 53
    DENSE_UNITS = 20
    DROPOUT_RATE = 0.452
    LEARNING_RATE = 0.0005002
    BATCH_SIZE = 128

# ==============================
# FEATURES (NO site_id)
# ==============================

FEATURE_COLS_BASELINE = [
    # Cyclical time encoding
    "hour_of_day_sin", "hour_of_day_cos",
    "month_of_year_sin", "month_of_year_cos",

    # Sun position
    "sunposition_thetaZ",
    "sunposition_solarAzimuth",

    # Clear-sky model
    "clearsky_diffuse",
    "clearsky_direct",
    "clearsky_global",

    # Weather features
    "TemperatureAt0",
    "RelativeHumidityAt0",
    "SolarRadiationGlobalAt0",
    "SolarRadiationDirectAt0",
    "SolarRadiationDiffuseAt0",
    "TotalCloudCoverAt0",
]

# ==============================
# ARTIFACT PATHS (Auto-versioned)
# ==============================

MODEL_PATH = _paths["model_path"]
SCALER_PATH = _paths["scaler_path"]
METRICS_PATH = _paths["metrics_path"]
PLOTS_DIR = _paths["plots_dir"]
