# config/embedded.py
"""
Configuration for Embedded LSTM (Multi-Site with Site Embeddings)
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
    # Use explicit version from environment
    _paths = get_versioned_paths("embedded_lstm", version=_MODEL_VERSION)
else:
    # Auto-detect next available version
    _paths = get_versioned_paths("embedded_lstm")

MODEL_VERSION = _paths["version"]

# ==============================
# DATA
# ==============================

DATA_GLOB = "data/raw/pv_*.csv"
NUM_SITES = 21

# ==============================
# TIME SERIES
# ==============================

TIME_STEPS = 8           # 8 timesteps × 3h = 24h
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

# ==============================
# TRAINING
# ==============================

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# ==============================
# MODEL
# ==============================

# v1 defaults (baseline multi-site LSTM)
EMBEDDING_DIM = 4
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

# v2 Optuna-optimized hyperparameters (Trial 26, val_loss: 0.006052)
if MODEL_VERSION == "v2":
    EMBEDDING_DIM = 2
    LSTM_UNITS_1 = 85
    LSTM_UNITS_2 = 33
    DENSE_UNITS = 27
    DROPOUT_RATE = 0.243373
    LEARNING_RATE = 0.002073
    BATCH_SIZE = 64

# v3: Leave-One-Site-Out experiment (v1 defaults for fair generalization test)
if MODEL_VERSION == "v3":
    NUM_SITES = 21  # 20 training + 1 cold-start embedding slot for unseen site
    EMBEDDING_DIM = 4
    LEAVE_ONE_OUT = True
else:
    LEAVE_ONE_OUT = False

# ==============================
# FEATURES
# ==============================

from config.baseline import FEATURE_COLS_BASELINE

# ==============================
# ARTIFACT PATHS (Auto-versioned)
# ==============================

MODEL_PATH = _paths["model_path"]
SCALER_PATH = _paths["scaler_path"]
METRICS_PATH = _paths["metrics_path"]
PLOTS_DIR = _paths["plots_dir"]  # NEW: Version-specific plots directory
