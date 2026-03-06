# config/embedded_gru.py
"""
Configuration for Embedded GRU (Multi-Site with Site Embeddings)
Mirrors config/embedded.py but for GRU model.
"""

# ==============================
# AUTO-VERSIONING
# ==============================
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.auto_version import get_versioned_paths

_MODEL_VERSION = os.getenv("MODEL_VERSION")
if _MODEL_VERSION:
    _paths = get_versioned_paths("embedded_gru", version=_MODEL_VERSION)
else:
    _paths = get_versioned_paths("embedded_gru")

MODEL_VERSION = _paths["version"]

# ==============================
# DATA
# ==============================

DATA_GLOB = "data/raw/pv_*.csv"
NUM_SITES = 21

# ==============================
# TIME SERIES
# ==============================

TIME_STEPS = 8
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

# v1 defaults (matching embedded LSTM v1 for fair comparison)
EMBEDDING_DIM = 4
GRU_UNITS_1 = 64
GRU_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

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
PLOTS_DIR = _paths["plots_dir"]
