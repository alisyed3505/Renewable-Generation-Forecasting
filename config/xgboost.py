# config/xgboost.py
"""
Configuration for XGBoost (Single-Site Solar Power Forecasting)
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
    _paths = get_versioned_paths("xgboost", version=_MODEL_VERSION)
else:
    # Auto-detect next available version
    _paths = get_versioned_paths("xgboost")

MODEL_VERSION = _paths["version"]

# ==============================
# DATA
# ==============================

# Use baseline data file (single site)
from config.baseline import DATA_FILE, TIME_STEPS, TEST_SPLIT

# ==============================
# MODEL HYPERPARAMETERS (Optuna-optimized for v2)
# ==============================

if MODEL_VERSION == "v1":
    # v1: Default hyperparameters
    MAX_DEPTH = 6
    MIN_CHILD_WEIGHT = 1
    LEARNING_RATE = 0.1
    N_ESTIMATORS = 100
    GAMMA = 0.0
    SUBSAMPLE = 1.0
    COLSAMPLE_BYTREE = 1.0
    REG_ALPHA = 0.0
    REG_LAMBDA = 1.0
else:
    # v2+: Optuna-optimized hyperparameters
    MAX_DEPTH = 9
    MIN_CHILD_WEIGHT = 6
    LEARNING_RATE = 0.029710
    N_ESTIMATORS = 130
    GAMMA = 0.007025
    SUBSAMPLE = 0.845905
    COLSAMPLE_BYTREE = 0.879990
    REG_ALPHA = 3.353364
    REG_LAMBDA = 2.981327

# ==============================
# ARTIFACT PATHS (Auto-versioned)
# ==============================

MODEL_PATH = _paths["model_path"]
METRICS_PATH = _paths["metrics_path"]
PLOTS_DIR = _paths["plots_dir"]
