# config/embedded.py
"""
Configuration for Embedded LSTM (Multi-Site with Site Embeddings)
"""

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

EMBEDDING_DIM = 4
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

# ==============================
# FEATURES
# ==============================

from config.baseline import FEATURE_COLS_BASELINE

# ==============================
# ARTIFACT PATHS
# ==============================

MODEL_PATH = "models/embedded_lstm.keras"
SCALER_PATH = "models/embedded_scaler.pkl"
METRICS_PATH = "models/metrics/embedded_metrics.txt"
