# config/baseline.py
"""
Configuration for Baseline LSTM (Single-Site Only)

IMPORTANT:
- This config is ONLY for the baseline model
- It assumes exactly ONE PV site
- site_id is NOT used
"""
# ==============================
# MODEL VERSION
# ==============================
MODEL_VERSION = "v1"

# ==============================
# DATA
# ==============================

# Single fixed site (baseline constraint)
DATA_FILE = "data/raw/pv_01.csv"

# ==============================
# TIME SERIES
# ==============================

TIME_STEPS = 8           # 8 timesteps (3-hour resolution → 24h history)
TEST_SPLIT = 0.2         # final test split
VAL_SPLIT = 0.2          # validation split during training

# ==============================
# TRAINING
# ==============================

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# ==============================
# MODEL ARCHITECTURE
# ==============================

LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

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
# ARTIFACT PATHS
# ==============================

MODEL_PATH = f"models/baseline_lstm_{MODEL_VERSION}.keras"
SCALER_PATH = f"models/baseline_scaler_{MODEL_VERSION}.pkl"
METRICS_PATH = f"models/metrics/baseline_metrics_{MODEL_VERSION}.json"
