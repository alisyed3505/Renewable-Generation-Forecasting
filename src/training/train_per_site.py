# src/training/train_per_site.py
"""
Per-Site Training & Comparison

Trains Naive 24h, XGBoost (v2 Optuna params), Baseline LSTM (v1 arch),
and Baseline GRU (v1 arch) on EACH of the 21 PV sites independently.
Saves per-site metrics and generates comparison analysis.

Usage:
    python src/training/train_per_site.py              # all 21 sites
    python src/training/train_per_site.py --sites 1 3  # specific sites only
    python src/training/train_per_site.py --skip-lstm   # Skip LSTM (fast)
    python src/training/train_per_site.py --skip-gru    # Skip GRU
    python src/training/train_per_site.py --gru-only    # Run GRU only (for adding to existing results)
"""

import sys
import os
import argparse
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.metrics import save_metrics


# ──────────────────────────────────────────────────────────
# Constants (match existing project conventions)
# ──────────────────────────────────────────────────────────
DATA_DIR = "data/raw"
METRICS_DIR = "models/metrics/per_site"

FEATURE_COLS = [
    "hour_of_day_sin", "hour_of_day_cos",
    "month_of_year_sin", "month_of_year_cos",
    "sunposition_thetaZ", "sunposition_solarAzimuth",
    "clearsky_diffuse", "clearsky_direct", "clearsky_global",
    "TemperatureAt0", "RelativeHumidityAt0",
    "SolarRadiationGlobalAt0", "SolarRadiationDirectAt0",
    "SolarRadiationDiffuseAt0", "TotalCloudCoverAt0",
]

TIME_STEPS = 8
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# LSTM hyperparameters (v1 defaults — fair comparison)
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50

# GRU hyperparameters (v1 defaults — matching LSTM for fair comparison)
GRU_UNITS_1 = 64
GRU_UNITS_2 = 32

# XGBoost hyperparameters (v2 Optuna-optimized — best available)
XGB_PARAMS = {
    "max_depth": 9,
    "min_child_weight": 6,
    "learning_rate": 0.029710,
    "n_estimators": 130,
    "gamma": 0.007025,
    "subsample": 0.845905,
    "colsample_bytree": 0.879990,
    "reg_alpha": 3.353364,
    "reg_lambda": 2.981327,
}


# ──────────────────────────────────────────────────────────
# Data loading (standalone — no circular config imports)
# ──────────────────────────────────────────────────────────
def load_site_data(site_name):
    """Load and validate a single site CSV."""
    import pandas as pd

    path = os.path.join(DATA_DIR, f"{site_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, delimiter=";")
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features in {site_name}: {missing}")

    if "power_normed" not in df.columns:
        raise ValueError(f"Target column 'power_normed' not found in {site_name}")

    if "time_idx" in df.columns:
        df = df.sort_values("time_idx")

    return df


# ──────────────────────────────────────────────────────────
# Naive 24h
# ──────────────────────────────────────────────────────────
def train_naive(site_name, df):
    """Evaluate the naive 24h-ago persistence baseline on a site."""
    y = df["power_normed"].fillna(0).values.astype(np.float32)
    split_idx = int(len(y) * (1 - TEST_SPLIT))
    y_test = y[split_idx:]

    if len(y_test) <= TIME_STEPS:
        return None

    y_true = y_test[TIME_STEPS:]
    y_pred = y_test[:-TIME_STEPS]

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    metrics_path = os.path.join(METRICS_DIR, f"naive_24h_{site_name}_metrics.json")
    save_metrics(
        model_name="naive_24h",
        model_version="per_site",
        metrics={"rmse": rmse, "mae": mae},
        output_path=metrics_path,
        extra_info={
            "model_type": "naive",
            "scope": "per_site",
            "site": site_name,
            "time_steps": TIME_STEPS,
            "n_test_samples": len(y_true),
        },
    )
    return {"rmse": rmse, "mae": mae}


# ──────────────────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────────────────
def train_xgboost(site_name, df):
    """Train XGBoost (v2 Optuna params) on a single site."""
    import xgboost as xgb

    X = df[FEATURE_COLS].ffill().bfill().values.astype(np.float32)
    y = df["power_normed"].fillna(0).values.astype(np.float32)

    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        verbosity=0,
        **XGB_PARAMS,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = np.clip(model.predict(X_test), 0, 1)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    metrics_path = os.path.join(METRICS_DIR, f"xgboost_{site_name}_metrics.json")
    save_metrics(
        model_name="xgboost",
        model_version="per_site",
        metrics={"rmse": rmse, "mae": mae},
        output_path=metrics_path,
        extra_info={
            "model_type": "xgboost",
            "scope": "per_site",
            "site": site_name,
            **XGB_PARAMS,
        },
    )
    return {"rmse": rmse, "mae": mae}


# ──────────────────────────────────────────────────────────
# Baseline LSTM
# ──────────────────────────────────────────────────────────
def train_lstm(site_name, df):
    """Train baseline LSTM (v1 architecture, 50 epochs + early stop) on a single site."""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Preprocess
    X = df[FEATURE_COLS].ffill().bfill()
    y = df["power_normed"].fillna(0).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - TIME_STEPS):
        X_seq.append(X_scaled[i : i + TIME_STEPS])
        y_seq.append(y[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train/test split
    split_idx = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Build model (v1 architecture)
    model = Sequential([
        LSTM(LSTM_UNITS_1, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2]), name="lstm_1"),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS_2, return_sequences=False, name="lstm_2"),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu", name="dense_1"),
        Dense(1, activation="relu", name="output"),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])

    # Train with early stopping
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=0,
        ),
    ]

    model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks, verbose=0,
    )

    # Evaluate
    y_pred = np.clip(model.predict(X_test, verbose=0).flatten(), 0, 1)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    metrics_path = os.path.join(METRICS_DIR, f"baseline_lstm_{site_name}_metrics.json")
    save_metrics(
        model_name="baseline_lstm",
        model_version="per_site",
        metrics={"rmse": rmse, "mae": mae},
        output_path=metrics_path,
        extra_info={
            "model_type": "lstm",
            "scope": "per_site",
            "site": site_name,
            "time_steps": TIME_STEPS,
            "epochs_max": EPOCHS,
            "lstm_units": [LSTM_UNITS_1, LSTM_UNITS_2],
            "dense_units": DENSE_UNITS,
            "dropout_rate": DROPOUT_RATE,
        },
    )

    # Clean up to free GPU memory
    del model
    tf.keras.backend.clear_session()

    return {"rmse": rmse, "mae": mae}


# ──────────────────────────────────────────────────────────
# Baseline GRU
# ──────────────────────────────────────────────────────────
def train_gru(site_name, df):
    """Train baseline GRU (v1 architecture, 50 epochs + early stop) on a single site."""
    import tensorflow as tf
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Preprocess
    X = df[FEATURE_COLS].ffill().bfill()
    y = df["power_normed"].fillna(0).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - TIME_STEPS):
        X_seq.append(X_scaled[i : i + TIME_STEPS])
        y_seq.append(y[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Train/test split
    split_idx = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    # Build model (v1 architecture — same structure as LSTM but with GRU layers)
    model = Sequential([
        GRU(GRU_UNITS_1, return_sequences=True,
             input_shape=(X_train.shape[1], X_train.shape[2]), name="gru_1"),
        Dropout(DROPOUT_RATE),
        GRU(GRU_UNITS_2, return_sequences=False, name="gru_2"),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu", name="dense_1"),
        Dense(1, activation="relu", name="output"),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse", metrics=["mae"])

    # Train with early stopping
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=0,
        ),
    ]

    model.fit(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks, verbose=0,
    )

    # Evaluate
    y_pred = np.clip(model.predict(X_test, verbose=0).flatten(), 0, 1)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    metrics_path = os.path.join(METRICS_DIR, f"baseline_gru_{site_name}_metrics.json")
    save_metrics(
        model_name="baseline_gru",
        model_version="per_site",
        metrics={"rmse": rmse, "mae": mae},
        output_path=metrics_path,
        extra_info={
            "model_type": "gru",
            "scope": "per_site",
            "site": site_name,
            "time_steps": TIME_STEPS,
            "epochs_max": EPOCHS,
            "gru_units": [GRU_UNITS_1, GRU_UNITS_2],
            "dense_units": DENSE_UNITS,
            "dropout_rate": DROPOUT_RATE,
        },
    )

    # Clean up
    del model
    tf.keras.backend.clear_session()

    return {"rmse": rmse, "mae": mae}


# ──────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────
def get_all_sites():
    """Return sorted list of site names (pv_01 .. pv_21)."""
    return [f"pv_{i:02d}" for i in range(1, 22)]


def run_per_site(sites=None, skip_lstm=False, skip_gru=False, gru_only=False):
    """Run per-site training for selected sites."""
    if sites is None:
        sites = get_all_sites()

    os.makedirs(METRICS_DIR, exist_ok=True)

    results = {}
    total = len(sites)
    n_models = 4 - int(skip_lstm) - int(skip_gru)

    for idx, site in enumerate(sites, 1):
        print(f"\n{'=' * 60}")
        print(f"   SITE {idx}/{total}: {site}")
        print(f"{'=' * 60}")

        df = load_site_data(site)
        site_results = {}

        if not gru_only:
            # Naive
            print(f"  [1/{n_models}] Naive 24h ... ", end="", flush=True)
            r = train_naive(site, df)
            if r:
                print(f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}")
                site_results["naive_24h"] = r
            else:
                print("SKIPPED (not enough data)")

            # XGBoost
            print(f"  [2/{n_models}] XGBoost  ... ", end="", flush=True)
            r = train_xgboost(site, df)
            print(f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}")
            site_results["xgboost"] = r

            # LSTM
            if not skip_lstm:
                print(f"  [3/{n_models}] LSTM     ... ", end="", flush=True)
                r = train_lstm(site, df)
                print(f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}")
                site_results["baseline_lstm"] = r
            else:
                print(f"  [3/{n_models}] LSTM     ... SKIPPED")

        # GRU
        if not skip_gru:
            step = n_models if not gru_only else 1
            print(f"  [{step}/{step}] GRU      ... ", end="", flush=True)
            r = train_gru(site, df)
            print(f"RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}")
            site_results["baseline_gru"] = r

        results[site] = site_results

    # Summary
    print(f"\n{'=' * 60}")
    print("   PER-SITE TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"   Sites trained: {len(results)}")
    print(f"   Metrics saved to: {METRICS_DIR}/")

    # Save combined results JSON
    combined_path = os.path.join(METRICS_DIR, "per_site_summary.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"   Summary saved to: {combined_path}")

    return results


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-site model training")
    parser.add_argument("--sites", nargs="+", type=int, help="Site numbers to train (e.g. 1 3 5)")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training")
    parser.add_argument("--skip-gru", action="store_true", help="Skip GRU training")
    parser.add_argument("--gru-only", action="store_true", help="Run GRU only (add to existing results)")
    args = parser.parse_args()

    if args.sites:
        site_list = [f"pv_{s:02d}" for s in args.sites]
    else:
        site_list = None

    run_per_site(sites=site_list, skip_lstm=args.skip_lstm,
                 skip_gru=args.skip_gru, gru_only=args.gru_only)
