# src/evaluation/ts_cross_validation.py
"""
Time-Series Cross-Validation (Expanding Window)

Tests model stability across different time periods using expanding window CV.
No lookahead bias — each fold trains on ALL data before the test window.

Folds:
  Fold 1: Train [0-40%], Test [40-50%]
  Fold 2: Train [0-50%], Test [50-60%]
  Fold 3: Train [0-60%], Test [60-70%]
  Fold 4: Train [0-70%], Test [70-80%]
  Fold 5: Train [0-80%], Test [80-100%]

Runs on pv_01 for: XGBoost, Baseline LSTM, Baseline GRU
"""

import sys
import os

# Import xgboost before sys.path manipulation (avoid src/evaluation/xgboost/ shadow)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_eval_dir = _script_dir
_removed = [p for p in sys.path if _eval_dir in os.path.abspath(p)]
for p in _removed:
    sys.path.remove(p)
if "xgboost" in sys.modules:
    del sys.modules["xgboost"]
import xgboost as _xgb
for p in _removed:
    sys.path.insert(0, p)

sys.path.insert(0, os.path.abspath(os.path.join(_script_dir, "../..")))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
DATA_FILE = "data/raw/pv_01.csv"
OUTPUT_DIR = "models/metrics/cross_validation"
PLOTS_DIR = "src/evaluation/cross_validation"
TIME_STEPS = 8
RANDOM_SEED = 42

FEATURE_COLS = [
    "hour_of_day_sin", "hour_of_day_cos",
    "month_of_year_sin", "month_of_year_cos",
    "sunposition_thetaZ", "sunposition_solarAzimuth",
    "clearsky_diffuse", "clearsky_direct", "clearsky_global",
    "TemperatureAt0", "RelativeHumidityAt0",
    "SolarRadiationGlobalAt0", "SolarRadiationDirectAt0",
    "SolarRadiationDiffuseAt0", "TotalCloudCoverAt0",
]

# XGBoost best params (v2 Optuna)
XGB_PARAMS = {
    "max_depth": 9, "min_child_weight": 6,
    "learning_rate": 0.029710, "n_estimators": 130,
    "gamma": 0.007025, "subsample": 0.845905,
    "colsample_bytree": 0.879990,
    "reg_alpha": 3.353364, "reg_lambda": 2.981327,
}

# Fold definitions: (train_end_fraction, test_end_fraction)
FOLDS = [
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 1.00),
]


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
def load_data():
    """Load and preprocess pv_01."""
    df = pd.read_csv(DATA_FILE, delimiter=";")
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]
    if "time_idx" in df.columns:
        df = df.sort_values("time_idx")
    return df


# ──────────────────────────────────────────────────────────
# Model trainers
# ──────────────────────────────────────────────────────────
def cv_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost on fold and return RMSE."""
    model = _xgb.XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_SEED,
        verbosity=0, **XGB_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = np.clip(model.predict(X_test), 0, 1)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    return rmse, mae


def cv_rnn(X_train_seq, y_train_seq, X_test_seq, y_test_seq, model_type="lstm"):
    """Train LSTM or GRU on fold and return RMSE."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    if model_type == "lstm":
        from tensorflow.keras.layers import LSTM as RNNLayer
    else:
        from tensorflow.keras.layers import GRU as RNNLayer

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    model = Sequential([
        RNNLayer(64, return_sequences=True,
                 input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        Dropout(0.2),
        RNNLayer(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="relu"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.fit(
        X_train_seq, y_train_seq, epochs=50, batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )
    y_pred = np.clip(model.predict(X_test_seq, verbose=0).flatten(), 0, 1)
    rmse = float(np.sqrt(np.mean((y_test_seq - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test_seq - y_pred)))

    del model
    tf.keras.backend.clear_session()

    return rmse, mae


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def run_cv():
    print("=" * 60)
    print("   TIME-SERIES CROSS-VALIDATION (Expanding Window)")
    print("   Site: pv_01")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()
    n = len(df)
    print(f"   Total samples: {n}")

    # Prepare features
    X_flat = df[FEATURE_COLS].ffill().bfill().values.astype(np.float32)
    y_flat = df["power_normed"].fillna(0).values.astype(np.float32)

    # For RNN: create sequences from full data, then split
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS].ffill().bfill())
    X_seq_all, y_seq_all = [], []
    for i in range(len(X_scaled) - TIME_STEPS):
        X_seq_all.append(X_scaled[i:i + TIME_STEPS])
        y_seq_all.append(y_flat[i + TIME_STEPS])
    X_seq_all = np.array(X_seq_all)
    y_seq_all = np.array(y_seq_all)

    models = ["XGBoost", "LSTM", "GRU"]
    results = {m: {"rmse": [], "mae": []} for m in models}

    for fold_idx, (train_end, test_end) in enumerate(FOLDS):
        print(f"\n  Fold {fold_idx + 1}/5: "
              f"Train [0-{int(train_end*100)}%] "
              f"Test [{int(train_end*100)}-{int(test_end*100)}%]")

        # XGBoost splits (flat features, no sequences)
        train_end_idx = int(n * train_end)
        test_end_idx = int(n * test_end)

        X_train_f, y_train_f = X_flat[:train_end_idx], y_flat[:train_end_idx]
        X_test_f, y_test_f = X_flat[train_end_idx:test_end_idx], y_flat[train_end_idx:test_end_idx]

        # XGBoost
        print(f"    XGBoost ... ", end="", flush=True)
        rmse, mae = cv_xgboost(X_train_f, y_train_f, X_test_f, y_test_f)
        results["XGBoost"]["rmse"].append(rmse)
        results["XGBoost"]["mae"].append(mae)
        print(f"RMSE={rmse:.4f}  MAE={mae:.4f}")

        # RNN splits (sequences)
        n_seq = len(X_seq_all)
        train_end_seq = int(n_seq * train_end)
        test_end_seq = int(n_seq * test_end)

        X_train_s = X_seq_all[:train_end_seq]
        y_train_s = y_seq_all[:train_end_seq]
        X_test_s = X_seq_all[train_end_seq:test_end_seq]
        y_test_s = y_seq_all[train_end_seq:test_end_seq]

        # LSTM
        print(f"    LSTM    ... ", end="", flush=True)
        rmse, mae = cv_rnn(X_train_s, y_train_s, X_test_s, y_test_s, "lstm")
        results["LSTM"]["rmse"].append(rmse)
        results["LSTM"]["mae"].append(mae)
        print(f"RMSE={rmse:.4f}  MAE={mae:.4f}")

        # GRU
        print(f"    GRU     ... ", end="", flush=True)
        rmse, mae = cv_rnn(X_train_s, y_train_s, X_test_s, y_test_s, "gru")
        results["GRU"]["rmse"].append(rmse)
        results["GRU"]["mae"].append(mae)
        print(f"RMSE={rmse:.4f}  MAE={mae:.4f}")

    # ──────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("   CROSS-VALIDATION RESULTS")
    print(f"{'=' * 60}")

    summary = {}
    for model_name in models:
        rmse_arr = results[model_name]["rmse"]
        mae_arr = results[model_name]["mae"]
        summary[model_name] = {
            "rmse_per_fold": rmse_arr,
            "mae_per_fold": mae_arr,
            "rmse_mean": float(np.mean(rmse_arr)),
            "rmse_std": float(np.std(rmse_arr)),
            "mae_mean": float(np.mean(mae_arr)),
            "mae_std": float(np.std(mae_arr)),
        }
        print(f"\n  {model_name}:")
        print(f"    RMSE: {np.mean(rmse_arr):.4f} +/- {np.std(rmse_arr):.4f}")
        print(f"    MAE:  {np.mean(mae_arr):.4f} +/- {np.std(mae_arr):.4f}")
        print(f"    Per fold: {[f'{r:.4f}' for r in rmse_arr]}")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "cv_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  [OK] Results saved: {results_path}")

    # ──────────────────────────────────────────────────────
    # Plot: RMSE per fold
    # ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fold_labels = [f"Fold {i+1}" for i in range(len(FOLDS))]
    colors = {"XGBoost": "#fc8d62", "LSTM": "#8da0cb", "GRU": "#e78ac3"}

    for model_name in models:
        ax1.plot(fold_labels, results[model_name]["rmse"], 'o-',
                 label=model_name, color=colors[model_name], linewidth=2)
    ax1.set_ylabel("RMSE")
    ax1.set_title("RMSE per Fold (Expanding Window CV)", fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Bar chart: mean +/- std
    x_pos = np.arange(len(models))
    means = [summary[m]["rmse_mean"] for m in models]
    stds = [summary[m]["rmse_std"] for m in models]
    bars = ax2.bar(x_pos, means, yerr=stds, capsize=8,
                   color=[colors[m] for m in models], alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models)
    ax2.set_ylabel("RMSE")
    ax2.set_title("Mean RMSE (+/- std) Across Folds", fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "cv_results.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  [OK] Plot saved: {plot_path}")

    print("\n  DONE! Cross-validation complete.\n")


if __name__ == "__main__":
    run_cv()
