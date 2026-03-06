# src/evaluation/time_of_day_analysis.py
"""
Time-of-Day Error Analysis

Compares model errors across different times of day and stations.
Professor requirement: "Plot the errors as a comparison between the stations
and for different times of day"

Generates:
  1. Grouped bar chart: MAE by time-of-day for each model
  2. Heatmap: MAE per site x time-slot
  3. Line plot: RMSE by time-of-day for each model
"""

import sys
import os

# Import xgboost before adding project root to sys.path, because
# src/evaluation/xgboost/ directory shadows the real xgboost package.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_eval_dir = _script_dir  # src/evaluation/
# Temporarily remove paths that contain the shadowing directory
_removed = [p for p in sys.path if _eval_dir in os.path.abspath(p)]
for p in _removed:
    sys.path.remove(p)
# Also clear cached module if previously imported wrong
if "xgboost" in sys.modules:
    del sys.modules["xgboost"]
import xgboost as _xgb
# Restore paths
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
DATA_DIR = "data/raw"
OUTPUT_DIR = "src/evaluation/time_of_day"
TIME_STEPS = 8
TEST_SPLIT = 0.2
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

# XGBoost params (v2 Optuna)
XGB_PARAMS = {
    "max_depth": 9, "min_child_weight": 6,
    "learning_rate": 0.029710, "n_estimators": 130,
    "gamma": 0.007025, "subsample": 0.845905,
    "colsample_bytree": 0.879990,
    "reg_alpha": 3.353364, "reg_lambda": 2.981327,
}


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def load_site(site_name):
    """Load a single site CSV."""
    path = os.path.join(DATA_DIR, f"{site_name}.csv")
    df = pd.read_csv(path, delimiter=";")
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]
    if "time_idx" in df.columns:
        df = df.sort_values("time_idx")
    return df


def get_hours(df, offset):
    """Extract hour-of-day for test samples, accounting for sequence offset."""
    # Reconstruct hour from sin/cos encoding
    sin_h = df["hour_of_day_sin"].values
    cos_h = df["hour_of_day_cos"].values
    hours = np.round(np.arctan2(sin_h, cos_h) * 24 / (2 * np.pi)) % 24
    hours = hours.astype(int)
    # Offset for sequence models (they predict at time_idx + TIME_STEPS)
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    return hours[split_idx + offset:]


def get_naive_predictions(df):
    """Get naive 24h predictions and actuals on test set."""
    y = df["power_normed"].fillna(0).values.astype(np.float32)
    split_idx = int(len(y) * (1 - TEST_SPLIT))
    y_test = y[split_idx:]
    y_true = y_test[TIME_STEPS:]
    y_pred = y_test[:-TIME_STEPS]
    return y_true, y_pred


def get_xgb_predictions(df):
    """Train XGBoost and return test predictions."""
    XGBRegressor = _xgb.XGBRegressor

    X = df[FEATURE_COLS].ffill().bfill().values.astype(np.float32)
    y = df["power_normed"].fillna(0).values.astype(np.float32)
    split_idx = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = XGBRegressor(
        objective="reg:squarederror", random_state=RANDOM_SEED,
        verbosity=0, **XGB_PARAMS,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = np.clip(model.predict(X_test), 0, 1)
    return y_test, y_pred


def get_rnn_predictions(df, model_type="lstm"):
    """Train LSTM or GRU and return test predictions."""
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

    X = df[FEATURE_COLS].ffill().bfill()
    y = df["power_normed"].fillna(0).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - TIME_STEPS):
        X_seq.append(X_scaled[i:i + TIME_STEPS])
        y_seq.append(y[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    split_idx = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    model = Sequential([
        RNNLayer(64, return_sequences=True,
                 input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        RNNLayer(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="relu"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    model.fit(
        X_train, y_train, epochs=50, batch_size=32,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=0,
    )

    y_pred = np.clip(model.predict(X_test, verbose=0).flatten(), 0, 1)

    del model
    tf.keras.backend.clear_session()

    return y_test, y_pred


# ──────────────────────────────────────────────────────────
# Main Analysis
# ──────────────────────────────────────────────────────────
def run_analysis(sites=None, skip_rnn=False):
    """Run time-of-day error analysis across sites and models."""
    if sites is None:
        sites = [f"pv_{i:02d}" for i in range(1, 22)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect errors by (model, hour)
    models = ["Naive 24h", "XGBoost"]
    if not skip_rnn:
        models.extend(["LSTM", "GRU"])

    # Storage: {model: {hour: [errors across all sites]}}
    errors_by_hour = {m: {h: [] for h in range(0, 24, 3)} for m in models}
    # Storage for heatmap: {model: {site: {hour: mae}}}
    errors_by_site_hour = {m: {} for m in models}

    for idx, site in enumerate(sites):
        print(f"  [{idx+1}/{len(sites)}] Processing {site} ...", end=" ", flush=True)

        df = load_site(site)

        # Naive
        y_true_n, y_pred_n = get_naive_predictions(df)
        hours_n = get_hours(df, TIME_STEPS)
        min_len = min(len(y_true_n), len(hours_n))
        y_true_n, y_pred_n, hours_n = y_true_n[:min_len], y_pred_n[:min_len], hours_n[:min_len]

        # XGBoost
        y_true_x, y_pred_x = get_xgb_predictions(df)
        hours_x = get_hours(df, 0)
        min_len = min(len(y_true_x), len(hours_x))
        y_true_x, y_pred_x, hours_x = y_true_x[:min_len], y_pred_x[:min_len], hours_x[:min_len]

        preds = {
            "Naive 24h": (y_true_n, y_pred_n, hours_n),
            "XGBoost": (y_true_x, y_pred_x, hours_x),
        }

        # LSTM & GRU
        if not skip_rnn:
            for rnn, label in [("lstm", "LSTM"), ("gru", "GRU")]:
                y_true_r, y_pred_r = get_rnn_predictions(df, rnn)
                hours_r = get_hours(df, TIME_STEPS)
                min_len = min(len(y_true_r), len(hours_r))
                y_true_r, y_pred_r, hours_r = y_true_r[:min_len], y_pred_r[:min_len], hours_r[:min_len]
                preds[label] = (y_true_r, y_pred_r, hours_r)

        # Aggregate errors by hour
        for model_name, (yt, yp, hrs) in preds.items():
            abs_errors = np.abs(yt - yp)
            site_hour_mae = {}
            for h in range(0, 24, 3):
                mask = (hrs == h)
                if mask.any():
                    errors_by_hour[model_name][h].extend(abs_errors[mask].tolist())
                    site_hour_mae[h] = float(np.mean(abs_errors[mask]))
                else:
                    site_hour_mae[h] = 0.0
            errors_by_site_hour[model_name][site] = site_hour_mae

        print("done")

    # ──────────────────────────────────────────────────────
    # Plot 1: MAE by Time-of-Day (grouped bar chart)
    # ──────────────────────────────────────────────────────
    hours = list(range(0, 24, 3))
    hour_labels = [f"{h:02d}:00" for h in hours]

    fig, ax = plt.subplots(figsize=(14, 6))
    n_models = len(models)
    bar_width = 0.8 / n_models
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]

    for i, model_name in enumerate(models):
        mae_vals = [np.mean(errors_by_hour[model_name][h]) if errors_by_hour[model_name][h] else 0
                    for h in hours]
        positions = np.arange(len(hours)) + i * bar_width
        ax.bar(positions, mae_vals, bar_width, label=model_name, color=colors[i % len(colors)])

    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax.set_title("Model Error by Time of Day (All Sites)", fontsize=14, fontweight='bold')
    ax.set_xticks(np.arange(len(hours)) + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(hour_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "mae_by_time_of_day.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path1}")

    # ──────────────────────────────────────────────────────
    # Plot 2: RMSE by Time-of-Day (line plot)
    # ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, model_name in enumerate(models):
        rmse_vals = [
            np.sqrt(np.mean(np.array(errors_by_hour[model_name][h])**2))
            if errors_by_hour[model_name][h] else 0
            for h in hours
        ]
        ax.plot(hour_labels, rmse_vals, 'o-', label=model_name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Time of Day", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Model RMSE by Time of Day (All Sites)", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "rmse_by_time_of_day.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path2}")

    # ──────────────────────────────────────────────────────
    # Plot 3: Heatmap (site x time-of-day) for best model
    # ──────────────────────────────────────────────────────
    for model_name in models:
        fig, ax = plt.subplots(figsize=(12, 8))
        site_names = list(errors_by_site_hour[model_name].keys())
        heatmap_data = np.array([
            [errors_by_site_hour[model_name][s].get(h, 0) for h in hours]
            for s in site_names
        ])

        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels(hour_labels)
        ax.set_yticks(range(len(site_names)))
        ax.set_yticklabels(site_names, fontsize=8)
        ax.set_xlabel("Time of Day")
        ax.set_ylabel("Site")
        ax.set_title(f"MAE Heatmap: {model_name} (Site x Time-of-Day)", fontweight='bold')
        plt.colorbar(im, ax=ax, label="MAE")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"heatmap_{safe_name}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  [OK] Saved: {path}")

    # ──────────────────────────────────────────────────────
    # Save raw data
    # ──────────────────────────────────────────────────────
    summary = {}
    for model_name in models:
        summary[model_name] = {
            "mae_by_hour": {
                str(h): float(np.mean(errors_by_hour[model_name][h]))
                if errors_by_hour[model_name][h] else 0
                for h in hours
            },
            "rmse_by_hour": {
                str(h): float(np.sqrt(np.mean(np.array(errors_by_hour[model_name][h])**2)))
                if errors_by_hour[model_name][h] else 0
                for h in hours
            },
        }

    summary_path = os.path.join(OUTPUT_DIR, "time_of_day_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [OK] Saved: {summary_path}")

    print("\n  DONE! Time-of-day analysis complete.\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", type=int, help="Site numbers")
    parser.add_argument("--skip-rnn", action="store_true", help="Skip LSTM/GRU (fast)")
    args = parser.parse_args()

    sites = [f"pv_{s:02d}" for s in args.sites] if args.sites else None

    print("=" * 60)
    print("   TIME-OF-DAY ERROR ANALYSIS")
    print("=" * 60)

    run_analysis(sites=sites, skip_rnn=args.skip_rnn)
