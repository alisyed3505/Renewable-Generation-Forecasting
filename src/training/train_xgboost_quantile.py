# src/training/train_xgboost_quantile.py
"""
Probabilistic XGBoost — Quantile Regression

Trains 3 XGBoost models to produce prediction intervals:
  - Lower bound (alpha=0.1, 10th percentile)
  - Median      (alpha=0.5, 50th percentile)
  - Upper bound (alpha=0.9, 90th percentile)

Uses XGBoost's built-in quantile regression objective.

Usage:
    python src/training/train_xgboost_quantile.py
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.metrics import save_metrics


# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
DATA_FILE = "data/raw/pv_01.csv"
MODELS_DIR = "models/xgboost_quantile"
METRICS_DIR = "models/metrics"
PLOTS_DIR = "src/evaluation/quantile"

FEATURE_COLS = [
    "hour_of_day_sin", "hour_of_day_cos",
    "month_of_year_sin", "month_of_year_cos",
    "sunposition_thetaZ", "sunposition_solarAzimuth",
    "clearsky_diffuse", "clearsky_direct", "clearsky_global",
    "TemperatureAt0", "RelativeHumidityAt0",
    "SolarRadiationGlobalAt0", "SolarRadiationDirectAt0",
    "SolarRadiationDiffuseAt0", "TotalCloudCoverAt0",
]

TEST_SPLIT = 0.2
RANDOM_SEED = 42

# Use Optuna-optimized hyperparameters from v2
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

# Quantile levels
QUANTILES = {
    "lower": 0.1,
    "median": 0.5,
    "upper": 0.9,
}


# ──────────────────────────────────────────────────────────
# Data loading (standalone)
# ──────────────────────────────────────────────────────────
def load_data():
    """Load and split data for quantile regression."""
    import pandas as pd

    df = pd.read_csv(DATA_FILE, delimiter=";")
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]

    X = df[FEATURE_COLS].ffill().bfill().values.astype(np.float32)
    y = df["power_normed"].fillna(0).values.astype(np.float32)

    split_idx = int(len(X) * (1 - TEST_SPLIT))
    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:],
    )


# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────
def train_quantile_models():
    """Train 3 XGBoost models for lower, median, upper quantiles."""
    import xgboost as xgb

    print("=" * 60)
    print("   XGBOOST QUANTILE REGRESSION")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_data()
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    models = {}
    predictions = {}

    for name, alpha in QUANTILES.items():
        print(f"\n  Training {name} quantile (alpha={alpha})...")

        model = xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            eval_metric="mae",
            random_state=RANDOM_SEED,
            verbosity=0,
            **XGB_PARAMS,
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = np.clip(model.predict(X_test), 0, 1)
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_test - y_pred)))

        print(f"    RMSE: {rmse:.4f}  MAE: {mae:.4f}")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"xgboost_quantile_{name}.json")
        model.save_model(model_path)
        print(f"    Model saved: {model_path}")

        models[name] = model
        predictions[name] = y_pred

    # ──────────────────────────────────────────────────────
    # Coverage and sharpness metrics
    # ──────────────────────────────────────────────────────
    y_lower = predictions["lower"]
    y_upper = predictions["upper"]
    y_median = predictions["median"]

    # Enforce monotonicity: prevent quantile crossing (lower <= median <= upper)
    y_lower = np.minimum(y_lower, y_median)
    y_upper = np.maximum(y_upper, y_median)

    # Coverage: % of actual values within prediction interval
    in_band = (y_test >= y_lower) & (y_test <= y_upper)
    coverage = float(np.mean(in_band) * 100)

    # Average interval width (sharpness)
    interval_width = float(np.mean(y_upper - y_lower))

    # Median metrics
    median_rmse = float(np.sqrt(np.mean((y_test - y_median) ** 2)))
    median_mae = float(np.mean(np.abs(y_test - y_median)))

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Median RMSE:      {median_rmse:.4f}")
    print(f"  Median MAE:       {median_mae:.4f}")
    print(f"  Coverage (10-90): {coverage:.1f}%")
    print(f"  Avg interval:     {interval_width:.4f}")

    # Save metrics
    save_metrics(
        model_name="xgboost_quantile",
        model_version="v1",
        metrics={
            "median_rmse": median_rmse,
            "median_mae": median_mae,
            "coverage_10_90": coverage,
            "avg_interval_width": interval_width,
        },
        output_path=os.path.join(METRICS_DIR, "xgboost_quantile_v1_metrics.json"),
        extra_info={
            "model_type": "xgboost_quantile",
            "scope": "single_site",
            "site": "pv_01",
            "quantiles": list(QUANTILES.values()),
            **XGB_PARAMS,
        },
    )

    return y_test, predictions, coverage, interval_width


# ──────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────
def generate_plots(y_test, predictions):
    """Generate prediction interval visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_lower = predictions["lower"]
    y_median = predictions["median"]
    y_upper = predictions["upper"]

    # Plot 1: Fan chart (first 200 samples)
    n_show = min(200, len(y_test))
    x = np.arange(n_show)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(x, y_lower[:n_show], y_upper[:n_show],
                     alpha=0.3, color="#2196F3", label="10-90% interval")
    ax.plot(x, y_median[:n_show], color="#1565C0", linewidth=1.5,
            label="Median prediction", alpha=0.9)
    ax.plot(x, y_test[:n_show], color="#FF5722", linewidth=1,
            label="Actual", alpha=0.8)
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Normalized Power", fontsize=12)
    ax.set_title("Probabilistic Forecast: Prediction Intervals", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "prediction_intervals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")

    # Plot 2: Coverage analysis
    in_band = (y_test >= y_lower) & (y_test <= y_upper)

    fig, ax = plt.subplots(figsize=(10, 5))
    # Running coverage
    window = 50
    running_coverage = np.convolve(in_band.astype(float), np.ones(window) / window, mode="valid")
    ax.plot(running_coverage * 100, color="#4CAF50", linewidth=1.5)
    ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="Expected: 80%")
    ax.set_xlabel("Time step", fontsize=12)
    ax.set_ylabel("Coverage (%)", fontsize=12)
    ax.set_title(f"Running Coverage (window={window})", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "coverage_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")

    # Plot 3: Interval width distribution
    widths = y_upper - y_lower

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(widths, bins=40, color="#9C27B0", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(widths), color="red", linestyle="--",
               label=f"Mean: {np.mean(widths):.4f}")
    ax.set_xlabel("Interval Width", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Prediction Interval Width Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "interval_width_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    y_test, predictions, coverage, interval_width = train_quantile_models()

    print("\n  Generating plots...")
    generate_plots(y_test, predictions)

    print(f"\n{'=' * 60}")
    print("  PROBABILISTIC XGBOOST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Models:  {MODELS_DIR}/")
    print(f"  Metrics: {METRICS_DIR}/xgboost_quantile_v1_metrics.json")
    print(f"  Plots:   {PLOTS_DIR}/")
