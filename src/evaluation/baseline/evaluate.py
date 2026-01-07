"""
Baseline LSTM evaluation on held-out test set.
"""
import sys
import os
import numpy as np
import joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config.baseline import (
    MODEL_PATH,
    SCALER_PATH,
    DATA_FILE,
    TIME_STEPS,
    FEATURE_COLS_BASELINE,
)

from src.data.baseline.data_loader import load_single_site_csv, create_sequences


def evaluate_baseline():
    print("=" * 60)
    print("   BASELINE LSTM EVALUATION")
    print("=" * 60)

    # ---------------------------
    # Load model & scaler
    # ---------------------------
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ---------------------------
    # Load data
    # ---------------------------
    df = load_single_site_csv(DATA_FILE)

    X = df[FEATURE_COLS_BASELINE].ffill().bfill().values
    y = df["power_normed"].fillna(0).values

    X_scaled = scaler.transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)

    # ---------------------------
    # Train / test split (same logic as training)
    # ---------------------------
    split_idx = int(len(X_seq) * 0.8)
    X_test = X_seq[split_idx:]
    y_test = y_seq[split_idx:]

    # ---------------------------
    # Predict
    # ---------------------------
    y_pred = model.predict(X_test).flatten()
    y_pred = np.clip(y_pred, 0, 1)

    # ---------------------------
    # Metrics
    # ---------------------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")

    return y_test, y_pred, rmse, mae


if __name__ == "__main__":
    evaluate_baseline()
