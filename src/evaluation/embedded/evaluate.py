"""
Embedded LSTM evaluation with per-site analysis.
"""

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.embedded import (
    MODEL_PATH,
    SCALER_PATH,
    DATA_GLOB,
    TIME_STEPS,
    NUM_SITES,
)
from config.baseline import FEATURE_COLS_BASELINE
from src.data.embedded.data_loader import preprocess_embedded_data


def evaluate_embedded():
    print("=" * 60)
    print("   EMBEDDED LSTM EVALUATION")
    print("=" * 60)

    # -------------------------
    # Load model
    # -------------------------
    model = load_model(MODEL_PATH)

    # -------------------------
    # Load data
    # -------------------------
    X_feat, X_site, y = preprocess_embedded_data(
        DATA_GLOB,
        scaler_path=SCALER_PATH,
    )

    # -------------------------
    # Train / test split
    # -------------------------
    split_idx = int(len(y) * 0.8)
    X_feat_test = X_feat[split_idx:]
    X_site_test = X_site[split_idx:]
    y_test = y[split_idx:]

    # -------------------------
    # Predict
    # -------------------------
    y_pred = model.predict([X_site_test, X_feat_test]).flatten()
    y_pred = np.clip(y_pred, 0, 1)

    # -------------------------
    # Overall metrics
    # -------------------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")

    # -------------------------
    # Per-site MAE
    # -------------------------
    site_mae = {}
    for site in range(NUM_SITES):
        mask = X_site_test.flatten() == site
        if mask.sum() > 0:
            site_mae[site] = mean_absolute_error(
                y_test[mask], y_pred[mask]
            )

    return y_test, y_pred, X_site_test.flatten(), rmse, mae, site_mae


if __name__ == "__main__":
    evaluate_embedded()
