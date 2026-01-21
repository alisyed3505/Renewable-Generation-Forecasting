import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from config.baseline import DATA_FILE, TEST_SPLIT
from src.models.xgboost.data import prepare_xgboost_data
from src.utils.metrics import save_metrics

def evaluate_xgboost():
    print("=" * 60)
    print("   XGBOOST EVALUATION (SINGLE SITE)")
    print("=" * 60)

    # --------------------------------------------------
    # Load data (same as training)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = prepare_xgboost_data(
        DATA_FILE,
        TEST_SPLIT,
    )

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------
    model_path = "models/xgboost/xgb_model.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = xgb.XGBRegressor()
    model.load_model(model_path)

    # --------------------------------------------------
    # Predict
    # --------------------------------------------------
    print("\n📈 Running predictions...")
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 1)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics_path = "models/metrics/xgboost_v1_metrics.json"
    # os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # with open(metrics_path, "w") as f:
    #     f.write("XGBoost Model Performance (Single Site)\n")
    #     f.write("=" * 45 + "\n")
    #     f.write(f"Train samples: {len(X_train)}\n")
    #     f.write(f"Test samples:  {len(X_test)}\n\n")
    #     f.write(f"RMSE: {rmse:.6f}\n")
    #     f.write(f"MAE:  {mae:.6f}\n")

    save_metrics(
        model_name="xgboost",
        model_version="v1",
        metrics={
            "rmse": rmse,
            "mae": mae
        },
        output_path=metrics_path,
        extra_info={
            "model_type": "tree",
            "scope": "single_site",
            "site": "pv_01",
            "n_test_samples": len(y_test)
        }
    )


    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plot_dir = "src/evaluation/xgboost"
    os.makedirs(plot_dir, exist_ok=True)

    # Prediction vs Actual
    plt.figure(figsize=(10, 4))
    plt.plot(y_test[:200], label="Actual")
    plt.plot(y_pred[:200], label="XGBoost")
    plt.title("XGBoost — Prediction vs Actual")
    plt.xlabel("Time step (3h)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/prediction_vs_actual.png")
    plt.close()

    # Error distribution
    errors = y_pred - y_test
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50)
    plt.title("XGBoost — Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/error_distribution.png")
    plt.close()

    print(f"\n✅ Metrics saved to: {metrics_path}")
    print(f"✅ Plots saved to: {plot_dir}/")


if __name__ == "__main__":
    evaluate_xgboost()
