import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config.baseline import DATA_FILE, TEST_SPLIT, TIME_STEPS
from src.data.baseline.data_loader import load_single_site_csv
from src.evaluation.naive.naive_24h import naive_24h_predict
from src.utils.metrics import save_metrics


def evaluate_naive_24h():
    print("=" * 60)
    print("   24h-AGO NAÏVE BASELINE EVALUATION")
    print("=" * 60)

    # --------------------------------------------------
    # Load data using SAME logic as baseline
    # --------------------------------------------------
    df = load_single_site_csv(DATA_FILE)

    # Target = last column (same assumption as baseline)
    y = df.iloc[:, -1].values.astype(np.float32)

    # --------------------------------------------------
    # Train / test split (same as baseline)
    # --------------------------------------------------
    split_idx = int(len(y) * (1 - TEST_SPLIT))
    y_test = y[split_idx:]

    # --------------------------------------------------
    # Naive 24h prediction: y(t) = y(t - 8)
    # --------------------------------------------------
    y_true, y_pred = naive_24h_predict(y_test, TIME_STEPS)

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")

    # --------------------------------------------------
    # Save metrics
    # --------------------------------------------------
    metrics_path = "models/metrics/naive_24h_v1_metrics.json"
    # os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # with open(metrics_path, "w") as f:
    #     f.write("24h-ago Naive Baseline Performance\n")
    #     f.write("=" * 40 + "\n")
    #     f.write(f"Test samples: {len(y_true)}\n\n")
    #     f.write(f"RMSE: {rmse:.6f}\n")
    #     f.write(f"MAE:  {mae:.6f}\n")

    save_metrics(
        model_name="naive_24h",
        model_version="v1",
        metrics={
            "rmse": rmse,
            "mae": mae
        },
        output_path="models/metrics/naive_24h_v1_metrics.json",
        extra_info={
            "model_type": "naive",
            "scope": "single_site",
            "site": "pv_01",
            "time_steps": TIME_STEPS,
            "n_test_samples": len(y_true)
        }
    )

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------
    plot_dir = "src/evaluation/naive"
    os.makedirs(plot_dir, exist_ok=True)

    # Prediction vs Actual
    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:200], label="Actual")
    plt.plot(y_pred[:200], label="Naive (24h ago)")
    plt.title("Naive 24h Baseline: Prediction vs Actual")
    plt.xlabel("Time step (3h)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/prediction_vs_actual.png")
    plt.close()

    # Error distribution
    errors = y_pred - y_true
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=50)
    plt.title("Naive 24h Baseline Error Distribution")
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/error_distribution.png")
    plt.close()

    print(f"✅ Metrics saved to: {metrics_path}")
    print(f"✅ Plots saved to: {plot_dir}/")


if __name__ == "__main__":
    evaluate_naive_24h()
