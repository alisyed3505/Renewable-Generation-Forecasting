import os
import pandas as pd
import matplotlib.pyplot as plt


METRICS_FILES = {
    "Naive (24h)": "models/metrics/naive_24h_metrics.txt",
    "LSTM": "models/metrics/baseline_metrics.txt",
    "XGBoost": "models/metrics/xgboost_metrics.txt",
}


def read_metrics(path):
    metrics = {}
    with open(path, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":")
                metrics[key.strip()] = float(value.strip())
    return metrics


def compare_models():
    print("=" * 60)
    print("   MODEL COMPARISON — SINGLE SITE")
    print("=" * 60)

    rows = []

    for model, path in METRICS_FILES.items():
        m = read_metrics(path)
        rows.append({
            "Model": model,
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
        })

    df = pd.DataFrame(rows).sort_values("RMSE")

    # Save table
    os.makedirs("src/evaluation/comparison", exist_ok=True)
    csv_path = "src/evaluation/comparison/comparison.csv"
    df.to_csv(csv_path, index=False)

    print(df.to_string(index=False))
    print(f"\n✅ Comparison table saved to: {csv_path}")

    # Optional plot
    plt.figure(figsize=(6, 4))
    plt.bar(df["Model"], df["RMSE"])
    plt.title("Model Comparison (RMSE)")
    plt.ylabel("RMSE")
    plt.tight_layout()

    plot_path = "src/evaluation/comparison/comparison.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    compare_models()
