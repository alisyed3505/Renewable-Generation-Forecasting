import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.data.baseline.data_loader import load_single_site_csv
from config.baseline import DATA_FILE, TEST_SPLIT

def xgboost_feature_importance():
    print("=" * 60)
    print("   XGBOOST FEATURE IMPORTANCE")
    print("=" * 60)

    # --------------------------------------------------
    # Load data to recover feature names
    # --------------------------------------------------
    df = load_single_site_csv(DATA_FILE)

    feature_names = df.columns[:-1].tolist()  # target is last column

    # --------------------------------------------------
    # Load trained XGBoost model
    # --------------------------------------------------
    model_path = "models/xgboost/xgb_model.json"
    booster = xgb.Booster()
    booster.load_model(model_path)

    # --------------------------------------------------
    # Get feature importance (GAIN)
    # --------------------------------------------------
    importance = booster.get_score(importance_type="gain")

    # Map f0, f1, ... → real feature names
    mapped_importance = {
        feature_names[int(k[1:])]: v
        for k, v in importance.items()
    }

    imp_df = (
        pd.DataFrame(
            mapped_importance.items(),
            columns=["feature", "importance"]
        )
        .sort_values("importance", ascending=False)
    )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    os.makedirs("src/evaluation/xgboost", exist_ok=True)
    csv_path = "src/evaluation/xgboost/feature_importance.csv"
    imp_df.to_csv(csv_path, index=False)

    # --------------------------------------------------
    # Plot top 20
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    imp_df.head(20).iloc[::-1].plot(
        x="feature",
        y="importance",
        kind="barh",
        legend=False
    )
    plt.title("XGBoost Feature Importance (Gain)")
    plt.xlabel("Importance")
    plt.tight_layout()

    plot_path = "src/evaluation/xgboost/feature_importance.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"✅ Feature importance saved to: {csv_path}")
    print(f"✅ Plot saved to: {plot_path}")


if __name__ == "__main__":
    xgboost_feature_importance()
