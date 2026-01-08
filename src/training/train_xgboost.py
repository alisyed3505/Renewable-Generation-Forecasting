import os
import sys
import numpy as np

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.baseline import DATA_FILE, TEST_SPLIT
from src.models.xgboost.model import build_xgboost_model
from src.models.xgboost.data import prepare_xgboost_data


def main():
    print("=" * 60)
    print("   XGBOOST TRAINING (SINGLE SITE)")
    print("=" * 60)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = prepare_xgboost_data(
        DATA_FILE,
        TEST_SPLIT,
    )

    print(f"   X_train shape: {X_train.shape}")
    print(f"   X_test shape:  {X_test.shape}")

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    print("\n🏗️ Building XGBoost model...")
    model = build_xgboost_model()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    print("\n🚀 Training XGBoost...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    model_dir = "models/xgboost"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "xgb_model.json")
    model.save_model(model_path)

    print("\n✅ XGBoost training complete")
    print(f"   Model saved to: {model_path}")


if __name__ == "__main__":
    main()
