import os
import sys
import numpy as np

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.xgboost import (
    DATA_FILE,
    TEST_SPLIT,
    MODEL_PATH,
    METRICS_PATH,
    PLOTS_DIR,
    MODEL_VERSION,
    MAX_DEPTH,
    MIN_CHILD_WEIGHT,
    LEARNING_RATE,
    N_ESTIMATORS,
    GAMMA,
    SUBSAMPLE,
    COLSAMPLE_BYTREE,
    REG_ALPHA,
    REG_LAMBDA,
)
from src.utils.metrics import save_metrics

# Import the correct model version
if MODEL_VERSION == "v1":
    from src.models.xgboost.model_v1 import build_xgboost_model
else:
    from src.models.xgboost.model_v2 import build_xgboost_v2 as build_xgboost_model

from src.models.xgboost.data import prepare_xgboost_data


def main():
    print("=" * 60)
    print("   XGBOOST TRAINING (SINGLE SITE)")
    print("=" * 60)
    print(f"   Training version: {MODEL_VERSION}")
    print(f"   Plots  directory: {PLOTS_DIR}")
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
    print(f"\n🏗️ Building XGBoost {MODEL_VERSION}...")
    
    if MODEL_VERSION == "v1":
        model = build_xgboost_model()
    else:
        model = build_xgboost_model(
            max_depth=MAX_DEPTH,
            min_child_weight=MIN_CHILD_WEIGHT,
            learning_rate=LEARNING_RATE,
            n_estimators=N_ESTIMATORS,
            gamma=GAMMA,
            subsample=SUBSAMPLE,
            colsample_bytree=COLSAMPLE_BYTREE,
            reg_alpha=REG_ALPHA,
            reg_lambda=REG_LAMBDA,
        )
    
    print(f"   Model hyperparameters:")
    print(f"   - max_depth: {MAX_DEPTH}")
    print(f"   - learning_rate: {LEARNING_RATE}")
    print(f"   - n_estimators: {N_ESTIMATORS}")

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
    # Evaluate
    # --------------------------------------------------
    print("\n📈 Evaluating on test set...")
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 1)
    
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    # --------------------------------------------------
    # Save model & metrics
    # --------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    
    save_metrics(
        model_name="xgboost",
        model_version=MODEL_VERSION,
        metrics={
            "rmse": rmse,
            "mae": mae
        },
        output_path=METRICS_PATH,
        extra_info={
            "model_type": "xgboost",
            "scope": "single_site",
            "site": "pv_01",
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "n_estimators": N_ESTIMATORS
        }
    )

    print("\n" + "=" * 60)
    print("✅ XGBoost training complete")
    print("=" * 60)
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Test MAE:  {mae:.4f}")
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Metrics saved to: {METRICS_PATH}")
    print(f"   Plots directory: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
