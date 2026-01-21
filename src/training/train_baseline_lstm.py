# src/training/train_baseline_lstm.py
"""
Training script for Baseline LSTM (Single-Site).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# for plots
from src.evaluation.baseline.plots import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution,
)
from src.evaluation.baseline.evaluate import evaluate_baseline
from src.utils.metrics import save_metrics
from config.baseline import (
    DATA_FILE,
    TIME_STEPS,
    TEST_SPLIT,
    VAL_SPLIT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    PLOTS_DIR,  # NEW: Version-specific plots directory
    MODEL_VERSION,  # NEW: Auto-detected version
    LSTM_UNITS_1,  # NEW: For model building
    LSTM_UNITS_2,  # NEW: For model building
    DENSE_UNITS,   # NEW: For model building
    DROPOUT_RATE,  # NEW: For model building
)

from src.data.baseline.data_loader import (
    load_single_site_csv,
    preprocess_baseline_data,
)

# Import the correct model version based on MODEL_VERSION
if MODEL_VERSION == "v1":
    from src.models.baseline.lstm_v1 import build_baseline_lstm
else:
    # v2 and beyond use the versioned model files
    from src.models.baseline.lstm_v2 import build_baseline_lstm_v2 as build_baseline_lstm


def set_seeds(seed: int):
    """Ensure reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_baseline_lstm():
    print("=" * 60)
    print("   BASELINE LSTM TRAINING (SINGLE SITE)")
    print("=" * 60)
    print(f"   Training version: {MODEL_VERSION}")
    print(f"   Plots directory: {PLOTS_DIR}")
    print("=" * 60)

    set_seeds(RANDOM_SEED)

    # =====================================
    # 1. LOAD DATA
    # =====================================
    print("\n📂 Loading baseline data...")
    df = load_single_site_csv(DATA_FILE)
    print(f"   Loaded {len(df)} rows from {DATA_FILE}")

    # =====================================
    # 2. PREPROCESS
    # =====================================
    print("\n🔧 Preprocessing data...")
    X_seq, y_seq, scaler = preprocess_baseline_data(
        df,
        scaler_path=SCALER_PATH,
        time_steps=TIME_STEPS,
    )

    print(f"   X shape: {X_seq.shape}")
    print(f"   y shape: {y_seq.shape}")

    # =====================================
    # 3. TRAIN / TEST SPLIT
    # =====================================
    split_idx = int(len(X_seq) * (1 - TEST_SPLIT))

    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"\n📊 Train samples: {len(y_train)}")
    print(f"📊 Test samples:  {len(y_test)}")

    # =====================================
    # 4. BUILD MODEL
    # =====================================
    print("\n🏗️ Building baseline LSTM...")
    model = build_baseline_lstm(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        learning_rate=LEARNING_RATE,
    )

    model.summary()

    # =====================================
    # 5. TRAIN
    # =====================================
    print("\n🚀 Starting training...")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # =====================================
    # 6. SAVE FINAL MODEL & METRICS
    # =====================================
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    print("\n📈 Evaluating on test set...")

    y_pred = model.predict(X_test).flatten()
    y_pred = np.clip(y_pred, 0, 1)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    # os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    # with open(METRICS_PATH, "w") as f:
    #     f.write("Baseline LSTM Model Performance\n")
    #     f.write("=" * 40 + "\n")
    #     f.write(f"Train samples: {len(X_train)}\n")
    #     f.write(f"Test samples:  {len(X_test)}\n\n")
    #     f.write(f"RMSE: {rmse:.6f}\n")
    #     f.write(f"MAE:  {mae:.6f}\n")

    save_metrics(
        model_name="baseline_lstm",
        model_version=MODEL_VERSION,  # Use auto-detected version
        metrics={
            "rmse": rmse,
            "mae": mae
        },
        output_path=METRICS_PATH,
        extra_info={
            "model_type": "lstm",
            "scope": "single_site",
            "site": "pv_01",
            "time_steps": TIME_STEPS,
            "epochs_trained": len(history.history["loss"])
        }
    )
    
    print("\n" + "=" * 60)
    print("\n✅ Baseline model training complete")
    print("=" * 60)
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Test MAE:  {mae:.4f}")
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Scaler saved to: {SCALER_PATH}")
    print(f"   Metrics saved to: {METRICS_PATH}")
    print(f"   Plots will be saved to: {PLOTS_DIR}")

    return model, history, (X_test, y_test)


if __name__ == "__main__":
    # train_baseline_lstm()

    # temporarily the following code for plots generation
    _, history, _ = train_baseline_lstm()
    
    # Generate plots
    print("\n📊 Generating plots...")
    y_true, y_pred, _, _ = evaluate_baseline()
    
    plot_training_history(history, PLOTS_DIR)
    plot_predictions(y_true, y_pred, PLOTS_DIR)
    plot_error_distribution(y_true, y_pred, PLOTS_DIR)
    
    print(f"✅ Plots saved to {PLOTS_DIR}")
