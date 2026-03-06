# src/training/train_gru.py
"""
Training script for Baseline GRU (Single-Site).
Mirrors train_baseline_lstm.py but uses GRU architecture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.metrics import save_metrics
from config.gru import (
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
    PLOTS_DIR,
    MODEL_VERSION,
    GRU_UNITS_1,
    GRU_UNITS_2,
    DENSE_UNITS,
    DROPOUT_RATE,
)

from src.data.baseline.data_loader import (
    load_single_site_csv,
    preprocess_baseline_data,
)

from src.models.baseline.gru_v1 import build_baseline_gru


def set_seeds(seed: int):
    """Ensure reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_baseline_gru():
    print("=" * 60)
    print("   BASELINE GRU TRAINING (SINGLE SITE)")
    print("=" * 60)
    print(f"   Training version: {MODEL_VERSION}")
    print(f"   Plots directory: {PLOTS_DIR}")
    print("=" * 60)

    set_seeds(RANDOM_SEED)

    # =====================================
    # 1. LOAD DATA
    # =====================================
    print("\n[+] Loading baseline data...")
    df = load_single_site_csv(DATA_FILE)
    print(f"   Loaded {len(df)} rows from {DATA_FILE}")

    # =====================================
    # 2. PREPROCESS
    # =====================================
    print("\n[+] Preprocessing data...")
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

    print(f"\n   Train samples: {len(y_train)}")
    print(f"   Test samples:  {len(y_test)}")

    # =====================================
    # 4. BUILD MODEL
    # =====================================
    print("\n[+] Building baseline GRU...")
    model = build_baseline_gru(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        gru_units_1=GRU_UNITS_1,
        gru_units_2=GRU_UNITS_2,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )

    model.summary()

    # =====================================
    # 5. TRAIN
    # =====================================
    print("\n[+] Starting training...")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

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
    # 6. SAVE MODEL & METRICS
    # =====================================
    model.save(MODEL_PATH)

    print("\n[+] Evaluating on test set...")

    y_pred = model.predict(X_test).flatten()
    y_pred = np.clip(y_pred, 0, 1)

    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    save_metrics(
        model_name="baseline_gru",
        model_version=MODEL_VERSION,
        metrics={
            "rmse": rmse,
            "mae": mae
        },
        output_path=METRICS_PATH,
        extra_info={
            "model_type": "gru",
            "scope": "single_site",
            "site": "pv_01",
            "time_steps": TIME_STEPS,
            "epochs_trained": len(history.history["loss"]),
            "gru_units_1": GRU_UNITS_1,
            "gru_units_2": GRU_UNITS_2,
            "dense_units": DENSE_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
        }
    )

    print("\n" + "=" * 60)
    print("   BASELINE GRU TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Test MAE:  {mae:.4f}")
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Scaler saved to: {SCALER_PATH}")
    print(f"   Metrics saved to: {METRICS_PATH}")

    return model, history, (X_test, y_test)


if __name__ == "__main__":
    train_baseline_gru()
