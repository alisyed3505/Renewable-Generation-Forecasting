"""
Training script for Embedded LSTM (Multi-Site with Site Embeddings)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# For plots
from src.evaluation.embedded.evaluate import evaluate_embedded
from src.utils.metrics import save_metrics
from src.evaluation.embedded.plots import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution,
    plot_site_mae,
)

from config.embedded import (
    DATA_GLOB,
    NUM_SITES,
    TIME_STEPS,
    TEST_SPLIT,
    VAL_SPLIT,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    EMBEDDING_DIM,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    DENSE_UNITS,
    DROPOUT_RATE,
    MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
)

from src.data.embedded.data_loader import preprocess_embedded_data
from src.models.embedded.lstm_v1 import build_embedded_lstm


def set_seeds(seed: int):
    """Ensure reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_embedded_lstm():
    print("=" * 60)
    print("   EMBEDDED LSTM TRAINING (MULTI-SITE)")
    print("=" * 60)

    set_seeds(RANDOM_SEED)

    # =====================================
    # 1. LOAD & PREPROCESS DATA
    # =====================================
    print("\n📂 Loading and preprocessing multi-site data...")

    X_feat_seq, X_site_seq, y_seq = preprocess_embedded_data(
        DATA_GLOB,
        scaler_path=SCALER_PATH,
    )

    print(f"   Feature sequences shape: {X_feat_seq.shape}")
    print(f"   Site ID shape:           {X_site_seq.shape}")
    print(f"   Targets shape:           {y_seq.shape}")

    # =====================================
    # 2. TRAIN / TEST SPLIT (chronological)
    # =====================================
    split_idx = int(len(y_seq) * (1 - TEST_SPLIT))

    X_feat_train, X_feat_test = (
        X_feat_seq[:split_idx],
        X_feat_seq[split_idx:],
    )
    X_site_train, X_site_test = (
        X_site_seq[:split_idx],
        X_site_seq[split_idx:],
    )
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"\n📊 Train samples: {len(y_train)}")
    print(f"📊 Test samples:  {len(y_test)}")

    # =====================================
    # 3. BUILD MODEL
    # =====================================
    print("\n🏗️ Building embedded LSTM...")

    model = build_embedded_lstm(
        num_sites=NUM_SITES,
        embedding_dim=EMBEDDING_DIM,
        time_steps=TIME_STEPS,
        num_features=X_feat_train.shape[2],
        lstm_units_1=LSTM_UNITS_1,
        lstm_units_2=LSTM_UNITS_2,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )

    model.summary()

    # =====================================
    # 4. TRAIN
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
        [X_site_train, X_feat_train],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    # =====================================
    # 5. SAVE MODEL & METRICS
    # =====================================
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    print("\n📈 Evaluating on test set...")
    y_pred = model.predict([X_site_test, X_feat_test]).flatten()
    y_pred = np.clip(y_pred, 0, 1)

    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    mae = np.mean(np.abs(y_test - y_pred))

    # os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    # with open(METRICS_PATH, "w") as f:
    #     f.write("Embedded LSTM Model Performance\n")
    #     f.write("=" * 40 + "\n")
    #     f.write(f"Train samples: {len(y_train)}\n")
    #     f.write(f"Test samples:  {len(y_test)}\n")
    #     f.write(f"Sites used:    {NUM_SITES}\n\n")
    #     f.write(f"RMSE: {rmse:.6f}\n")
    #     f.write(f"MAE:  {mae:.6f}\n")

    save_metrics(
        model_name="embedded_lstm",
        model_version="v1",
        metrics={
            "rmse": rmse,
            "mae": mae
        },
        output_path=METRICS_PATH,
        extra_info={
            "model_type": "lstm",
            "scope": "multi_site",
            "num_sites": NUM_SITES,
            "embedding_dim": EMBEDDING_DIM
        }
    )


    print("\n" + "=" * 60)
    print("   EMBEDDED LSTM TRAINING COMPLETE")
    print("=" * 60)
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Test MAE:  {mae:.4f}")
    print(f"\n   Model saved to:  {MODEL_PATH}")
    print(f"   Scaler saved to: {SCALER_PATH}")
    print(f"   Metrics saved to: {METRICS_PATH}")

    return model, history, (X_site_test, X_feat_test, y_test)


if __name__ == "__main__":
    # train_embedded_lstm()

    # Temporarily for plot generation
    model, history, _ = train_embedded_lstm()
    
    # Generate plots
    print("\n📊 Generating plots...")
    y_true, y_pred, site_ids, rmse, mae, site_mae = evaluate_embedded()
    
    plot_training_history(history, "src/evaluation/embedded")
    plot_predictions(y_true, y_pred, "src/evaluation/embedded")
    plot_error_distribution(y_true, y_pred, "src/evaluation/embedded")
    plot_site_mae(site_mae, "src/evaluation/embedded")
    
    print("✅ Plots saved to src/evaluation/embedded/")
