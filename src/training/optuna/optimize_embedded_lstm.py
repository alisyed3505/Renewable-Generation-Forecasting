import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import joblib
from datetime import datetime

import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.data.embedded.data_loader import preprocess_embedded_data
from config.embedded import DATA_GLOB, NUM_SITES, TIME_STEPS, TEST_SPLIT


# ============================================================
# Optuna Objective Function
# ============================================================
def objective(trial):
    """
    Optuna objective function for Embedded LSTM hyperparameter tuning.
    
    Optimizes:
    - LSTM architecture: lstm_units_1, lstm_units_2, dense_units
    - Regularization: dropout_rate
    - Learning: learning_rate
    - Embedding: embedding_dim
    """
    # -----------------------------
    # Hyperparameters to tune
    # -----------------------------
    embedding_dim = trial.suggest_int('embedding_dim', 2, 8)
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 128)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 64)
    dense_units = trial.suggest_int('dense_units', 8, 32)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # -----------------------------
    # Load & preprocess data
    # -----------------------------
    # Use temporary scaler path for optimization (won't be saved permanently)
    temp_scaler_path = "models/temp_scaler_optuna.pkl"
    
    X_feat_seq, X_site_seq, y_seq = preprocess_embedded_data(
        DATA_GLOB,
        scaler_path=temp_scaler_path,
    )
    
    # Train/val split
    split_idx = int(len(y_seq) * (1 - TEST_SPLIT))
    X_feat_train, X_feat_val = X_feat_seq[:split_idx], X_feat_seq[split_idx:]
    X_site_train, X_site_val = X_site_seq[:split_idx], X_site_seq[split_idx:]
    y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]

    # -----------------------------
    # Build model
    # -----------------------------
    # Site embedding input - site_id needs to be repeated for each timestep
    site_input = Input(shape=(TIME_STEPS,), name='site_input', dtype='int32')
    site_embedding = Embedding(
        input_dim=NUM_SITES,
        output_dim=embedding_dim,
        name='site_embedding'
    )(site_input)
    
    # Feature input
    feature_input = Input(shape=(TIME_STEPS, X_feat_train.shape[2]), name='feature_input')
    
    # Concatenate site embeddings with features along feature dimension
    combined = Concatenate(axis=2)([site_embedding, feature_input])
    
    # LSTM layers
    x = LSTM(lstm_units_1, return_sequences=True)(combined)
    x = Dropout(dropout)(x)
    x = LSTM(lstm_units_2)(x)
    x = Dropout(dropout)(x)
    x = Dense(dense_units, activation='relu')(x)
    output = Dense(1, activation='relu')(x)
    
    model = tf.keras.Model(inputs=[site_input, feature_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse'
    )
    
    # Prepare site_id data - repeat site_id for each timestep
    # X_site is currently (samples, 1), need it to be (samples, TIME_STEPS)
    X_site_train_expanded = np.repeat(X_site_train, TIME_STEPS, axis=1)
    X_site_val_expanded = np.repeat(X_site_val, TIME_STEPS, axis=1)

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        [X_site_train_expanded, X_feat_train],
        y_train,
        validation_data=([X_site_val_expanded, X_feat_val], y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    return min(history.history['val_loss'])


# ============================================================
# Main entry point
# ============================================================
def main():
    print("=" * 60)
    print("   EMBEDDED LSTM OPTUNA OPTIMIZATION")
    print("=" * 60)
    
    # --------------------------------------------------
    # Create Optuna study
    # --------------------------------------------------
    study = optuna.create_study(direction="minimize")
    
    print("\n🔍 Starting hyperparameter optimization...")
    print(f"   Number of trials: 50")
    print(f"   Objective: Minimize validation loss")
    print(f"   Model: Multi-site Embedded LSTM")
    
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("   OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Best trial (Trial {study.best_trial.number}):")
    print(f"   Validation Loss: {study.best_value:.6f}")
    print(f"\n   Best hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.6f}")
        else:
            print(f"   - {key}: {value}")

    # --------------------------------------------------
    # Output directories
    # --------------------------------------------------
    STUDY_DIR = "models/optuna/embedded_lstm_v2"
    os.makedirs(STUDY_DIR, exist_ok=True)
    os.makedirs(f"{STUDY_DIR}/plots", exist_ok=True)

    # --------------------------------------------------
    # Save Optuna study (FULL history)
    # --------------------------------------------------
    joblib.dump(study, os.path.join(STUDY_DIR, "study.pkl"))

    # --------------------------------------------------
    # Save best parameters
    # --------------------------------------------------
    with open(os.path.join(STUDY_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    # --------------------------------------------------
    # Export all trials to CSV (for analysis & plots)
    # --------------------------------------------------
    trials_data = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value": trial.value,
            **trial.params
        }
        trials_data.append(row)

    import pandas as pd
    df_trials = pd.DataFrame(trials_data)
    df_trials.to_csv(os.path.join(STUDY_DIR, "trials.csv"), index=False)

    print(f"\n✅ Optuna study saved to: {STUDY_DIR}")
    print("✅ Best parameters saved to: best_params.json")
    print("✅ Trial history exported to: trials.csv")
    
    print("\n📈 To generate plots, run:")
    print("   python src/training/optuna/plot_optuna_history_embedded.py")


if __name__ == "__main__":
    main()
