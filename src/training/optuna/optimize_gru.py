import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import joblib

import optuna
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.data.baseline.data_loader import load_single_site_csv
from config.baseline import DATA_FILE, TIME_STEPS, TEST_SPLIT


# ============================================================
# Utility: Create sequences
# ============================================================
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# ============================================================
# Optuna Objective Function
# ============================================================
def objective(trial):
    # Hyperparameters to tune
    gru_1_units = trial.suggest_int("gru_1_units", 32, 128)
    gru_2_units = trial.suggest_int("gru_2_units", 16, 64)
    dense_units = trial.suggest_int("dense_units", 8, 32)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Load & preprocess data
    df = load_single_site_csv(DATA_FILE)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    X_seq, y_seq = create_sequences(X, y, TIME_STEPS)

    split = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    # Model definition (GRU instead of LSTM)
    model = Sequential([
        Input(shape=(TIME_STEPS, X.shape[1])),
        GRU(gru_1_units, return_sequences=True),
        Dropout(dropout),
        GRU(gru_2_units),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="mse"
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )

    return min(history.history["val_loss"])


# ============================================================
# Main entry point
# ============================================================
def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\n============================================================")
    print("   OPTUNA GRU OPTIMIZATION COMPLETE")
    print("============================================================")
    print("Best trial:")
    print(study.best_trial)

    # Output directories
    STUDY_DIR = "models/optuna/baseline_gru_v2"
    os.makedirs(STUDY_DIR, exist_ok=True)

    # Save Optuna study
    joblib.dump(study, os.path.join(STUDY_DIR, "study.pkl"))

    # Save best parameters
    with open(os.path.join(STUDY_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    # Export all trials to CSV
    import pandas as pd
    trials_data = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value": trial.value,
            **trial.params
        }
        trials_data.append(row)

    df_trials = pd.DataFrame(trials_data)
    df_trials.to_csv(os.path.join(STUDY_DIR, "trials.csv"), index=False)

    print(f"\n[OK] Optuna study saved to: {STUDY_DIR}")
    print("[OK] Best parameters saved")
    print("[OK] Trial history exported to trials.csv")

    # Print best params for config/gru.py update
    bp = study.best_params
    print("\n--- Copy to config/gru.py v2 block ---")
    print(f'if MODEL_VERSION == "v2":')
    print(f'    GRU_UNITS_1 = {bp["gru_1_units"]}')
    print(f'    GRU_UNITS_2 = {bp["gru_2_units"]}')
    print(f'    DENSE_UNITS = {bp["dense_units"]}')
    print(f'    DROPOUT_RATE = {bp["dropout"]}')
    print(f'    LEARNING_RATE = {bp["lr"]}')
    print(f'    BATCH_SIZE = {bp["batch_size"]}')


if __name__ == "__main__":
    main()
