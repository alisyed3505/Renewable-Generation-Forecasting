# src/models/baseline/lstm_v2.py
"""
Baseline LSTM v2 - Optuna-Optimized Model

This version uses hyperparameters optimized by Optuna based on 30 trials.
Key differences from v1:
- Increased LSTM layer 1 units: 64 → 78 (+21.9%)
- Decreased LSTM layer 2 units: 32 → 16 (-50%)
- Slightly decreased dense units: 16 → 14 (-12.5%)
- Increased dropout: 0.2 → 0.317 (+58.5%)
- Increased learning rate: 0.001 → 0.00682 (+582%)

Optuna optimization results:
- Best validation loss: achieved through 30 trials
- Optimized for validation set performance
- See models/optuna/baseline_lstm_v2/ for full optimization history
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_baseline_lstm_v2(
    input_shape,
    lstm_units_1=78,      # Optuna best (was 64 in v1)
    lstm_units_2=16,      # Optuna best (was 32 in v1)
    dense_units=14,       # Optuna best (was 16 in v1)
    dropout_rate=0.317,   # Optuna best (was 0.2 in v1)
    learning_rate=0.006817739988871779,  # Optuna best (was 0.001 in v1)
):
    """
    Build and compile baseline LSTM v2 model with Optuna-optimized hyperparameters.

    Parameters:
        input_shape: (time_steps, num_features)
        lstm_units_1: Units in first LSTM layer (default: 78, Optuna optimized)
        lstm_units_2: Units in second LSTM layer (default: 16, Optuna optimized)
        dense_units: Units in dense layer (default: 14, Optuna optimized)
        dropout_rate: Dropout rate (default: 0.317, Optuna optimized)
        learning_rate: Learning rate for Adam optimizer (default: 0.00682, Optuna optimized)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential(
        [
            LSTM(
                lstm_units_1,
                return_sequences=True,
                input_shape=input_shape,
                name="lstm_1_optuna",
            ),
            Dropout(dropout_rate),

            LSTM(
                lstm_units_2,
                return_sequences=False,
                name="lstm_2_optuna",
            ),
            Dropout(dropout_rate),

            Dense(dense_units, activation="relu", name="dense_1_optuna"),
            Dense(1, activation="relu", name="output"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
