# src/models/baseline/lstm_v2.py
"""
Baseline LSTM v2 - Optuna-Optimized Model

This version uses hyperparameters optimized by Optuna based on 50 trials.
Key differences from v1:
- Increased LSTM layer 1 units: 64 -> 85 (+26%)
- Increased LSTM layer 2 units: 32 -> 51 (+59%)
- Increased dense layer units: 16 -> 17 (+6.25%)
- Increased dropout rate: 0.2 -> 0.391 (+95.5%)
- Increased learning rate: 0.0001 -> 0.000172 (+72%)
- Increased batch size: 32 -> 128 (+281.25%)

Optuna optimization results (Trial 13):
- Best Validation Loss: achieved through 50 trails
- Optimized for validation set performance
- See Optuna logs for detailed hyperparameter values
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


def build_baseline_lstm_v2(
    input_shape,  # (time_steps, num_features) tuple
    lstm_units_1=85,  # Optuna optimized (was 64 in v1)
    lstm_units_2=51,  # Optuna optimized (was 32 in v1)
    dense_units=17,  # Optuna optimized (was 16 in v1)
    dropout_rate=0.391,  # Optuna optimized (was 0.2 in v1)
    learning_rate=0.000172,  # Optuna optimized (was 0.0001 in v1)
):
    """
    Build and compile baseline LSTM v2 model with Optuna-optimized hyperparameters (50 trials).

    Parameters:
        input_shape: Tuple of (time_steps, num_features)
        lstm_units_1: Units in first LSTM layer (default: 85, Optuna optimized)
        lstm_units_2: Units in second LSTM layer (default: 51, Optuna optimized)
        dense_units: Units in dense layer (default: 17, Optuna optimized)
        dropout_rate: Dropout rate (default: 0.391, Optuna optimized)
        learning_rate: Learning rate for Adam optimizer (default: 0.000172, Optuna optimized)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(
                lstm_units_1,
                return_sequences=True,
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
