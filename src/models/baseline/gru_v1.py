# src/models/baseline/gru_v1.py
"""
Baseline GRU model for single-site solar power forecasting.

Mirrors the Baseline LSTM v1 architecture but uses GRU layers instead.
GRU has fewer parameters than LSTM (no separate cell state),
which can lead to faster training while maintaining similar performance.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_baseline_gru(
    input_shape,
    gru_units_1=64,
    gru_units_2=32,
    dense_units=16,
    dropout_rate=0.2,
    learning_rate=1e-3,
):
    """
    Build and compile baseline GRU model.

    Parameters:
        input_shape: (time_steps, num_features)
    """
    model = Sequential(
        [
            GRU(
                gru_units_1,
                return_sequences=True,
                input_shape=input_shape,
                name="gru_1",
            ),
            Dropout(dropout_rate),

            GRU(
                gru_units_2,
                return_sequences=False,
                name="gru_2",
            ),
            Dropout(dropout_rate),

            Dense(dense_units, activation="relu", name="dense_1"),
            Dense(1, activation="relu", name="output"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
