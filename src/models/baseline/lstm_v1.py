# src/models/baseline/lstm.py
"""
Baseline LSTM model for single-site solar power forecasting.

This model:
- Assumes ONE PV site
- Uses ONLY continuous weather/time features
- Does NOT use site_id
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_baseline_lstm(
    input_shape,
    lstm_units_1=64,
    lstm_units_2=32,
    dense_units=16,
    dropout_rate=0.2,
    learning_rate=1e-3,
):
    """
    Build and compile baseline LSTM model.

    Parameters:
        input_shape: (time_steps, num_features)
    """
    model = Sequential(
        [
            LSTM(
                lstm_units_1,
                return_sequences=True,
                input_shape=input_shape,
                name="lstm_1",
            ),
            Dropout(dropout_rate),

            LSTM(
                lstm_units_2,
                return_sequences=False,
                name="lstm_2",
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
