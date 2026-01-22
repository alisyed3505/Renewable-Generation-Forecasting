"""
Embedded LSTM v2 for Multi-Site Solar Power Forecasting.

Optimized with Optuna (Trial 26):
- Validation Loss: 0.006052
- embedding_dim: 2
- lstm_units_1: 85
- lstm_units_2: 33
- dense_units: 27
- dropout: 0.243373
- lr: 0.002073
- batch_size: 64
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    RepeatVector,
    Concatenate,
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


def build_embedded_lstm(
    num_sites,
    embedding_dim=2,  # Optuna optimized
    time_steps=8,
    num_features=15,
    lstm_units_1=85,  # Optuna optimized
    lstm_units_2=33,  # Optuna optimized
    dense_units=27,  # Optuna optimized
    dropout_rate=0.243373,  # Optuna optimized
    learning_rate=0.002073,  # Optuna optimized
):
    """
    Build embedded LSTM model with Optuna-optimized hyperparameters.

    Inputs:
        site_id_input: (batch, 1)
        feature_input: (batch, time_steps, num_features)
    """

    # -----------------------
    # Inputs
    # -----------------------
    site_input = Input(shape=(1,), dtype='int32', name="site_id")
    feature_input = Input(
        shape=(time_steps, num_features),
        name="features",
    )

    # -----------------------
    # Site embedding
    # -----------------------
    site_embedding = Embedding(
        input_dim=num_sites,
        output_dim=embedding_dim,
        name="site_embedding",
    )(site_input)

    # Shape: (batch, 1, embedding_dim)
    site_embedding = RepeatVector(time_steps)(site_embedding[:, 0, :])
    # Shape: (batch, time_steps, embedding_dim)

    # -----------------------
    # Concatenate features + site context
    # -----------------------
    x = Concatenate(axis=-1)(
        [feature_input, site_embedding]
    )

    # -----------------------
    # LSTM backbone
    # -----------------------
    x = LSTM(
        lstm_units_1,
        return_sequences=True,
        name="lstm_1",
    )(x)
    x = Dropout(dropout_rate)(x)

    x = LSTM(
        lstm_units_2,
        return_sequences=False,
        name="lstm_2",
    )(x)
    x = Dropout(dropout_rate)(x)

    # -----------------------
    # Dense head
    # -----------------------
    x = Dense(dense_units, activation="relu", name="dense_1")(x)
    output = Dense(1, activation="relu", name="output")(x)

    model = Model(
        inputs=[site_input, feature_input],
        outputs=output,
        name="embedded_lstm_v2",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
