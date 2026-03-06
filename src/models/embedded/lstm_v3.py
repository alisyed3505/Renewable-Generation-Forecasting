# src/models/embedded/lstm_v3.py
"""
Embedded LSTM v3 — Leave-One-Site-Out Architecture

Same architecture as v1 (fair generalization test, not Optuna-tuned).
Used for cross-site generalization experiments:
  - Train on 20 sites
  - Evaluate on 1 unseen site

Key differences from v1/v2:
  - num_sites parameter can vary (20 instead of 21 during leave-one-out)
  - Otherwise identical architecture to v1 for fair comparison
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
    embedding_dim,
    time_steps,
    num_features,
    lstm_units_1=64,
    lstm_units_2=32,
    dense_units=16,
    dropout_rate=0.2,
    learning_rate=1e-3,
):
    """
    Build embedded LSTM model for leave-one-site-out experiment.

    Architecture identical to v1 for fair generalization comparison.

    Inputs:
        site_id_input: (batch, 1)  — site index in the training subset
        feature_input: (batch, time_steps, num_features)
    """

    # Inputs
    site_input = Input(shape=(1,), name="site_id")
    feature_input = Input(
        shape=(time_steps, num_features),
        name="features",
    )

    # Site embedding
    site_embedding = Embedding(
        input_dim=num_sites,
        output_dim=embedding_dim,
        name="site_embedding",
    )(site_input)

    # Shape: (batch, 1, embedding_dim) -> (batch, time_steps, embedding_dim)
    site_embedding = RepeatVector(time_steps)(site_embedding[:, 0, :])

    # Concatenate features + site context
    x = Concatenate(axis=-1)(
        [feature_input, site_embedding]
    )

    # LSTM backbone (same as v1)
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

    # Dense head
    x = Dense(dense_units, activation="relu", name="dense_1")(x)
    output = Dense(1, activation="relu", name="output")(x)

    model = Model(
        inputs=[site_input, feature_input],
        outputs=output,
        name="embedded_lstm_v3",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
