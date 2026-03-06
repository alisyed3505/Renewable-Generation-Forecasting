"""
Embedded GRU for Multi-Site Solar Power Forecasting.

Mirrors embedded LSTM v1 architecture but uses GRU layers.
- Learns a site embedding
- Injects site identity into each timestep via RepeatVector + Concatenate
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    RepeatVector,
    Concatenate,
    GRU,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam


def build_embedded_gru(
    num_sites,
    embedding_dim,
    time_steps,
    num_features,
    gru_units_1=64,
    gru_units_2=32,
    dense_units=16,
    dropout_rate=0.2,
    learning_rate=1e-3,
):
    """
    Build embedded GRU model.

    Inputs:
        site_id_input: (batch, 1)
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

    # GRU backbone (same structure as LSTM)
    x = GRU(
        gru_units_1,
        return_sequences=True,
        name="gru_1",
    )(x)
    x = Dropout(dropout_rate)(x)

    x = GRU(
        gru_units_2,
        return_sequences=False,
        name="gru_2",
    )(x)
    x = Dropout(dropout_rate)(x)

    # Dense head
    x = Dense(dense_units, activation="relu", name="dense_1")(x)
    output = Dense(1, activation="relu", name="output")(x)

    model = Model(
        inputs=[site_input, feature_input],
        outputs=output,
        name="embedded_gru",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model
