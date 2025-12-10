# current sequential LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape):
    """
    Builds a robust LSTM model for time series forecasting.

    parameters:
        input_shape: Shape of the input data (e.g., (24, 15))
    returns:
        model: Compiled LSTM model

    structure:
        1. LSTM (The Sequence Reader)
        2. LSTM (The Decision Maker)
        3. Dense (Interpretation)
        4. Output Layer (The Prediction)

        working details:
        Sequential means the layers are stacked one after another.
        for example: Input → LSTM → LSTM → Dense → Output

    """
    model = Sequential([
        # Layer 1: LSTM (The Sequence Reader)
        # Units=64: Each unit is a neuron and can learn different patterns. so 64 units means 64 different patterns.
        # return_sequences=True: Passes the full sequence to the next LSTM layer. i.e This means the layer returns
        # output at every time step, not only the last one. We need this because we are stacking another LSTM layer on top.
        # input_shape: (24, 15) -> 24 hours of history, 15 features per hour.
        # Dropout: Randomly ignores 20% of neurons to prevent overfitting (memorization).
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), 
        
        # Layer 2: LSTM (The Decision Maker)
        # This layer is smaller because it is summarizing the sequence into a final understanding.
        # return_sequences=False: Condenses the sequence into a single vector of understanding.
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Layer 3: Dense (Interpretation), Dense layer is a fully connected neural network layer.
        # Standard neural network layer to interpret the LSTM's output.
        # Without activation functions, a neural network is just linear math. so Activation functions make 
        # neural networks capable of learning curves, shapes, and complex relationships i.e hidden patterns.
        Dense(16, activation='relu'),

        # Output Layer: The Prediction
        # Units=1: Outputs a single number (the predicted power).
        # Linear activation: Allows the output to be any continuous value (regression).
        Dense(1, activation='linear')
    ])
    
    # COMPILE: Setting the Rules for Learning
    # Optimizer='adam': An adaptive learning rate algorithm. Default LR is usually 0.001.
    # Loss='mse': Mean Squared Error. The model tries to minimize the square of the difference between prediction and reality.
    # Metrics=['mae']: We also watch Mean Absolute Error to understand average error in simple terms.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
