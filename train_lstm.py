import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from data_loader import load_data, preprocess_for_lstm, split_data_lstm

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

"""
 Train an LSTM model on data from this data_path, using time_steps history, and save it to model_path
 parameters:
    data_path: Path to the data file
    model_path: Path to save the model
    time_steps: Number of time steps to look back
"""
def train_lstm(data_path='GermanSolarFarm/data/pv_01.csv', model_path='solar_lstm_model.h5', time_steps=24):
    print("Loading and preprocessing data...")
    X, y = load_data(data_path)
    
    if X is None:
        return

    # Create sequences
    # time_steps=24 means we look at the past 24 hours to predict the next hour
    X_seq, y_seq, scaler = preprocess_for_lstm(X, y, time_steps=time_steps)
    
    print(f"Input shape: {X_seq.shape}")
    
    # Split data
    # X_train, y_train → used to train the model
    # X_test, y_test → used to evaluate how well the model performs on future/unseen data
    X_train, X_test, y_train, y_test = split_data_lstm(X_seq, y_seq)
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Callbacks
    # patience=10: If the validation loss doesn’t improve for 10 epochs → STOP training -> Prevents overfitting.
    # ModelCheckpoint: Save the best model during training. You never lose the best-performing weights.
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    
    # Train the LSTM model on training sequences and monitor validation performance.
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )

    # FUTURE IMPROVEMENTS
    # Monitor overfitting
    # Improve model based on training behavior
    # Plot training curves/history
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.legend()
    # plt.title('Training History')
    
    # Ask the model to forecast power for unseen sequences (test set).
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # RMSE: punishes large mistakes more.
    # MAE: simple average error.
    # These tell you how good your model is in numbers.
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"LSTM Model Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save metrics
    with open('lstm_metrics.txt', 'w') as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_lstm()
