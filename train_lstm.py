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
    """
    model = Sequential([
        # Layer 1: LSTM (The Sequence Reader)
        # Units=64: The capacity of this layer to remember patterns.
        # return_sequences=True: Passes the full sequence to the next LSTM layer.
        # input_shape: (24, 15) -> 24 hours of history, 15 features per hour.
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), # Dropout: Randomly ignores 20% of neurons to prevent overfitting (memorization).
        
        # Layer 2: LSTM (The Decision Maker)
        # return_sequences=False: Condenses the sequence into a single vector of understanding.
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Layer 3: Dense (Interpretation)
        # Standard neural network layer to interpret the LSTM's output.
        Dense(16, activation='relu'),

        # Output Layer: The Prediction
        # Units=1: Outputs a single number (the predicted power).
        # Linear activation: Allows the output to be any continuous value (regression).
        Dense(1, activation='linear') # Linear activation for regression
    ])
    
    # COMPILE: Setting the Rules for Learning
    # Optimizer='adam': An adaptive learning rate algorithm. Default LR is usually 0.001.
    # Loss='mse': Mean Squared Error. The model tries to minimize the square of the difference between prediction and reality.
    # Metrics=['mae']: We also watch Mean Absolute Error to understand average error in simple terms.
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

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
    X_train, X_test, y_train, y_test = split_data_lstm(X_seq, y_seq)
    
    # Build model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
    
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
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
