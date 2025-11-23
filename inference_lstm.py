import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def load_lstm_model(model_path='solar_lstm_model.h5'):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_scaler(scaler_path='scaler.pkl'):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

def predict_realtime_lstm(model, scaler, recent_data, time_steps=24):
    """
    Predicts power output based on a sequence of recent weather data.
    
    Args:
        model: Trained LSTM model.
        scaler: Fitted MinMaxScaler.
        recent_data (pd.DataFrame or list of dicts): Data for the last 'time_steps' hours.
                                                     Must contain all feature columns.
        time_steps (int): The sequence length the model expects (default 24).
        
    Returns:
        float: Predicted power output (normalized).
    """
    # Expected features in order
    feature_cols = [
        'hour_of_day', 'month_of_year',
        'sunposition_thetaZ', 'sunposition_solarAzimuth', 
        'clearsky_diffuse', 'clearsky_direct', 'clearsky_global',
        'TemperatureAt0', 'RelativeHumidityAt0', 
        'SolarRadiationGlobalAt0', 'SolarRadiationDirectAt0', 'SolarRadiationDiffuseAt0',
        'TotalCloudCoverAt0', 'LowerWindSpeed', 'LowerWindDirection'
    ]
    
    # Convert to DataFrame if list
    if isinstance(recent_data, list):
        df = pd.DataFrame(recent_data)
    else:
        df = recent_data.copy()
        
    # Ensure columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
            
    # Select and reorder columns
    df = df[feature_cols]
    
    # Check if we have enough data
    if len(df) < time_steps:
        print(f"Error: Not enough data. Need {time_steps} time steps, got {len(df)}.")
        return None
        
    # Take the last 'time_steps' rows
    df_seq = df.iloc[-time_steps:]
    
    # Normalize
    X_scaled = scaler.transform(df_seq)
    
    # Reshape for LSTM (1, time_steps, features)
    X_input = X_scaled.reshape(1, time_steps, len(feature_cols))
    
    prediction = model.predict(X_input)
    return prediction[0][0]

if __name__ == "__main__":
    model = load_lstm_model()
    scaler = load_scaler()
    
    if model and scaler:
        print("Model and scaler loaded. Simulating real-time prediction...")
        
        # Create dummy sequence data (24 hours of data)
        # Simulating a day cycle
        dummy_data = []
        for i in range(24):
            hour = i
            # Simple simulation of sun rising and setting
            is_day = 6 <= hour <= 18
            rad = 800 if is_day else 0
            
            row = {
                'hour_of_day': hour,
                'month_of_year': 6,
                'sunposition_thetaZ': 0.5 if is_day else 1.0,
                'sunposition_solarAzimuth': 180,
                'clearsky_diffuse': 50 if is_day else 0,
                'clearsky_direct': 800 if is_day else 0,
                'clearsky_global': 850 if is_day else 0,
                'TemperatureAt0': 20 + (5 if is_day else 0),
                'RelativeHumidityAt0': 40,
                'SolarRadiationGlobalAt0': rad,
                'SolarRadiationDirectAt0': rad * 0.8,
                'SolarRadiationDiffuseAt0': rad * 0.2,
                'TotalCloudCoverAt0': 0.1,
                'LowerWindSpeed': 3,
                'LowerWindDirection': 180
            }
            dummy_data.append(row)
            
        pred = predict_realtime_lstm(model, scaler, dummy_data)
        if pred is not None:
            print(f"Predicted Normalized Power for next hour: {pred:.4f}")
    else:
        print("Please train the model first using train_lstm.py")
