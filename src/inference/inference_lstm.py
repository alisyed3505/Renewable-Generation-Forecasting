import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def load_lstm_model(model_path='models/solar_lstm_model.keras'):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_scaler(scaler_path='models/scaler.pkl'):
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
    # Expected features in order (Must match training)
    from config import FEATURE_COLS
    
    # Check if 'site_id' is in config but not in input data
    # If missing, default to site 1 (or prompt error)
    # We handle this inside the dataframe preparation
    
    # Convert to DataFrame if list
    if isinstance(recent_data, list):
        df = pd.DataFrame(recent_data)
    else:
        df = recent_data.copy()
        
    if 'hour_of_day' in df.columns:
        df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

    if 'month_of_year' in df.columns:
        df['month_of_year_sin'] = np.sin(2 * np.pi * (df['month_of_year'] - 1) / 12)
        df['month_of_year_cos'] = np.cos(2 * np.pi * (df['month_of_year'] - 1) / 12)

    # Ensure columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            if col == 'site_id':
                # Default to site 1 if not provided
                print("Warning: 'site_id' missing in input data. Defaulting to 1.")
                df[col] = 1
            else:
                df[col] = 0
            
    # Select and reorder columns
    df = df[FEATURE_COLS]
    
    # Check if we have enough data
    if len(df) < time_steps:
        print(f"Error: Not enough data. Need {time_steps} time steps, got {len(df)}.")
        return None
        
    # Take the last 'time_steps' rows
    df_seq = df.iloc[-time_steps:]
    
    # Normalize
    X_scaled = scaler.transform(df_seq)
    
    # Reshape for LSTM (1, time_steps, features)
    X_input = X_scaled.reshape(1, time_steps, len(FEATURE_COLS))
    
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
            month = 6

            is_day = 6 <= hour <= 18
            rad = 800 if is_day else 0

            row = {
                'site_id': 1,
                'hour_of_day': hour,
                'month_of_year': month,
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
                'TotalCloudCoverAt0': 0.1
            }

            # Cyclical time encoding
            row['hour_of_day_sin']  = np.sin(2 * np.pi * hour / 24)
            row['hour_of_day_cos']  = np.cos(2 * np.pi * hour / 24)
            row['month_of_year_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
            row['month_of_year_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

            dummy_data.append(row)
            
        pred = predict_realtime_lstm(model, scaler, dummy_data)
        if pred is not None:
            # If I later obtain installed capacity → I'll convert to real kW/MW
            # installed_capacity_kw = 8500   # 8.5 MW
            # pred_kw = pred * installed_capacity_kw
            # print(f"Predicted Power: {pred_kw:.1f} kW ({pred*100:.2f}%)")

            if pred <= 0.05:
                status = "Very low / night time"
            elif pred <= 0.30:
                status = "Low production"
            elif pred <= 0.70:
                status = "Moderate production"
            else:
                status = "High production"

            print(f"Normalized Prediction: {pred:.4f}")
            print(f"Percentage of max capacity: {pred*100:.2f}%")
            print(f"Estimated status: {status}")
    else:
        print("Please train the model first using train_lstm.py")
