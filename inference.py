import joblib
import pandas as pd
import numpy as np
import sys

def load_model(model_path='solar_model.pkl'):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_realtime(model, weather_data):
    """
    Predicts power output based on weather data.
    
    Args:
        model: Trained Random Forest model.
        weather_data (dict): Dictionary containing weather features.
        
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
    
    # Create DataFrame from input
    input_df = pd.DataFrame([weather_data])
    
    # Ensure all columns are present (fill missing with 0 or defaults)
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to match training
    input_df = input_df[feature_cols]
    
    prediction = model.predict(input_df)
    return prediction[0]

if __name__ == "__main__":
    model = load_model()
    
    if model:
        print("Model loaded. Simulating real-time prediction...")
        
        # Example dummy data (representing a sunny day at noon)
        dummy_weather = {
            'hour_of_day': 12,
            'month_of_year': 6,
            'sunposition_thetaZ': 0.2, # High sun
            'sunposition_solarAzimuth': 180,
            'clearsky_diffuse': 50,
            'clearsky_direct': 800,
            'clearsky_global': 850,
            'TemperatureAt0': 25,
            'RelativeHumidityAt0': 40,
            'SolarRadiationGlobalAt0': 800,
            'SolarRadiationDirectAt0': 700,
            'SolarRadiationDiffuseAt0': 100,
            'TotalCloudCoverAt0': 0.1,
            'LowerWindSpeed': 3,
            'LowerWindDirection': 180
        }
        
        pred = predict_realtime(model, dummy_weather)
        print(f"Predicted Normalized Power: {pred:.4f}")
        print("Note: This is a normalized value (0-1). Multiply by installed capacity for kW.")
    else:
        print("Please train the model first using train_model.py")
