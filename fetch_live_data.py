import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import config

# API Keys (Ideally move these to env vars or config)
SOLCAST_KEY = 'yzC0UuVu9DZakpdfen7juaRMGHc_S3VE'
OWM_KEY = 'a88daa846d2bd86e09b83becc0cc088b'

# Location (Berlin, Germany - matching training data region)
LAT = 52.5200
LON = 13.4050

def fetch_solcast_history():
    """
    Fetches the last 24 hours of solar radiation data from Solcast.
    Returns a DataFrame with columns: ghi, dni, dhi, cloud_opacity
    """
    print("Fetching Solcast data...")
    url = f"https://api.solcast.com.au/world_radiation/estimated_actuals?latitude={LAT}&longitude={LON}&hours=24&format=json&api_key={SOLCAST_KEY}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['estimated_actuals'])
            
            # Solcast gives 30 min intervals usually. We need hourly.
            # Convert period_end to datetime
            df['period_end'] = pd.to_datetime(df['period_end'])
            df.set_index('period_end', inplace=True)
            
            # Resample to hourly means
            df_hourly = df.resample('h').mean()
            return df_hourly
        else:
            print(f"Solcast Error: {response.text}")
            return None
    except Exception as e:
        print(f"Solcast Request Failed: {e}")
        return None

def fetch_openweather_current():
    """
    Fetches current weather (Temp, Humidity, Wind) from OpenWeatherMap.
    Note: OWM Free OneCall might not give history easily without paid subscription.
    For now, we will fetch current and assume it's constant or try to find a history endpoint if available.
    """
    print("Fetching OpenWeatherMap data...")
    # Using the 'weather' endpoint for current data (free tier standard)
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={OWM_KEY}&units=metric"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'wind_deg': data['wind']['deg'],
                'clouds': data['clouds']['all'] / 100.0 # Convert 0-100 to 0-1
            }
        else:
            print(f"OWM Error: {response.text}")
            return None
    except Exception as e:
        print(f"OWM Request Failed: {e}")
        return None

def prepare_live_sequence():
    """
    Combines Solcast and OWM data to create the 24h sequence for the model.
    """
    solcast_df = fetch_solcast_history()
    if solcast_df is None:
        print("Using dummy data fallback due to API error...")
        # Fallback to dummy data for demonstration
        dummy_data = []
        for i in range(24):
            hour = i
            is_day = 6 <= hour <= 18
            rad = 500 if is_day else 0
            row = {
                'hour_of_day': hour,
                'month_of_year': 6,
                'sunposition_thetaZ': 0.5,
                'sunposition_solarAzimuth': 180,
                'clearsky_diffuse': 50,
                'clearsky_direct': 800,
                'clearsky_global': 850,
                'TemperatureAt0': 20,
                'RelativeHumidityAt0': 50,
                'SolarRadiationGlobalAt0': rad,
                'SolarRadiationDirectAt0': rad * 0.8,
                'SolarRadiationDiffuseAt0': rad * 0.2,
                'TotalCloudCoverAt0': 0.1,
                'LowerWindSpeed': 3,
                'LowerWindDirection': 180
            }
            dummy_data.append(row)
        return dummy_data
        
    owm_data = fetch_openweather_current()
    if owm_data is None:
        # Fallback defaults if OWM fails
        owm_data = {'temp': 15, 'humidity': 50, 'wind_speed': 3, 'wind_deg': 180, 'clouds': 0.5}
    
    # We need to construct the exact feature columns expected by the model
    # FEATURE_COLS = [
    #     'hour_of_day', 'month_of_year',
    #     'sunposition_thetaZ', 'sunposition_solarAzimuth', 
    #     'clearsky_diffuse', 'clearsky_direct', 'clearsky_global',
    #     'TemperatureAt0', 'RelativeHumidityAt0', 
    #     'SolarRadiationGlobalAt0', 'SolarRadiationDirectAt0', 'SolarRadiationDiffuseAt0',
    #     'TotalCloudCoverAt0', 'LowerWindSpeed', 'LowerWindDirection'
    # ]
    
    sequence_data = []
    
    # Iterate through the Solcast history (which should be sorted by time)
    # Solcast returns newest first usually? Let's check sort.
    solcast_df = solcast_df.sort_index()
    
    # We need exactly 24 steps. Take the last 24.
    if len(solcast_df) < 24:
        print(f"Warning: Solcast returned only {len(solcast_df)} hourly points. Padding...")
        # Pad with the first row if needed
        while len(solcast_df) < 24:
            solcast_df = pd.concat([solcast_df.iloc[[0]], solcast_df])
            
    solcast_df = solcast_df.iloc[-24:]
    
    for index, row in solcast_df.iterrows():
        dt = index
        
        # Calculate Sun Position (Approximate or use library like pvlib if installed)
        # For now, we'll use simple approximations or placeholders
        # Ideally: import pvlib; solpos = pvlib.solarposition.get_solarposition(dt, LAT, LON)
        
        # Mapping
        # Solcast 'ghi' -> SolarRadiationGlobalAt0
        # Solcast 'dni' -> SolarRadiationDirectAt0
        # Solcast 'dhi' -> SolarRadiationDiffuseAt0
        # Solcast 'cloud_opacity' -> TotalCloudCoverAt0 (approx)
        
        # For ClearSky, we can use GHI/DNI as proxies if it's sunny, or just duplicate them for now
        # (In a real app, calculate clear sky using a model)
        
        data_point = {
            'hour_of_day': dt.hour,
            'month_of_year': dt.month,
            'sunposition_thetaZ': 0.5, # Placeholder - would need pvlib for accuracy
            'sunposition_solarAzimuth': 180, # Placeholder
            'clearsky_diffuse': row['dhi'], # Approx
            'clearsky_direct': row['dni'],  # Approx
            'clearsky_global': row['ghi'],  # Approx
            'TemperatureAt0': owm_data['temp'], # Using current temp for history (approximation)
            'RelativeHumidityAt0': owm_data['humidity'],
            'SolarRadiationGlobalAt0': row['ghi'],
            'SolarRadiationDirectAt0': row['dni'],
            'SolarRadiationDiffuseAt0': row['dhi'],
            'TotalCloudCoverAt0': row['cloud_opacity'] / 100.0,
            'LowerWindSpeed': owm_data['wind_speed'],
            'LowerWindDirection': owm_data['wind_deg']
        }
        sequence_data.append(data_point)
        
    return sequence_data

if __name__ == "__main__":
    seq = prepare_live_sequence()
    if seq:
        print(f"Successfully prepared sequence of length {len(seq)}")
        print("Sample:", seq[-1])
