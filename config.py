# Configuration for Solar Power Forecasting

# Data Paths
DATA_FILE = 'GermanSolarFarm/data/pv_*.csv'
MODEL_FILE = 'solar_lstm_model.keras'
SCALER_FILE = 'scaler.pkl'

# Model Hyperparameters
# TIME_STEPS = 24: The model looks at the past 24 hours of data to make a prediction.
# This is the "window" of history the LSTM sees.
TIME_STEPS = 24  # Number of past hours to look at
BATCH_SIZE = 32
EPOCHS = 50
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

# Feature Columns (Do not change order unless retraining)
# These are the INPUTS the model sees for every hour in the time window.
# It learns patterns from these 15 variables.
FEATURE_COLS = [
    'site_id', # Added site_id as the first feature
    'hour_of_day', 'month_of_year',
    'sunposition_thetaZ', 'sunposition_solarAzimuth', 
    'clearsky_diffuse', 'clearsky_direct', 'clearsky_global',
    'TemperatureAt0', 'RelativeHumidityAt0', 
    'SolarRadiationGlobalAt0', 'SolarRadiationDirectAt0', 'SolarRadiationDiffuseAt0',
    'TotalCloudCoverAt0', 'LowerWindSpeed', 'LowerWindDirection'
]

# Column Metadata for UI
COLUMN_METADATA = {
    'time_idx': {'label': 'Time Index', 'unit': '', 'description': 'Sequential index of the time step'},
    'power_normed': {'label': 'Power Output (Norm)', 'unit': '0-1', 'description': 'Normalized solar power output'},
    'hour_of_day': {'label': 'Hour of Day', 'unit': 'h', 'description': 'Hour of the day (0-23)'},
    'month_of_year': {'label': 'Month', 'unit': '', 'description': 'Month of the year (1-12)'},
    'sunposition_thetaZ': {'label': 'Zenith Angle', 'unit': 'deg', 'description': 'Solar zenith angle'},
    'sunposition_solarAzimuth': {'label': 'Azimuth Angle', 'unit': 'deg', 'description': 'Solar azimuth angle'},
    'clearsky_diffuse': {'label': 'Clear Sky Diffuse', 'unit': 'W/m²', 'description': 'Diffuse radiation under clear sky'},
    'clearsky_direct': {'label': 'Clear Sky Direct', 'unit': 'W/m²', 'description': 'Direct radiation under clear sky'},
    'clearsky_global': {'label': 'Clear Sky Global', 'unit': 'W/m²', 'description': 'Global radiation under clear sky'},
    'TemperatureAt0': {'label': 'Temperature', 'unit': '°C', 'description': 'Ambient temperature'},
    'RelativeHumidityAt0': {'label': 'Humidity', 'unit': '%', 'description': 'Relative humidity'},
    'SolarRadiationGlobalAt0': {'label': 'Global Radiation', 'unit': 'W/m²', 'description': 'Global solar radiation'},
    'SolarRadiationDirectAt0': {'label': 'Direct Radiation', 'unit': 'W/m²', 'description': 'Direct solar radiation'},
    'SolarRadiationDiffuseAt0': {'label': 'Diffuse Radiation', 'unit': 'W/m²', 'description': 'Diffuse solar radiation'},
    'TotalCloudCoverAt0': {'label': 'Cloud Cover', 'unit': '%', 'description': 'Total cloud cover'},
    'LowerWindSpeed': {'label': 'Wind Speed', 'unit': 'm/s', 'description': 'Wind speed at lower level'},
    'LowerWindDirection': {'label': 'Wind Direction', 'unit': 'deg', 'description': 'Wind direction at lower level'}
}
