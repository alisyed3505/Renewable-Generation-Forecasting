# Configuration for Solar Power Forecasting

# Data Paths
DATA_FILE = 'GermanSolarFarm/data/pv_01.csv'
MODEL_FILE = 'solar_lstm_model.h5'
SCALER_FILE = 'scaler.pkl'

# Model Hyperparameters
TIME_STEPS = 24  # Number of past hours to look at
BATCH_SIZE = 32
EPOCHS = 50
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2

# Feature Columns (Do not change order unless retraining)
FEATURE_COLS = [
    'hour_of_day', 'month_of_year',
    'sunposition_thetaZ', 'sunposition_solarAzimuth', 
    'clearsky_diffuse', 'clearsky_direct', 'clearsky_global',
    'TemperatureAt0', 'RelativeHumidityAt0', 
    'SolarRadiationGlobalAt0', 'SolarRadiationDirectAt0', 'SolarRadiationDiffuseAt0',
    'TotalCloudCoverAt0', 'LowerWindSpeed', 'LowerWindDirection'
]
