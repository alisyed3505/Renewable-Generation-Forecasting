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
# These are the INPUTS the model sees for every 3 hours in the time window.
# It learns patterns from these 16 variables.
FEATURE_COLS = [
    'site_id',

    # Cyclical time encoding
    'hour_of_day_sin', 'hour_of_day_cos',
    'month_of_year_sin', 'month_of_year_cos',

    # Sun position
    'sunposition_thetaZ',
    'sunposition_solarAzimuth',

    # Clearsky model
    'clearsky_diffuse',
    'clearsky_direct',
    'clearsky_global',

    # Weather features
    'TemperatureAt0',
    'RelativeHumidityAt0',
    'SolarRadiationGlobalAt0',
    'SolarRadiationDirectAt0',
    'SolarRadiationDiffuseAt0',
    'TotalCloudCoverAt0'
]

# Column Metadata for UI
COLUMN_METADATA = {
    'time_idx': {
        'label': 'Time Index',
        'unit': '',
        'description': 'Sequential index of the time step'
    },

    'site_id': {
        'label': 'PV Site ID',
        'unit': '',
        'description': 'Unique identifier for the solar plant'
    },

    'power_normed': {
        'label': 'Power Output (Norm)',
        'unit': '0-1',
        'description': 'Normalized solar power output'
    },

    # -------------------------------
    # Cyclical Time Encoding
    # -------------------------------
    'hour_of_day_sin': {
        'label': 'Hour of Day (Sine)',
        'unit': '',
        'description': 'sin(2π * hour / 24) encoding of hour'
    },
    'hour_of_day_cos': {
        'label': 'Hour of Day (Cosine)',
        'unit': '',
        'description': 'cos(2π * hour / 24) encoding of hour'
    },
    'month_of_year_sin': {
        'label': 'Month (Sine)',
        'unit': '',
        'description': 'sin(2π * (month-1) / 12) encoding of month'
    },
    'month_of_year_cos': {
        'label': 'Month (Cosine)',
        'unit': '',
        'description': 'cos(2π * (month-1) / 12) encoding of month'
    },

    # -------------------------------
    # Sun Geometry
    # -------------------------------
    'sunposition_thetaZ': {
        'label': 'Zenith Angle',
        'unit': 'deg',
        'description': 'Solar zenith angle'
    },
    'sunposition_solarAzimuth': {
        'label': 'Azimuth Angle',
        'unit': 'deg',
        'description': 'Solar azimuth angle'
    },

    # -------------------------------
    # Clear-sky Radiation
    # -------------------------------
    'clearsky_diffuse': {
        'label': 'Clear Sky Diffuse',
        'unit': 'W/m²',
        'description': 'Diffuse radiation under clear sky'
    },
    'clearsky_direct': {
        'label': 'Clear Sky Direct',
        'unit': 'W/m²',
        'description': 'Direct radiation under clear sky'
    },
    'clearsky_global': {
        'label': 'Clear Sky Global',
        'unit': 'W/m²',
        'description': 'Global radiation under clear sky'
    },

    # -------------------------------
    # Weather
    # -------------------------------
    'TemperatureAt0': {
        'label': 'Temperature',
        'unit': '°C',
        'description': 'Ambient temperature'
    },
    'RelativeHumidityAt0': {
        'label': 'Humidity',
        'unit': '%',
        'description': 'Relative humidity'
    },

    # -------------------------------
    # Measured Solar Radiation
    # -------------------------------
    'SolarRadiationGlobalAt0': {
        'label': 'Global Radiation',
        'unit': 'W/m²',
        'description': 'Global solar radiation'
    },
    'SolarRadiationDirectAt0': {
        'label': 'Direct Radiation',
        'unit': 'W/m²',
        'description': 'Direct solar radiation'
    },
    'SolarRadiationDiffuseAt0': {
        'label': 'Diffuse Radiation',
        'unit': 'W/m²',
        'description': 'Diffuse solar radiation'
    },

    'TotalCloudCoverAt0': {
        'label': 'Cloud Cover',
        'unit': '%',
        'description': 'Total cloud cover'
    }
}
