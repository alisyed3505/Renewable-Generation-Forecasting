# Live Data Guide for Solar Power Forecasting

To use this LSTM model for real-time commercial or scientific forecasting, you need to feed it accurate, live weather data. The model expects a sequence of the past 24 hours of data to predict the next hour.

## Required Data Points

Your model requires the following parameters for every hour:

1.  **Solar Radiation**:
    *   `SolarRadiationGlobalAt0` (GHI - Global Horizontal Irradiance)
    *   `SolarRadiationDirectAt0` (DNI - Direct Normal Irradiance)
    *   `SolarRadiationDiffuseAt0` (DHI - Diffuse Horizontal Irradiance)
    *   `clearsky_*` (Clear Sky GHI/DNI/DHI - theoretical max radiation)
2.  **Weather**:
    *   `TemperatureAt0` (Ambient Temperature)
    *   `RelativeHumidityAt0`
    *   `TotalCloudCoverAt0` (0-1 or 0-100%)
    *   `LowerWindSpeed`
    *   `LowerWindDirection`
3.  **Time/Sun**:
    *   `sunposition_thetaZ` (Zenith Angle)
    *   `sunposition_solarAzimuth`

## Recommended APIs

### 1. Solcast API (Best for Solar)
Solcast specializes in solar data and provides exactly the parameters needed (GHI, DNI, DHI, Zenith, Azimuth).
*   **Website**: [solcast.com](https://solcast.com/)
*   **Pros**: High accuracy, specific solar parameters, commercial grade.
*   **Cons**: Paid for commercial use (free tier available for researchers/hobbyists).

### 2. OpenWeatherMap (One Call API 3.0)
Good for general weather (Temp, Humidity, Clouds, Wind).
*   **Website**: [openweathermap.org](https://openweathermap.org/)
*   **Pros**: Cheap, easy to use.
*   **Cons**: Doesn't always provide DNI/DHI directly (only GHI/UV index often), so you might need to estimate them or use a solar-specific add-on.

## How to Integrate

1.  **Get an API Key** from one of the providers above.
2.  **Create a Fetch Script**: Write a Python script to fetch the last 24 hours of data.
    *   *Note: Most APIs give you a "Forecast" (future) or "History" (past). For the LSTM input, you need the **past 24 hours of observed/estimated data**.*
3.  **Format the Data**: Map the API response fields to the model's expected column names (see `inference_lstm.py`).
4.  **Run Inference**: Pass this list of 24 data points to `predict_realtime_lstm`.

### Example Solcast Mapping
```python
# Solcast response -> Model input
model_input = {
    'SolarRadiationGlobalAt0': solcast_data['ghi'],
    'SolarRadiationDirectAt0': solcast_data['dni'],
    'TemperatureAt0': solcast_data['air_temp'],
    # ... etc
}
```

## Scientific/Commercial Tips
*   **Location**: Ensure the API request is for the exact lat/long of your PV site.
*   **Calibration**: Real-world sensors (pyranometers) on site are better than satellite APIs. If you have on-site hardware, log that data to a CSV and read it instead of using an API.
*   **Retraining**: As you collect more real data from your specific site, retrain the LSTM model periodically (e.g., monthly) to adapt to local microclimates.
