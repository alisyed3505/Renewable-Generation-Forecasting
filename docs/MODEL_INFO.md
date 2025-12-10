# Solar Power Forecasting Model Card

## Model Overview
*   **Type**: Long Short-Term Memory (LSTM) Neural Network.
*   **Framework**: TensorFlow / Keras.
*   **Purpose**: Forecast solar power generation for the next hour based on the past 24 hours of weather and solar data.
*   **Training Data**: 21 Solar Farms in Germany (`pv_01.csv` to `pv_21.csv`).

## Input Features (Parameters)
The model uses **15 input features** for every time step. It looks at a sequence of these 15 features over the last **24 hours** (Total input size: 24 x 15).

### 1. Time & Sun Position
*   `hour_of_day`: 0-23. Helps model learn daily cycles.
*   `month_of_year`: 1-12. Helps model learn seasonal shifts.
*   `sunposition_thetaZ`: Zenith angle (height of sun). Critical for potential power.
*   `sunposition_solarAzimuth`: Direction of sun.

### 2. Solar Radiation (The Fuel)
*   `SolarRadiationGlobalAt0` (GHI): Total radiation hitting flat ground. Most important.
*   `SolarRadiationDirectAt0` (DNI): Direct beam from sun.
*   `SolarRadiationDiffuseAt0` (DHI): Scattered light (clouds/atmosphere).
*   `clearsky_*`: Theoretical maximums for GHI/DNI/DHI if there were no clouds.

### 3. Weather Conditions
*   `TotalCloudCoverAt0`: 0.0 (Clear) to 1.0 (Overcast). Heavily impacts output.
*   `TemperatureAt0`: Ambient temp. Panels are *less* efficient at high temps.
*   `RelativeHumidityAt0`: Affects atmosphere.
*   `LowerWindSpeed` / `LowerWindDirection`: Wind cools panels, improving efficiency.

## Model Capabilities & Insights
*   **Short-Term Forecasting**: Predicts power output for the next hour.
*   **Cloud Impact**: Can predict drops in power due to incoming cloud fronts (captured in the 24h history).
*   **Seasonal Awareness**: Knows that 12:00 PM in Winter produces less power than 12:00 PM in Summer.
*   **Efficiency**: Implicitly learns the temperature coefficient of the panels (heat = lower efficiency).

## Performance
*   **RMSE (Root Mean Squared Error)**: ~0.06 (Normalized).
    *   This means the error is typically around **6%** of the system's total capacity.
*   **MAE (Mean Absolute Error)**: ~0.03 (Normalized).
    *   On average, the prediction is off by **3%**.

## How to Interpret Results
The output is a **normalized value between 0 and 1**.
*   `0.0` = No power (Night).
*   `1.0` = Maximum rated capacity (Peak sunny day).
*   **To get kW**: Multiply the prediction by your system's size (e.g., 5 kW system * 0.8 prediction = 4 kW output).
