# Project Progress Update: Solar Power Forecasting

This document outlines the current state of the repository, detailing the data preprocessing pipeline and the implementation of the LSTM model for solar power forecasting.

## 1. Repository & Dataset Overview

The project is built around the **GermanSolarFarm** dataset, which provides a robust foundation for training our forecasting model.

*   **Initial State**: The repository was initialized with historical solar data separated into multiple CSV files.
*   **Data Composition**: The dataset effectively combines two critical data sources:
    *   **Numerical Weather Prediction (NWP)**: Includes predictive features such as Global/Direct/Diffuse Irradiance, Temperature, Humidity, Wind Speed/Direction, and **Total Cloud Cover**.
    *   **Local Measurements (LMD)**: Contains the actual **PV Power Output** (`power_normed`), which serves as our target variable.
*   **Feature Enhancements**: We have verified the inclusion of calculated features like **Clear Sky Models** and **Sun Position** (Zenith & Azimuth angles), which provide essential physical context for the model.

## 2. Data Preprocessing Pipeline

We have implemented a robust preprocessing pipeline in `data_loader.py` to transform raw data into a format suitable for Deep Learning.

### Key Steps:
1.  **Data Aggregation**:
    *   Implemented a dynamic loading function that uses `glob` to find and merge multiple dataset files (`pv_*.csv`) into a single, continuous DataFrame.
    *   This ensures we can train on the entire available history rather than just single segments.

2.  **Cleaning & Imputation**:
    *   Automated the removal of artifact columns (e.g., 'Unnamed').
    *   Applied **Forward Fill** and **Backward Fill** strategies to handle missing values, ensuring no data gaps disrupt the time series.

3.  **Feature Selection**:
    *   Curated a specific list of 15 input features, focusing on those with the highest correlation to power output (e.g., `SolarRadiationGlobalAt0`, `TemperatureAt0`, `TotalCloudCoverAt0`).

4.  **Normalization**:
    *   Applied `MinMaxScaler` to scale all input features to a range of `[0, 1]`.
    *   **Why**: LSTMs are sensitive to the scale of input data; normalization prevents features with large values (like Irradiance) from dominating the gradient updates.

5.  **Sequence Generation (Sliding Window)**:
    *   Transformed the 2D tabular data into 3D sequences: `(Samples, Time Steps, Features)`.
    *   **Configuration**: We are using a **24-hour lookback window** (`time_steps=24`). The model looks at the past 24 hours of weather and power data to predict the power output for the next hour.

## 3. LSTM Model Implementation

We have moved beyond basic regression and are now building a custom **Long Short-Term Memory (LSTM)** network in `train_lstm.py`.

### Architecture Design:
*   **Input Layer**: Accepts sequences of shape `(24, 15)` (24 hours, 15 features).
*   **First LSTM Layer (64 units)**:
    *   `return_sequences=True`: Passes the full sequence of hidden states to the next layer, allowing the model to learn complex temporal dependencies.
*   **Dropout (0.2)**: Randomly drops 20% of connections during training to prevent overfitting.
*   **Second LSTM Layer (32 units)**:
    *   `return_sequences=False`: Compresses the temporal information into a single vector for the final prediction.
*   **Output Layer**: A single Dense neuron with **Linear Activation** to predict the continuous value of normalized power output.

### Training Strategy:
*   **Optimizer**: Adam (adaptive learning rate).
*   **Loss Function**: Mean Squared Error (MSE), standard for regression tasks.
*   **Callbacks**:
    *   `EarlyStopping`: Monitors validation loss and stops training if it stops improving, saving time and preventing overfitting.
    *   `ModelCheckpoint`: Automatically saves the best version of the model during training.

## 4. System Architecture & Prediction Logic

This section explains how the system functions in a real-world scenario to predict future power output (and potentially price).

### The "Sliding Window" Mechanism
The core of our prediction logic is the **Sliding Window** approach. The model does not just look at the "current" moment; it looks at a history window to understand trends (e.g., "is the sun rising or setting?", "is cloud cover increasing?").

```mermaid
graph LR
    subgraph "Input Window (Past 24 Hours)"
        T_minus_23[t-23]
        T_minus_22[t-22]
        dots[...]
        T_0[Current Time (t)]
    end

    subgraph "The Model"
        LSTM[LSTM Network]
    end

    subgraph "Prediction (Future)"
        T_plus_1[Forecast: t+1 Hour]
    end

    T_minus_23 --> LSTM
    T_minus_22 --> LSTM
    dots --> LSTM
    T_0 --> LSTM
    LSTM --> T_plus_1
```

### How Prediction Works (Step-by-Step)
1.  **Data Collection**: The system collects the last 24 data points (hours) of weather and power data.
2.  **Preprocessing**: This 24-hour block is normalized using the same scaler saved during training.
3.  **Inference**: The LSTM processes this sequence. It recognizes patterns (e.g., "Morning ramp-up with decreasing clouds").
4.  **Output**: The model outputs a single value: the predicted **Normalized Power** for the *next hour*.
5.  **Post-Processing**: We inverse-transform this value to get the actual Kilowatts (kW).

### Power vs. Price Prediction
*   **Current Capability**: The system currently predicts **Solar Power Generation (kW)**.
*   **Price Prediction**: To predict **Price** (Revenue), we would apply a simple transformation:
    $$ \text{Predicted Price} = \text{Predicted Power (kW)} \times \text{Current Energy Price (\$/kWh)} $$
*   **Future Expansion**: If energy prices fluctuate dynamically (e.g., spot market prices), we can add "Price" as an input feature to the LSTM to forecast price directly, but currently, we focus on the physical generation.

## 5. Summary of Progress
We have successfully transitioned from raw data analysis to a fully functional Deep Learning pipeline. The system can now:
1.  Ingest and clean raw CSV data.
2.  Preprocess and structure data for time-series forecasting.
3.  Train a multi-layer LSTM model to learn the relationship between weather patterns and solar output.
