# Dataset Overview: GermanSolarFarm

This document provides a visual overview of the dataset used for training the Solar Power Forecasting model.

## Data Structure Visualization

The dataset is constructed from two primary sources, similar to your description: **Numerical Weather Prediction (NWP)** and **Local Measurements**.

```mermaid
graph TD
    subgraph "Data Sources"
        NWP[Numerical Weather Prediction]
        LMD[Local Measurements]
    end

    subgraph "Dataset Features"
        direction TB
        
        subgraph "Weather Features (from NWP)"
            W1[Global Irradiance]
            W2[Direct Irradiance]
            W3[Diffuse Irradiance]
            W4[Temperature]
            W5[Humidity]
            W6[Wind Speed]
            W7[Wind Direction]
            W8[Cloud Cover]
        end

        subgraph "Local Features (from LMD)"
            L1[PV Power Output]
        end
        
        subgraph "Calculated / Time Features"
            C1[Clear Sky Models]
            C2[Sun Position]
            C3[Time (Hour, Month)]
        end
    end

    NWP --> W1 & W2 & W3 & W4 & W5 & W6 & W7 & W8
    LMD --> L1
    
    W1 & W2 & W3 & W4 & W5 & W6 & W7 & W8 --> ModelInput
    L1 --> ModelInput
    C1 & C2 & C3 --> ModelInput

    ModelInput[Final Training Dataset]
```

## Detailed Feature Comparison

Here is a comparison between the dataset structure you described and the actual **GermanSolarFarm** dataset used in this project.

| Feature Category | Feature Name | Source | Included in Our Dataset? | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Irradiance** | Global Irradiance | NWP | ✅ | `SolarRadiationGlobalAt0` |
| | Direct Irradiance | NWP | ✅ | `SolarRadiationDirectAt0` |
| | Diffuse Irradiance | NWP | ✅ | `SolarRadiationDiffuseAt0` |
| **Weather** | Temperature | NWP | ✅ | `TemperatureAt0` |
| | Humidity | NWP | ✅ | `RelativeHumidityAt0` |
| | Wind Speed | NWP | ✅ | `LowerWindSpeed` |
| | Wind Direction | NWP | ✅ | `LowerWindDirection` |
| | Pressure | NWP | ❌ | Not currently used in `config.py` |
| | **Cloud Cover** | NWP | ✅ | **Extra Feature** (`TotalCloudCoverAt0`) |
| **Local Data** | **PV Power Output** | LMD | ✅ | Target Variable (`power_normed`) |
| **Calculated** | Clear Sky Models | Calculated | ✅ | Reference values for clear days |
| | Sun Position | Calculated | ✅ | Zenith & Azimuth angles |

## Summary

*   **Matches:** The core structure is identical: **Weather Data + Local Power Output**.
*   **Enhancements:** Our dataset includes **Cloud Cover** and **Clear Sky** calculations, which are highly beneficial for solar forecasting.
*   **Missing:** We are not explicitly using **Pressure**, likely because it has a lower correlation with short-term solar output compared to cloud cover and irradiance.
