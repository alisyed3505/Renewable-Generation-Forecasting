# Solar Power Forecasting Repository Index
## LLM-Optimized Repository Structure and Dependency Map

**Repository Name:** Solar-Power-Forecasting  
**Purpose:** PV (Photovoltaic) Energy Forecasting using LSTM and XGBoost models  
**Primary Language:** Python  
**Framework:** TensorFlow/Keras for deep learning  
**Last Updated:** 2026-01-08

---

## 📁 Repository Structure

```
Solar-Power-Forecasting/
│
├── config/                      # Configuration files for different model types
│   ├── baseline.py             # Config for single-site baseline LSTM
│   └── embedded.py             # Config for multi-site embedded LSTM
│
├── data/                       # Data storage
│   └── raw/                    # Raw PV data files
│       ├── pv_01.csv to pv_21.csv  # 21 different solar farm sites
│       ├── solarfarm_locations.jpg  # Geographic map of sites
│       └── README.txt
│
├── models/                     # Saved model artifacts
│   ├── xgboost/               # XGBoost models
│   │   └── xgb_model.json
│   ├── baseline_lstm.keras    # Saved baseline LSTM model
│   ├── embedded_lstm.keras    # Saved embedded LSTM model
│   ├── baseline_scaler.pkl    # Data scaler for baseline
│   ├── embedded_scaler.pkl    # Data scaler for embedded
│   └── metrics/               # Performance metrics
│       ├── baseline_metrics.txt
│       └── embedded_metrics.txt
│
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   │   ├── baseline/
│   │   │   └── data_loader.py
│   │   └── embedded/
│   │       └── data_loader.py
│   │
│   ├── models/                 # Model architectures
│   │   ├── baseline/
│   │   │   └── lstm_v1.py
│   │   ├── embedded/
│   │   │   └── lstm_v1.py
│   │   └── xgboost/
│   │       ├── __init__.py
│   │       ├── data.py
│   │       └── model_v1.py
│   │
│   ├── training/               # Training scripts
│   │   ├── train_baseline_lstm.py
│   │   ├── train_embedded_lstm.py
│   │   ├── train_xgboost.py
│   │   ├── train_all.py
│   │   └── optuna/
│   │       └── optimize_lstm.py
│   │
│   ├── evaluation/             # Model evaluation and visualization
│   │   ├── baseline/
│   │   │   ├── evaluate.py
│   │   │   ├── plots.py
│   │   │   └── validate_lstm.py
│   │   ├── embedded/
│   │   │   ├── evaluate.py
│   │   │   └── plots.py
│   │   ├── comparison/
│   │   │   └── compare_models.py
│   │   ├── naive/
│   │   │   ├── __init__.py
│   │   │   ├── evaluate.py
│   │   │   └── naive_24h.py
│   │   └── xgboost/
│   │       ├── __init__.py
│   │       ├── evaluate.py
│   │       └── feature_importance.py
│   │
│   ├── inference/              # Real-time prediction
│   │   ├── baseline/
│   │   │   └── predict.py
│   │   ├── embedded/
│   │   │   └── inference_embedded.py
│   │   ├── inference_xgb.py
│   │   └── utils_preprocess.py
│   │
│   ├── optimization/           # Hyperparameter optimization
│   │   └── optuna_lstm.py
│   │
│   ├── live/                   # Live data fetching
│   │   └── fetch_live_data.py
│   │
│   └── utils/                  # Utility functions
│       └── get_dataset_dates.py
│
├── web/                        # Web application
│   ├── main.py                # Main entry point
│   ├── server.py              # FastAPI server
│   └── static/
│       ├── visualization.html
│       └── js/
│           └── visualization.js
│
├── Scripts/                    # Analysis scripts
│   ├── PCA.py
│   └── Clear_Sky.py
│
├── docs/                       # Documentation
│
├── References/                 # Research papers
│   ├── 1843_smc2016.pdf
│   └── 122389.pdf
│
├── requirements.txt            # Python dependencies
├── .gitignore
└── _config.yml                # GitHub Pages config
```

---

## 🔗 File Dependencies and Import Graph

### Configuration Layer

#### `config/baseline.py`
**Purpose:** Configuration for single-site baseline LSTM model  
**Exports:**
- `DATA_FILE`: Path to single PV site data (pv_01.csv)
- `TIME_STEPS`: 8 (24h history at 3h resolution)
- `TEST_SPLIT`, `VAL_SPLIT`: Train/val/test split ratios
- `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`, `RANDOM_SEED`: Training hyperparameters
- `LSTM_UNITS_1`, `LSTM_UNITS_2`, `DENSE_UNITS`, `DROPOUT_RATE`: Model architecture params
- `FEATURE_COLS_BASELINE`: List of 14 features (cyclical time, sun position, clear-sky, weather)
- `MODEL_PATH`, `SCALER_PATH`, `METRICS_PATH`: Artifact paths

**Imports:** None (pure configuration)  
**Referenced By:**
- `src/training/train_baseline_lstm.py`
- `src/evaluation/baseline/evaluate.py`
- `config/embedded.py` (imports FEATURE_COLS_BASELINE)

---

#### `config/embedded.py`
**Purpose:** Configuration for multi-site embedded LSTM model  
**Exports:**
- `DATA_GLOB`: Pattern for all 21 PV site files
- `NUM_SITES`: 21
- `EMBEDDING_DIM`: 4 (site embedding dimension)
- Same training/model params as baseline
- Reuses `FEATURE_COLS_BASELINE` from baseline config

**Imports:**
- `from config.baseline import FEATURE_COLS_BASELINE`

**Referenced By:**
- `src/training/train_embedded_lstm.py`
- `src/evaluation/embedded/evaluate.py`

---

### Data Layer

#### `src/data/baseline/data_loader.py`
**Purpose:** Load and preprocess single-site data for baseline LSTM  
**Key Functions:**
- `load_single_site_csv(file_path)`: Load single CSV with datetime parsing
- `preprocess_baseline_data(df, scaler_path, time_steps)`: Create sequences, scale features

**Imports:**
- Standard: `pandas`, `numpy`, `sklearn.preprocessing.MinMaxScaler`, `pickle`
- Project: `config.baseline` (FEATURE_COLS_BASELINE)

**Referenced By:**
- `src/training/train_baseline_lstm.py`
- `src/evaluation/baseline/evaluate.py`

---

#### `src/data/embedded/data_loader.py`
**Purpose:** Load and preprocess multi-site data with site embeddings  
**Key Functions:**
- `load_all_sites(data_glob)`: Load all 21 sites and assign site_id
- `preprocess_embedded_data(data_glob, scaler_path)`: Create sequences with site IDs

**Imports:**
- Standard: `pandas`, `numpy`, `sklearn`, `pickle`, `glob`
- Project: `config.embedded` (DATA_GLOB, TIME_STEPS, FEATURE_COLS_BASELINE)

**Referenced By:**
- `src/training/train_embedded_lstm.py`
- `src/evaluation/embedded/evaluate.py`

---

### Model Architecture Layer

#### `src/models/baseline/lstm_v1.py`
**Purpose:** Baseline LSTM architecture for single-site forecasting  
**Key Functions:**
- `build_baseline_lstm(input_shape, lstm_units_1=64, lstm_units_2=32, ...)`: Builds Sequential model

**Architecture:**
- LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1)
- Loss: MSE, Optimizer: Adam, Metrics: MAE

**Imports:**
- `tensorflow.keras.models.Sequential`
- `tensorflow.keras.layers` (LSTM, Dense, Dropout)
- `tensorflow.keras.optimizers.Adam`

**Referenced By:**
- `src/training/train_baseline_lstm.py`

---

#### `src/models/embedded/lstm_v1.py`
**Purpose:** Embedded LSTM with site embeddings for multi-site forecasting  
**Key Functions:**
- `build_embedded_lstm(num_sites, embedding_dim, time_steps, num_features, ...)`: Builds functional model

**Architecture:**
- Site Input → Embedding(21→4) → Repeat across time
- Feature Input (time_steps, features)
- Concatenate [Features, Site Embedding]
- LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1)

**Inputs:**
- `site_id`: (batch, 1)
- `features`: (batch, time_steps, num_features)

**Imports:**
- `tensorflow.keras.models.Model`
- `tensorflow.keras.layers` (Input, Embedding, RepeatVector, Concatenate, LSTM, Dense, Dropout)
- `tensorflow.keras.optimizers.Adam`

**Referenced By:**
- `src/training/train_embedded_lstm.py`

---

#### `src/models/xgboost/model_v1.py`
**Purpose:** XGBoost model for solar forecasting  
**Key Functions:**
- `build_xgboost_model()`: Creates XGBRegressor with default params

**Imports:**
- `xgboost.XGBRegressor`

**Referenced By:**
- `src/training/train_xgboost.py`

---

#### `src/models/xgboost/data.py`
**Purpose:** Data preparation for XGBoost (flattened features)  
**Key Functions:**
- `prepare_xgboost_data(df)`: Create lag features and flatten time series

**Imports:**
- Standard: `pandas`, `numpy`

**Referenced By:**
- `src/training/train_xgboost.py`

---

### Training Layer

#### `src/training/train_baseline_lstm.py`
**Purpose:** Main training script for baseline single-site LSTM  
**Key Functions:**
- `set_seeds(seed)`: Ensure reproducibility
- `train_baseline_lstm()`: Full training pipeline

**Workflow:**
1. Load data → Preprocess → Train/test split
2. Build model → Train with EarlyStopping/ModelCheckpoint
3. Evaluate → Save model, scaler, metrics
4. Generate plots (training history, predictions, error distribution)

**Imports:**
- Standard: `sys`, `os`, `numpy`, `tensorflow`
- Config: `config.baseline` (all params)
- Data: `src.data.baseline.data_loader` (load_single_site_csv, preprocess_baseline_data)
- Model: `src.models.baseline.lstm` (build_baseline_lstm)
- Evaluation: `src.evaluation.baseline.evaluate`, `src.evaluation.baseline.plots`

**Referenced By:**
- `web/main.py`

---

#### `src/training/train_embedded_lstm.py`
**Purpose:** Main training script for embedded multi-site LSTM  
**Key Functions:**
- `set_seeds(seed)`: Ensure reproducibility
- `train_embedded_lstm()`: Full training pipeline

**Workflow:**
1. Load all 21 sites → Preprocess with site IDs → Train/test split
2. Build embedded model → Train with callbacks
3. Evaluate → Save model, scaler, metrics
4. Generate plots (training history, predictions, error dist, per-site MAE)

**Imports:**
- Standard: `sys`, `os`, `numpy`, `tensorflow`
- Config: `config.embedded` (all params)
- Data: `src.data.embedded.data_loader` (preprocess_embedded_data)
- Model: `src.models.embedded.lstm` (build_embedded_lstm)
- Evaluation: `src.evaluation.embedded.evaluate`, `src.evaluation.embedded.plots`

**Referenced By:**
- Standalone execution

---

#### `src/training/train_xgboost.py`
**Purpose:** Training script for XGBoost baseline  
**Imports:**
- Config: `config.baseline` (DATA_FILE, TEST_SPLIT)
- Model: `src.models.xgboost.model` (build_xgboost_model)
- Data: `src.models.xgboost.data` (prepare_xgboost_data)

---

#### `src/training/train_all.py`
**Purpose:** Train all models sequentially (orchestrator script)

---

#### `src/training/optuna/optimize_lstm.py`
**Purpose:** Hyperparameter optimization using Optuna framework  
**Imports:**
- `optuna`, `json`

---

### Evaluation Layer

#### `src/evaluation/baseline/evaluate.py`
**Purpose:** Evaluate baseline LSTM on test set  
**Key Functions:**
- `evaluate_baseline()`: Load model, predict, calculate metrics

**Imports:**
- Config: `config.baseline`
- Data: `src.data.baseline.data_loader`
- Standard: `tensorflow`, `numpy`, `pickle`

**Referenced By:**
- `src/training/train_baseline_lstm.py`
- `src/evaluation/comparison/compare_models.py`

---

#### `src/evaluation/baseline/plots.py`
**Purpose:** Generate visualization plots for baseline model  
**Key Functions:**
- `plot_training_history(history, save_dir)`: Loss/MAE curves
- `plot_predictions(y_true, y_pred, save_dir)`: Actual vs predicted
- `plot_error_distribution(y_true, y_pred, save_dir)`: Error histogram

**Imports:**
- `matplotlib.pyplot`, `numpy`

**Referenced By:**
- `src/training/train_baseline_lstm.py`

---

#### `src/evaluation/embedded/evaluate.py`
**Purpose:** Evaluate embedded LSTM on test set  
**Key Functions:**
- `evaluate_embedded()`: Load model, predict, calculate overall and per-site metrics

**Imports:**
- Config: `config.embedded`
- Data: `src.data.embedded.data_loader`
- Standard: `tensorflow`, `numpy`, `pickle`

**Referenced By:**
- `src/training/train_embedded_lstm.py`
- `src/evaluation/comparison/compare_models.py`

---

#### `src/evaluation/embedded/plots.py`
**Purpose:** Generate visualization plots for embedded model  
**Key Functions:**
- `plot_training_history()`, `plot_predictions()`, `plot_error_distribution()`
- `plot_site_mae(site_mae, save_dir)`: Bar chart of MAE per site

**Imports:**
- `matplotlib.pyplot`, `numpy`

**Referenced By:**
- `src/training/train_embedded_lstm.py`

---

#### `src/evaluation/comparison/compare_models.py`
**Purpose:** Compare baseline vs embedded LSTM performance  
**Imports:**
- `src.evaluation.baseline.evaluate`
- `src.evaluation.embedded.evaluate`

---

#### `src/evaluation/naive/naive_24h.py`
**Purpose:** Naive baseline (persistence model) for benchmarking  
**Key Functions:**
- Predict next value = current value (24h ago)

---

#### `src/evaluation/xgboost/evaluate.py`
**Purpose:** Evaluate XGBoost model

---

#### `src/evaluation/xgboost/feature_importance.py`
**Purpose:** Analyze and plot XGBoost feature importance

---

### Inference Layer

#### `src/inference/baseline/predict.py`
**Purpose:** Real-time prediction using baseline LSTM

---

#### `src/inference/embedded/inference_embedded.py`
**Purpose:** Real-time prediction using embedded LSTM

---

#### `src/inference/inference_xgb.py`
**Purpose:** Real-time prediction using XGBoost

---

#### `src/inference/utils_preprocess.py`
**Purpose:** Shared preprocessing utilities for inference

---

### Optimization Layer

#### `src/optimization/optuna_lstm.py`
**Purpose:** Hyperparameter tuning using Optuna  
**Imports:**
- `optuna`

---

### Live Data Layer

#### `src/live/fetch_live_data.py`
**Purpose:** Fetch live weather/solar data for real-time forecasting  
**Key Functions:**
- `prepare_live_sequence()`: Prepare recent data for model input

**Referenced By:**
- `web/server.py`
- `web/main.py`

---

### Web Application Layer

#### `web/main.py`
**Purpose:** CLI entry point for training and inference  
**Imports:**
- `argparse`, `sys`
- Config: `config`
- Training: `src.models.lstm_baseline` (train_lstm)
- Inference: `src.inference.inference_lstm`
- Live: `fetch_live_data`

---

#### `web/server.py`
**Purpose:** FastAPI REST API server for web interface  
**Endpoints:**
- GET `/`: Serve visualization.html
- GET `/api/raw-data`: Return raw PV data
- POST `/api/predict`: Real-time prediction

**Imports:**
- `fastapi` (FastAPI, HTTPException, StaticFiles, FileResponse, CORSMiddleware)
- `contextlib.asynccontextmanager`
- Config: `config`
- Inference: `src.inference.inference_lstm` (load_lstm_model, load_scaler, predict_realtime_lstm)
- Live: `fetch_live_data` (prepare_live_sequence)
- Standard: `os`, `pandas`, `math`

---

#### `web/static/visualization.html`
**Purpose:** Frontend HTML for data visualization

---

#### `web/static/js/visualization.js`
**Purpose:** JavaScript for interactive charts and API calls

---

### Utility Layer

#### `src/utils/get_dataset_dates.py`
**Purpose:** Extract date ranges from dataset  
**Imports:**
- `pandas`, `os`, `glob`

---

### Analysis Scripts

#### `Scripts/PCA.py`
**Purpose:** Principal Component Analysis on features

---

#### `Scripts/Clear_Sky.py`
**Purpose:** Clear sky model calculations

---

## 📊 Data Flow

### Training Pipeline (Baseline)
```
data/raw/pv_01.csv
    ↓
src/data/baseline/data_loader.py (load + preprocess)
    ↓
src/training/train_baseline_lstm.py
    ↓ (builds model)
src/models/baseline/lstm_v1.py
    ↓ (trains)
models/baseline_lstm.keras + baseline_scaler.pkl
    ↓ (evaluates)
src/evaluation/baseline/evaluate.py
    ↓
src/evaluation/baseline/plots.py
    ↓
metrics/baseline_metrics.txt + plots
```

### Training Pipeline (Embedded)
```
data/raw/pv_*.csv (21 sites)
    ↓
src/data/embedded/data_loader.py (load all + add site_id)
    ↓
src/training/train_embedded_lstm.py
    ↓ (builds model)
src/models/embedded/lstm_v1.py
    ↓ (trains)
models/embedded_lstm.keras + embedded_scaler.pkl
    ↓ (evaluates)
src/evaluation/embedded/evaluate.py
    ↓
src/evaluation/embedded/plots.py
    ↓
metrics/embedded_metrics.txt + plots + per-site analysis
```

### Inference Pipeline
```
Live Weather Data
    ↓
src/live/fetch_live_data.py
    ↓
web/server.py (FastAPI endpoint)
    ↓
src/inference/inference_lstm.py (load model + scaler)
    ↓
Prediction (next 3h solar power)
```

---

## 🔑 Key Features

### Features Used (14 total)
1. **Cyclical Time:** hour_of_day_sin/cos, month_of_year_sin/cos
2. **Sun Position:** sunposition_thetaZ, sunposition_solarAzimuth
3. **Clear-Sky Model:** clearsky_diffuse, clearsky_direct, clearsky_global
4. **Weather:** TemperatureAt0, RelativeHumidityAt0, SolarRadiationGlobalAt0, SolarRadiationDirectAt0, SolarRadiationDiffuseAt0, TotalCloudCoverAt0

### Model Comparison
| Model | Scope | Site Handling | Architecture |
|-------|-------|---------------|--------------|
| **Baseline LSTM** | Single-site | Uses only pv_01.csv | Sequential LSTM without site info |
| **Embedded LSTM** | Multi-site | All 21 sites | Functional model with site embedding layer |
| **XGBoost** | Single-site | Traditional ML baseline | Gradient boosting with flattened features |
| **Naive** | Benchmark | Persistence model | y(t+24h) = y(t) |

---

## 📦 Dependencies

From `requirements.txt`:
- tensorflow
- pandas
- numpy
- scikit-learn
- xgboost
- optuna
- fastapi
- matplotlib

---

## 🎯 Entry Points

1. **Train Baseline LSTM:** `python src/training/train_baseline_lstm.py`
2. **Train Embedded LSTM:** `python src/training/train_embedded_lstm.py`
3. **Train XGBoost:** `python src/training/train_xgboost.py`
4. **Hyperparameter Tuning:** `python src/optimization/optuna_lstm.py`
5. **Web Server:** `python web/server.py` or `python web/main.py`
6. **Model Comparison:** `python src/evaluation/comparison/compare_models.py`

---

## 📝 Important Notes for LLM

1. **Two Distinct Model Paradigms:**
   - `baseline/`: Single-site LSTM (simpler, faster)
   - `embedded/`: Multi-site LSTM with learned embeddings (more complex, better generalization)

2. **Config Inheritance:**
   - `embedded.py` imports `FEATURE_COLS_BASELINE` from `baseline.py`
   - Both share same features but different data loading strategies

3. **Model Input Differences:**
   - Baseline: `(batch, time_steps, features)`
   - Embedded: `[(batch, 1), (batch, time_steps, features)]` — separate site_id input

4. **File Naming Convention:**
   - `_v1.py` suffix indicates version 1 of implementation
   - Allows for future versions without breaking imports

5. **Time Resolution:**
   - Raw data: 3-hour intervals
   - TIME_STEPS=8 → 24 hours of history
   - Predicts next 3h power output

6. **Site Indexing:**
   - Sites numbered 1-21 in filenames (pv_01.csv to pv_21.csv)
   - In embedded model, site_id mapped 0-20 (0-indexed for embedding layer)

---

## 🔄 Typical Workflow

1. **Data Exploration** → `Scripts/PCA.py`, `src/utils/get_dataset_dates.py`
2. **Train Models** → `src/training/train_*.py`
3. **Hyperparameter Tuning** → `src/optimization/optuna_lstm.py`
4. **Evaluation** → `src/evaluation/*/evaluate.py`
5. **Comparison** → `src/evaluation/comparison/compare_models.py`
6. **Deployment** → `web/server.py` (FastAPI + static frontend)
7. **Real-time Inference** → API endpoint `/api/predict`

---

*This index is designed for LLM consumption to quickly understand repository structure, file purposes, and dependency relationships.*
