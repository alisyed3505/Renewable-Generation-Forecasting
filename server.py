from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import config
from inference_lstm import load_lstm_model, load_scaler, predict_realtime_lstm
from fetch_live_data import prepare_live_sequence
import os
import pandas as pd
import math

# Load model and scaler on startup
model = None
scaler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler
    print("Loading model and scaler...")
    model = load_lstm_model(config.MODEL_FILE)
    scaler = load_scaler(config.SCALER_FILE)
    if model:
        print("Model loaded successfully.")
    else:
        print("Model failed to load.")
    yield
    # Clean up if needed
    print("Shutting down...")

app = FastAPI(title="Solar Power Forecasting API", lifespan=lifespan)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse('static/index.html')

@app.get("/api/health")
def health_check():
    return {"status": "online", "service": "Solar Power Forecasting"}

@app.get("/api/data")
def get_data(page: int = 1, limit: int = 50):
    """
    Returns paginated data from the dataset.
    """
    try:
        # Load data (caching this in memory would be better for production, but reading file is fine for now)
        df = pd.read_csv(config.DATA_FILE, delimiter=';')
        
        # Handle Unnamed column if present
        if df.columns[-1].startswith('Unnamed'):
            df = df.iloc[:, :-1]
            
        total_rows = len(df)
        total_pages = math.ceil(total_rows / limit)
        
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        
        data = df.iloc[start_idx:end_idx].to_dict(orient='records')
        
        return {
            "data": data,
            "total": total_rows,
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "column_metadata": config.COLUMN_METADATA
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/live")
def predict_live():
    """
    Fetches live weather data and returns a forecast.
    """
    global model, scaler
    if not model or not scaler:
        # Try reloading
        model = load_lstm_model(config.MODEL_FILE)
        scaler = load_scaler(config.SCALER_FILE)
        if not model:
            raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    try:
        sequence = prepare_live_sequence()
        if not sequence:
            raise HTTPException(status_code=500, detail="Failed to fetch live data.")
            
        prediction = predict_realtime_lstm(model, scaler, sequence, time_steps=config.TIME_STEPS)
        
        # Determine if we used dummy data (hacky check, but useful for UI)
        # In a real app, prepare_live_sequence should return metadata
        is_dummy = sequence[0]['SolarRadiationGlobalAt0'] == 500 and sequence[0]['hour_of_day'] == 6 # Heuristic based on dummy logic
        
        return {
            "predicted_power_normalized": float(prediction),
            "predicted_power_kw": float(prediction) * 5.0, # Assuming 5kW system for demo
            "source": "dummy_fallback" if is_dummy else "live_api",
            "timestamp": "Next Hour"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    """
    Returns the latest training metrics.
    """
    try:
        if os.path.exists('lstm_metrics.txt'):
            with open('lstm_metrics.txt', 'r') as f:
                lines = f.readlines()
                stats = {}
                for line in lines:
                    key, val = line.strip().split(': ')
                    stats[key] = float(val)
                return stats
        else:
            return {"RMSE": 0.0, "MAE": 0.0}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
