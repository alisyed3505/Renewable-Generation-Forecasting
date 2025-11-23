import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from data_loader import load_data, split_data

def train_model(data_path='pv_01.csv', model_path='solar_model.pkl'):
    """
    Trains a Random Forest Regressor and saves the model.
    """
    print("Loading data...")
    X, y = load_data(data_path)
    
    if X is None:
        print("Failed to load data.")
        return

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize Random Forest
    # n_jobs=-1 uses all processors
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("Training model (this may take a moment)...")
    rf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = rf.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save metrics to file
    with open('model_metrics.txt', 'w') as f:
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
    
    print(f"Saving model to {model_path}...")
    joblib.dump(rf, model_path)
    print("Done!")

if __name__ == "__main__":
    train_model()
