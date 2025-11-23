import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import glob
import os

def load_data(file_pattern='pv_*.csv'):
    """
    Loads and preprocesses the solar power dataset from multiple files.
    """
    try:
        files = glob.glob(file_pattern)
        print(f"Found {len(files)} dataset files: {files}")
        
        all_dfs = []
        for file_path in files:
            # print(f"Loading {file_path}...")
            df = pd.read_csv(file_path, delimiter=';')
            
            if df.columns[-1].startswith('Unnamed'):
                df = df.iloc[:, :-1]
            
            # Add a site_id column if we want to distinguish (optional for now)
            # df['site_id'] = file_path
            
            all_dfs.append(df)
            
        if not all_dfs:
            print("No files found!")
            return None, None
            
        # Combine all data
        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total Combined Rows: {len(full_df)}")
            
        feature_cols = [
            'hour_of_day', 'month_of_year',
            'sunposition_thetaZ', 'sunposition_solarAzimuth', 
            'clearsky_diffuse', 'clearsky_direct', 'clearsky_global',
            'TemperatureAt0', 'RelativeHumidityAt0', 
            'SolarRadiationGlobalAt0', 'SolarRadiationDirectAt0', 'SolarRadiationDiffuseAt0',
            'TotalCloudCoverAt0', 'LowerWindSpeed', 'LowerWindDirection'
        ]
        
        available_cols = [c for c in feature_cols if c in full_df.columns]
        
        X = full_df[available_cols]
        y = full_df['power_normed']
        
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(0)
        
        return X, y

    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def create_sequences(X, y, time_steps=24):
    """
    Creates sequences for LSTM training.
    """
    Xs, ys = [], []
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
        
    # Optimization: Use numpy striding or simple loop
    # For very large datasets, this loop can be slow. 
    # But for ~100k rows it's acceptable.
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
        
    return np.array(Xs), np.array(ys)

def preprocess_for_lstm(X, y, time_steps=24, scaler_path='scaler.pkl'):
    """
    Normalizes data and creates sequences.
    """
    print("Normalizing data...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, scaler_path)
    
    print("Creating sequences (this may take a moment)...")
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    
    return X_seq, y_seq, scaler

def split_data_lstm(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_data()
    if X is not None:
        print("Data loaded.")
