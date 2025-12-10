import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import glob
import os

def load_data(file_pattern='data/raw/pv_*.csv'):
    """
    Loads and preprocesses the solar power dataset from multiple files.
    """
    try:
        # finding the paths if they exist in the pattern provided in file_pattern variable/parameter
        # and returns the path in a list
        files = glob.glob(file_pattern)
        print(f"Found {len(files)} dataset files: {files}")
        
        # initialize an empty list to store dataframes
        all_dfs = []
        
        # loop through each file path
        for file_path in files:
            # print(f"Loading {file_path}...")
            df = pd.read_csv(file_path, delimiter=';')
            
            # If pandas accidentally created a ghost column 'Unnamed' at the end because of a trailing semicolon, delete it
            if df.columns[-1].startswith('Unnamed'):
                # iloc is used to select rows and columns by integer-location based indexing
                # first parameter is for rows and second is for columns
                # :-1 means select all rows and all columns except the last one
                df = df.iloc[:, :-1]
            
            # Extract site_id from filename (e.g., pv_12.csv -> 12)
            try:
                # Assumes format .../pv_XX.csv
                basename = os.path.basename(file_path)
                site_id_str = basename.replace('pv_', '').replace('.csv', '')
                site_id = int(site_id_str)
            except ValueError:
                site_id = 0 # Default if pattern doesn't match
                
            df['site_id'] = site_id
            
            all_dfs.append(df)
        
        # Check if any files were loaded
        if not all_dfs:
            print("No files found!")
            return None, None
            
        # Combine all data
        full_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total Combined Rows: {len(full_df)}")
        
        # Ensure cyclical encoding exists (dataset already contains them)
        required_time_cols = [
            'hour_of_day_sin', 'hour_of_day_cos',
            'month_of_year_sin', 'month_of_year_cos'
        ]

        for col in required_time_cols:
            if col not in full_df.columns:
                raise ValueError(f"Missing required time feature: {col}")

        # Import config to get valid features
        from config import FEATURE_COLS
        
        # Get available columns
        available_cols = [c for c in FEATURE_COLS if c in full_df.columns]
        
        print(f"Using {len(available_cols)} features: {available_cols}")
        
        # Select features and target
        # Features are the input variables (weather, time, etc.)
        # Target is the output variable (power output)
        X = full_df[available_cols]
        y = full_df['power_normed']
        
        # Fill missing values with forward fill and backward fill
        # This handles missing data by using the nearest available values
        # For example, if a value is missing, it uses the previous value (forward fill)
        # If the previous value is also missing, it uses the next available value (backward fill)
        # This is a common approach for time series data
        X = X.ffill().bfill()
        
        # Fill missing target values with 0
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
        
    #PREDICTION::::::
    # Optimization: Use numpy striding or simple loop
    # For very large datasets, this loop can be slow. 
    # But for ~100k rows it's acceptable.
    # PREDICTION BASIS: Time-Series Sequencing
    # We slide a window of size 'time_steps' (24) across the data.
    # Input (X): Rows i to i+24 (The past 24 hours)
    # Target (y): Row i+24 (The power output at the very next hour)
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
        
    return np.array(Xs), np.array(ys)

"""
PREPROCESSING::::::
parameters:
    X: Features (weather, time, etc.)
    y: Target (power output)
    time_steps: Number of past hours to look at
    scaler_path: Path to save the scaler
returns:
    X_seq: Input sequences for LSTM
    y_seq: Target sequences for LSTM
    scaler: Scaler object for later use

Note: Scaler is used to normalize the data and it is important for the LSTM model to perform well. 
It works by subtracting the mean of the data from each value and then dividing by the standard deviation of the data. 
For example:
    X = [1, 2, 3, 4, 5]
    mean = 3
    std = 1.58
    normalized_X = [(1-3)/1.58, (2-3)/1.58, (3-3)/1.58, (4-3)/1.58, (5-3)/1.58]
"""
def preprocess_for_lstm(X, y, time_steps=24, scaler_path='models/scaler.pkl'):
    """
    Normalizes data and creates sequences.
    """
    print("Normalizing data...")
    # using minmaxscaler to normalize the data between 0 and 1 
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    joblib.dump(scaler, scaler_path)
    
    print("Creating sequences (this may take a moment)...")
    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)
    
    return X_seq, y_seq, scaler

def split_data_lstm(X, y, test_size=0.2):
    """
    Splits data into training and testing sets.
    parameters:
        X: Features (weather, time, etc.)
        y: Target (power output)
        test_size: Proportion of data to use for testing
    returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
    """
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X, y = load_data()
    if X is not None:
        print("Data loaded.")
