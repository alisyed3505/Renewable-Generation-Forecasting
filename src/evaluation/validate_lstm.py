"""
LSTM Model Validation Script

Validates the trained LSTM model by:
1. Loading real data from the dataset
2. Running predictions across multiple sliding windows
3. Comparing predictions vs actual values
4. Calculating comprehensive accuracy metrics
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.inference.inference_lstm import load_lstm_model, load_scaler
from config import FEATURE_COLS


def load_validation_data(data_path='data/raw/pv_01.csv'):
    """
    Loads dataset for validation.
    
    Returns:
        DataFrame with all required features and power_normed target
    """
    try:
        df = pd.read_csv(data_path, delimiter=';')
        
        # Drop unnamed columns if present
        if df.columns[-1].startswith('Unnamed'):
            df = df.iloc[:, :-1]
            
        # Add site_id from filename
        basename = os.path.basename(data_path)
        site_id_str = basename.replace('pv_', '').replace('.csv', '')
        df['site_id'] = int(site_id_str) if site_id_str.isdigit() else 1
        
        print(f"Loaded {len(df)} rows from {data_path}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def validate_lstm_model(model, scaler, data_path='data/raw/pv_01.csv', 
                        time_steps=24, num_samples=500, step_size=24,
                        use_test_set=True):
    """
    Validates the LSTM model using sliding window evaluation.
    
    Args:
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler
        data_path: Path to the CSV data file
        time_steps: Number of hours in each input window (24)
        num_samples: Number of prediction samples to evaluate
        step_size: How many rows to skip between samples (avoids overlap)
        use_test_set: If True, only evaluate on unseen test data (last 20%)
    
    Returns:
        dict: Validation metrics and sample predictions
    """
    # Load data
    df = load_validation_data(data_path)
    if df is None:
        return None
    
    # Ensure all required features exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            if col == 'site_id':
                df[col] = 1
            else:
                print(f"Warning: Missing column '{col}', filling with 0")
                df[col] = 0
    
    # Store predictions and actuals
    predictions = []
    actuals = []
    sample_details = []
    
    # Determine starting point for validation
    if use_test_set:
        # Model was trained on first 80%, so test on last 20%
        start_offset = int(len(df) * 0.8)
        print(f"\n📊 Validating on TEST SET (rows {start_offset}+ of {len(df)})")
    else:
        start_offset = 0
        print(f"\n📊 Validating on ENTIRE dataset ({len(df)} rows)")
    
    # Calculate indices for sampling
    max_start_idx = len(df) - time_steps - 1
    sample_indices = range(start_offset, min(max_start_idx, start_offset + num_samples * step_size), step_size)
    
    print(f"Running validation on {len(list(sample_indices))} samples...")
    print("-" * 50)
    
    for i, start_idx in enumerate(sample_indices):
        # Get the 24-hour input window
        window = df.iloc[start_idx:start_idx + time_steps].copy()
        
        # Get the ACTUAL power output for the next hour (what we're predicting)
        actual_value = df.iloc[start_idx + time_steps]['power_normed']
        
        # Select and order features
        window_features = window[FEATURE_COLS]
        
        # Normalize using the saved scaler
        X_scaled = scaler.transform(window_features)
        
        # Reshape for LSTM: (1, time_steps, features)
        X_input = X_scaled.reshape(1, time_steps, len(FEATURE_COLS))
        
        # Predict
        pred = model.predict(X_input, verbose=0)[0][0]
        pred = max(0, min(1, pred))  # Clamp to [0, 1]
        
        predictions.append(pred)
        actuals.append(actual_value)
        
        # Save some sample details for display
        if len(sample_details) < 10:
            # Get hour info if available
            hour = window.iloc[-1].get('hour_of_day', 'N/A')
            sample_details.append({
                'sample': i + 1,
                'hour': hour,
                'actual': actual_value,
                'predicted': pred,
                'error': abs(pred - actual_value)
            })
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1} samples...")
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Additional metrics
    errors = np.abs(predictions - actuals)
    max_error = np.max(errors)
    min_error = np.min(errors)
    median_error = np.median(errors)
    
    # Percentage within thresholds
    within_5_pct = np.mean(errors < 0.05) * 100
    within_10_pct = np.mean(errors < 0.10) * 100
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'max_error': max_error,
        'min_error': min_error,
        'median_error': median_error,
        'within_5_pct': within_5_pct,
        'within_10_pct': within_10_pct,
        'num_samples': len(predictions),
        'sample_details': sample_details
    }
    
    return metrics


def print_results(metrics):
    """
    Prints validation results in a formatted way.
    """
    print("\n" + "=" * 60)
    print("          LSTM MODEL VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 ACCURACY METRICS ({metrics['num_samples']} samples tested)")
    print("-" * 40)
    print(f"  RMSE (Root Mean Squared Error):  {metrics['rmse']:.4f}")
    print(f"  MAE (Mean Absolute Error):       {metrics['mae']:.4f}")
    print(f"  R² Score:                        {metrics['r2_score']:.4f}")
    
    print(f"\n📏 ERROR DISTRIBUTION")
    print("-" * 40)
    print(f"  Minimum Error:                   {metrics['min_error']:.4f}")
    print(f"  Median Error:                    {metrics['median_error']:.4f}")
    print(f"  Maximum Error:                   {metrics['max_error']:.4f}")
    
    print(f"\n🎯 PREDICTION ACCURACY")
    print("-" * 40)
    print(f"  Within 5% of actual:             {metrics['within_5_pct']:.1f}%")
    print(f"  Within 10% of actual:            {metrics['within_10_pct']:.1f}%")
    
    print(f"\n📋 SAMPLE PREDICTIONS (first 10)")
    print("-" * 60)
    print(f"  {'#':<4} {'Hour':<6} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
    print("-" * 60)
    
    for sample in metrics['sample_details']:
        print(f"  {sample['sample']:<4} {str(sample['hour']):<6} "
              f"{sample['actual']:<10.4f} {sample['predicted']:<10.4f} "
              f"{sample['error']:<10.4f}")
    
    print("=" * 60)


def save_metrics(metrics, output_path='models/metrics/validation_metrics.txt'):
    """
    Saves validation metrics to file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("LSTM Model Validation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Samples Tested: {metrics['num_samples']}\n\n")
        f.write(f"RMSE: {metrics['rmse']:.6f}\n")
        f.write(f"MAE: {metrics['mae']:.6f}\n")
        f.write(f"R² Score: {metrics['r2_score']:.6f}\n\n")
        f.write(f"Max Error: {metrics['max_error']:.6f}\n")
        f.write(f"Min Error: {metrics['min_error']:.6f}\n")
        f.write(f"Median Error: {metrics['median_error']:.6f}\n\n")
        f.write(f"Within 5%: {metrics['within_5_pct']:.1f}%\n")
        f.write(f"Within 10%: {metrics['within_10_pct']:.1f}%\n")
    
    print(f"\n✅ Metrics saved to: {output_path}")


if __name__ == "__main__":
    print("🔍 LSTM Model Validation")
    print("=" * 60)
    
    # Load model and scaler
    model = load_lstm_model()
    scaler = load_scaler()
    
    if model is None or scaler is None:
        print("❌ Error: Model or scaler not found. Please train the model first.")
        print("   Run: python src/training/train_lstm.py")
        sys.exit(1)
    
    print("✅ Model and scaler loaded successfully")
    
    # Run validation
    metrics = validate_lstm_model(
        model=model,
        scaler=scaler,
        data_path='data/raw/pv_01.csv',
        time_steps=24,
        num_samples=500,  # Test on 500 different windows
        step_size=24      # Skip 24 rows between samples (non-overlapping days)
    )
    
    if metrics:
        print_results(metrics)
        save_metrics(metrics)
    else:
        print("❌ Validation failed")
