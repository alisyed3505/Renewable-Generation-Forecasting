"""
Model Comparison Script

Compares the performance of:
1. Baseline LSTM (treats site_id as numeric feature)
2. Embedded LSTM (uses embedding layer for site_id)

Both models are evaluated on the same test data for fair comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import joblib

import config.embedded


def load_models():
    """Loads both baseline and embedded models."""
    models = {}
    
    # Baseline model
    try:
        baseline_model = load_model(config.MODEL_FILE, compile=False)
        baseline_scaler = joblib.load(config.SCALER_FILE)
        models['baseline'] = {
            'model': baseline_model,
            'scaler': baseline_scaler,
            'name': 'Baseline LSTM'
        }
        print(f"✅ Loaded Baseline LSTM from {config.MODEL_FILE}")
    except Exception as e:
        print(f"⚠️ Could not load baseline model: {e}")
    
    # Embedded model
    try:
        embedded_model = load_model(config.EMBEDDED_MODEL_FILE, compile=False)
        embedded_scaler = joblib.load(config.EMBEDDED_SCALER_FILE)
        models['embedded'] = {
            'model': embedded_model,
            'scaler': embedded_scaler,
            'name': 'Embedded LSTM'
        }
        print(f"✅ Loaded Embedded LSTM from {config.EMBEDDED_MODEL_FILE}")
    except Exception as e:
        print(f"⚠️ Could not load embedded model: {e}")
    
    return models


def evaluate_baseline(model_info, test_data_path='data/raw/pv_01.csv', 
                      time_steps=24, num_samples=100):
    """Evaluates baseline model on test data."""
    from config import FEATURE_COLS
    
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Load test data
    df = pd.read_csv(test_data_path, delimiter=';')
    if df.columns[-1].startswith('Unnamed'):
        df = df.iloc[:, :-1]
    df['site_id'] = 1  # Default site for baseline
    
    # Start from test set (last 20%)
    start_offset = int(len(df) * 0.8)
    
    predictions = []
    actuals = []
    
    for i in range(num_samples):
        start_idx = start_offset + (i * 24)
        if start_idx + time_steps >= len(df):
            break
        
        window = df.iloc[start_idx:start_idx + time_steps].copy()
        actual = df.iloc[start_idx + time_steps]['power_normed']
        
        # Ensure all features exist
        for col in FEATURE_COLS:
            if col not in window.columns:
                window[col] = 0
        
        X = scaler.transform(window[FEATURE_COLS])
        X = X.reshape(1, time_steps, len(FEATURE_COLS))
        
        pred = model.predict(X, verbose=0)[0][0]
        pred = max(0, min(1, pred))
        
        predictions.append(pred)
        actuals.append(actual)
    
    return np.array(predictions), np.array(actuals)


def evaluate_embedded(model_info, test_data_path='data/raw/pv_*.csv',
                      time_steps=24, num_samples=100):
    """Evaluates embedded model on multi-site test data."""
    from config import FEATURE_COLS_EMBEDDED
    import glob
    
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Load all sites
    all_dfs = []
    for file_path in glob.glob(test_data_path):
        df = pd.read_csv(file_path, delimiter=';')
        if df.columns[-1].startswith('Unnamed'):
            df = df.iloc[:, :-1]
        
        basename = os.path.basename(file_path)
        site_id_str = basename.replace('pv_', '').replace('.csv', '')
        df['site_id'] = int(site_id_str) if site_id_str.isdigit() else 0
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Start from test set (last 20%)
    start_offset = int(len(full_df) * 0.8)
    
    predictions = []
    actuals = []
    samples_collected = 0
    
    for i in range(start_offset, len(full_df) - time_steps - 1, 24):
        if samples_collected >= num_samples:
            break
        
        window = full_df.iloc[i:i + time_steps]
        
        # Only use if all timesteps are from same site
        if len(window['site_id'].unique()) != 1:
            continue
        
        site_id = window['site_id'].iloc[0]
        actual = full_df.iloc[i + time_steps]['power_normed']
        
        # Prepare features
        X_feat = scaler.transform(window[FEATURE_COLS_EMBEDDED])
        X_feat = X_feat.reshape(1, time_steps, len(FEATURE_COLS_EMBEDDED))
        X_site = np.array([[site_id]])
        
        pred = model.predict([X_site, X_feat], verbose=0)[0][0]
        pred = max(0, min(1, pred))
        
        predictions.append(pred)
        actuals.append(actual)
        samples_collected += 1
    
    return np.array(predictions), np.array(actuals)


def calculate_metrics(predictions, actuals):
    """Calculates performance metrics."""
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    errors = np.abs(predictions - actuals)
    within_5 = np.mean(errors < 0.05) * 100
    within_10 = np.mean(errors < 0.10) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Within 5%': within_5,
        'Within 10%': within_10,
        'Samples': len(predictions)
    }


def compare_models():
    """Main comparison function."""
    print("=" * 70)
    print("         MODEL COMPARISON: Baseline vs Embedded LSTM")
    print("=" * 70)
    
    models = load_models()
    
    if len(models) < 2:
        print("\n⚠️ Need both models trained for comparison!")
        print("   Run: python src/training/train_lstm.py")
        print("   Run: python src/training/train_embedded_lstm.py")
        return
    
    results = {}
    
    # Evaluate baseline
    print("\n📊 Evaluating Baseline LSTM...")
    preds, actuals = evaluate_baseline(models['baseline'], num_samples=100)
    results['Baseline LSTM'] = calculate_metrics(preds, actuals)
    
    # Evaluate embedded
    print("📊 Evaluating Embedded LSTM...")
    preds, actuals = evaluate_embedded(models['embedded'], num_samples=100)
    results['Embedded LSTM'] = calculate_metrics(preds, actuals)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("                    PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<15} {'Baseline LSTM':<18} {'Embedded LSTM':<18} {'Winner'}")
    print("-" * 70)
    
    for metric in ['RMSE', 'MAE', 'R²', 'Within 5%', 'Within 10%']:
        baseline_val = results['Baseline LSTM'][metric]
        embedded_val = results['Embedded LSTM'][metric]
        
        # Determine winner (lower is better for RMSE/MAE, higher for others)
        if metric in ['RMSE', 'MAE']:
            winner = 'Embedded' if embedded_val < baseline_val else 'Baseline'
        else:
            winner = 'Embedded' if embedded_val > baseline_val else 'Baseline'
        
        if metric in ['Within 5%', 'Within 10%']:
            print(f"{metric:<15} {baseline_val:<18.1f} {embedded_val:<18.1f} {winner}")
        else:
            print(f"{metric:<15} {baseline_val:<18.4f} {embedded_val:<18.4f} {winner}")
    
    print("-" * 70)
    print(f"{'Samples':<15} {results['Baseline LSTM']['Samples']:<18} {results['Embedded LSTM']['Samples']:<18}")
    print("=" * 70)
    
    # Save comparison results
    with open('models/metrics/model_comparison.txt', 'w') as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 50 + "\n\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
    
    print("\n✅ Comparison saved to: models/metrics/model_comparison.txt")


if __name__ == "__main__":
    compare_models()
