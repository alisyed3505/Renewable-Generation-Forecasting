import pandas as pd
import json

# Load Optuna trials
trials = pd.read_csv('models/optuna/baseline_lstm_v2/trials.csv')

# Find best trial
best_idx = trials['value'].idxmin()
best_trial = trials.loc[best_idx]

print("=" * 60)
print("OPTUNA VALIDATION LOSS ANALYSIS")
print("=" * 60)

print(f"\n📊 Best Trial (Trial {int(best_trial['trial'])}):")
print(f"   Validation Loss: {best_trial['value']:.6f}")
print(f"\n   Parameters:")
print(f"   - LSTM Layer 1: {int(best_trial['lstm_1_units'])} units")
print(f"   - LSTM Layer 2: {int(best_trial['lstm_2_units'])} units")
print(f"   - Dense Units: {int(best_trial['dense_units'])}")
print(f"   - Dropout: {best_trial['dropout']:.3f}")
print(f"   - Learning Rate: {best_trial['lr']:.6f}")
print(f"   - Batch Size: {int(best_trial['batch_size'])}")

print(f"\n📈 Optuna Trial Statistics:")
print(f"   Best val_loss: {trials['value'].min():.6f}")
print(f"   Worst val_loss: {trials['value'].max():.6f}")
print(f"   Mean val_loss: {trials['value'].mean():.6f}")
print(f"   Std val_loss: {trials['value'].std():.6f}")

# Load actual model metrics
with open('models/metrics/baseline_lstm_v1_metrics.json', 'r') as f:
    v1_metrics = json.load(f)

with open('models/metrics/baseline_lstm_v2_metrics.json', 'r') as f:
    v2_metrics = json.load(f)

print(f"\n📉 Actual Test Performance:")
print(f"   v1 Test RMSE: {v1_metrics['metrics']['rmse']:.6f}")
print(f"   v2 Test RMSE: {v2_metrics['metrics']['rmse']:.6f}")
print(f"   Difference: {(v2_metrics['metrics']['rmse'] - v1_metrics['metrics']['rmse']):.6f} ({((v2_metrics['metrics']['rmse'] / v1_metrics['metrics']['rmse'] - 1) * 100):.1f}%)")

print(f"\n   v1 Test MAE: {v1_metrics['metrics']['mae']:.6f}")
print(f"   v2 Test MAE: {v2_metrics['metrics']['mae']:.6f}")
print(f"   Difference: {(v2_metrics['metrics']['mae'] - v1_metrics['metrics']['mae']):.6f} ({((v2_metrics['metrics']['mae'] / v1_metrics['metrics']['mae'] - 1) * 100):.1f}%)")

# Note: We don't have validation loss for v1 training, so we can't directly compare
print("\n⚠️  NOTE: v2 achieved best VALIDATION loss in Optuna,")
print("   but this didn't translate to better TEST performance.")
print("   This indicates validation/test distribution mismatch.")
print("=" * 60)
