import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import json
import joblib
from datetime import datetime

import optuna
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.data.baseline.data_loader import load_single_site_csv
from config.baseline import DATA_FILE, TIME_STEPS, TEST_SPLIT


# ============================================================
# Utility: Create sequences
# ============================================================
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps].flatten())  # Flatten for XGBoost
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


# ============================================================
# Optuna Objective Function
# ============================================================
def objective(trial):
    """
    Optuna objective function for XGBoost hyperparameter tuning.
    
    Optimizes:
    - Tree structure: max_depth, min_child_weight
    - Learning: learning_rate (eta), n_estimators
    - Regularization: gamma, subsample, colsample_bytree
    - L1/L2: reg_alpha, reg_lambda
    """
    # -----------------------------
    # Hyperparameters to tune
    # -----------------------------
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
        'seed': 42,
        
        # Tree structure
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        
        # Learning
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        
        # Regularization
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        
        # L1/L2 regularization
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }

    # -----------------------------
    # Load & preprocess data
    # -----------------------------
    df = load_single_site_csv(DATA_FILE)
    
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    
    X_seq, y_seq = create_sequences(X, y, TIME_STEPS)
    
    split = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    # -----------------------------
    # Train XGBoost
    # -----------------------------
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predict and evaluate
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, 1)
    
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse


# ============================================================
# Main entry point
# ============================================================
def main():
    print("=" * 60)
    print("   XGBOOST OPTUNA OPTIMIZATION")
    print("=" * 60)
    
    # --------------------------------------------------
    # Create Optuna study
    # --------------------------------------------------
    study = optuna.create_study(direction="minimize")
    
    print("\n🔍 Starting hyperparameter optimization...")
    print(f"   Number of trials: 50")
    print(f"   Objective: Minimize validation RMSE")
    
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n" + "=" * 60)
    print("   OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"\n📊 Best trial (Trial {study.best_trial.number}):")
    print(f"   Validation RMSE: {study.best_value:.6f}")
    print(f"\n   Best hyperparameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"   - {key}: {value:.6f}")
        else:
            print(f"   - {key}: {value}")

    # --------------------------------------------------
    # Output directories
    # --------------------------------------------------
    STUDY_DIR = "models/optuna/xgboost_v2"
    os.makedirs(STUDY_DIR, exist_ok=True)
    os.makedirs(f"{STUDY_DIR}/plots", exist_ok=True)

    # --------------------------------------------------
    # Save Optuna study (FULL history)
    # --------------------------------------------------
    joblib.dump(study, os.path.join(STUDY_DIR, "study.pkl"))

    # --------------------------------------------------
    # Save best parameters
    # --------------------------------------------------
    with open(os.path.join(STUDY_DIR, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    # --------------------------------------------------
    # Export all trials to CSV (for analysis & plots)
    # --------------------------------------------------
    trials_data = []
    for trial in study.trials:
        row = {
            "trial": trial.number,
            "value": trial.value,
            **trial.params
        }
        trials_data.append(row)

    import pandas as pd
    df_trials = pd.DataFrame(trials_data)
    df_trials.to_csv(os.path.join(STUDY_DIR, "trials.csv"), index=False)

    print(f"\n✅ Optuna study saved to: {STUDY_DIR}")
    print("✅ Best parameters saved to: best_params.json")
    print("✅ Trial history exported to: trials.csv")
    
    print("\n📈 To generate plots, run:")
    print("   python src/training/optuna/plot_optuna_history_xgboost.py")


if __name__ == "__main__":
    main()
