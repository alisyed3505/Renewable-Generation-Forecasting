# src/models/xgboost/model_v2.py
"""
XGBoost v2 - Optuna-Optimized Model

This version uses hyperparameters optimized by Opt una based on 50 trials.
Key differences from v1:
- Deeper trees: max_depth 6 → 9 (+50%)
- Higher min_child_weight: 1 → 6 (more conservative splits)
- Lower learning rate: 0.1 → 0.0297 (-70%)
- More estimators: 100 → 130 (+30%)
- Added regularization: L1 (alpha=3.35), L2 (lambda=2.98)
-Enabled subsampling: 0.846 (vs 1.0)
- Enabled column sampling: 0.880 (vs 1.0)

Optuna optimization results:
- Best validation RMSE: 0.0649 (achieved through 50 trials)
- Optimized for validation set performance
- See models/optuna/xgboost_v2/ for full optimization history
"""

import xgboost as xgb


def build_xgboost_v2(
    max_depth=9,
    min_child_weight=6,
    learning_rate=0.029710,
    n_estimators=130,
    gamma=0.007025,
    subsample=0.845905,
    colsample_bytree=0.879990,
    reg_alpha=3.353364,
    reg_lambda=2.981327,
    random_state=42
):
    """
    Build XGBoost v2 model with Optuna-optimized hyperparameters.
    
    Args:
        max_depth: Maximum tree depth (Optuna best: 9)
        min_child_weight: Minimum sum of instance weight (Optuna best: 6)
        learning_rate: Step size shrinkage (Optuna best: 0.0297)
        n_estimators: Number of boosting rounds (Optuna best: 130)
        gamma: Minimum loss reduction for split (Optuna best: 0.007)
        subsample: Subsample ratio of training instances (Optuna best: 0.846)
        colsample_bytree: Subsample ratio of columns (Optuna best: 0.880)
        reg_alpha: L1 regularization term (Optuna best: 3.353)
        reg_lambda: L2 regularization term (Optuna best: 2.981)
        random_state: Random seed for reproducibility
        
    Returns:
        XGBRegressor model instance
    """
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        random_state=random_state,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        verbosity=0
    )
    
    return model
