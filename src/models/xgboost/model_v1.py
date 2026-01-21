import xgboost as xgb


def build_xgboost_model():
    """
    Build a baseline XGBoost regressor with sensible defaults.
    No hyperparameter tuning at this stage.
    """

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    return model
