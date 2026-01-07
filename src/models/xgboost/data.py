import numpy as np
from src.data.baseline.data_loader import load_single_site_csv

def prepare_xgboost_data(
    data_file: str,
    test_split: float,
):
    """
    Prepare flat (non-sequential) features for XGBoost.

    Uses current-time features only: X(t) -> y(t)

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    df = load_single_site_csv(data_file)

    # Target = last column (same as baseline)
    y = df.iloc[:, -1].values.astype(np.float32)

    # Features = all except target
    X = df.iloc[:, :-1].values.astype(np.float32)

    split_idx = int(len(X) * (1 - test_split))

    X_train = X[:split_idx]
    X_test = X[split_idx:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    return X_train, X_test, y_train, y_test
