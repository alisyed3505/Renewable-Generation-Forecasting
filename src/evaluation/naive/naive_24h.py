import numpy as np


def naive_24h_predict(y: np.ndarray, time_steps: int):
    """
    24h-ago naive baseline:
    y_hat(t) = y(t - time_steps)

    Parameters
    ----------
    y : np.ndarray
        Ground truth values (1D array)
    time_steps : int
        Number of steps corresponding to 24 hours (8 for 3h resolution)

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    """
    if len(y) <= time_steps:
        raise ValueError("Not enough data for naive prediction")

    y_true = y[time_steps:]
    y_pred = y[:-time_steps]

    return y_true, y_pred
