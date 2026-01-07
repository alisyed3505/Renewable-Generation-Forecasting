# src/data/baseline/data_loader.py
"""
Data loader for Baseline LSTM (Single-Site Only)

This loader:
- Loads exactly ONE PV site CSV
- Enforces feature correctness
- Prevents multi-site contamination
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

from config.baseline import FEATURE_COLS_BASELINE, TIME_STEPS


def load_single_site_csv(csv_path: str) -> pd.DataFrame:
    """
    Load and validate a single-site PV dataset.

    Fails loudly if:
    - file does not exist
    - required features are missing
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Baseline data file not found: {csv_path}")

    df = pd.read_csv(csv_path, delimiter=";")

    # Remove accidental trailing column
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]

    # Check required features
    missing = [c for c in FEATURE_COLS_BASELINE if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required baseline features: {missing}")

    # Target column check
    if "power_normed" not in df.columns:
        raise ValueError("Target column 'power_normed' not found")

    # Sort by time index if available
    if "time_idx" in df.columns:
        df = df.sort_values("time_idx")

    return df


def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int):
    """
    Create LSTM sequences (X[t:t+24] -> y[t+24]).
    """
    X_seq, y_seq = [], []

    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])

    return np.array(X_seq), np.array(y_seq)


def preprocess_baseline_data(
    df: pd.DataFrame,
    scaler_path: str,
    time_steps: int = TIME_STEPS,
):
    """
    Scale features and create LSTM sequences.
    """
    X = df[FEATURE_COLS_BASELINE].ffill().bfill()
    y = df["power_normed"].fillna(0).values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    X_seq, y_seq = create_sequences(X_scaled, y, time_steps)

    return X_seq, y_seq, scaler
