"""
Data loader for Embedded LSTM (Multi-Site, Site Embeddings)

Guarantees:
- Same features as baseline
- Fixed 24h window (8 timesteps)
- No cross-site sequences
"""

import glob
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

from config.embedded import TIME_STEPS
from config.baseline import FEATURE_COLS_BASELINE


def load_all_sites(data_glob):
    """
    Load all PV site CSVs and attach site_id.
    """
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {data_glob}")

    dfs = []

    for site_id, path in enumerate(files):
        df = pd.read_csv(path, delimiter=";")

        # Drop unnamed trailing column if present
        if df.columns[-1].startswith("Unnamed"):
            df = df.iloc[:, :-1]

        df["site_id"] = site_id
        dfs.append(df)

    return dfs


def create_site_sequences(df, time_steps):
    """
    Create sequences for ONE site only.
    """
    X = df[FEATURE_COLS_BASELINE].ffill().bfill().values
    y = df["power_normed"].fillna(0).values
    site_id = df["site_id"].iloc[0]

    X_seq, y_seq, site_seq = [], [], []

    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
        site_seq.append(site_id)

    return (
        np.array(X_seq),
        np.array(site_seq).reshape(-1, 1),
        np.array(y_seq),
    )


def preprocess_embedded_data(data_glob, scaler_path):
    """
    Load, scale, and sequence multi-site data.
    """
    dfs = load_all_sites(data_glob)

    # Fit scaler on ALL data (same as baseline philosophy)
    all_features = pd.concat(dfs)[FEATURE_COLS_BASELINE].ffill().bfill()
    scaler = MinMaxScaler()
    scaler.fit(all_features)

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)

    X_all, site_all, y_all = [], [], []

    for df in dfs:
        df_scaled = df.copy()
        df_scaled[FEATURE_COLS_BASELINE] = scaler.transform(
            df[FEATURE_COLS_BASELINE].ffill().bfill()
        )

        X_seq, site_seq, y_seq = create_site_sequences(df_scaled, TIME_STEPS)

        X_all.append(X_seq)
        site_all.append(site_seq)
        y_all.append(y_seq)

    return (
        np.vstack(X_all),
        np.vstack(site_all),
        np.concatenate(y_all),
    )
