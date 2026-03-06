# src/training/train_leave_one_out.py
"""
Leave-One-Site-Out Experiment — Embedded LSTM v3

For each of the 21 sites:
  1. Train an embedded LSTM on the OTHER 20 sites
  2. Prepare the held-out site's data separately
  3. Evaluate on the held-out site
  4. Save metrics per held-out site

This tests cross-site generalization: can the model predict
for a solar farm it has NEVER seen during training?

Usage:
    python src/training/train_leave_one_out.py                # all 21 folds
    python src/training/train_leave_one_out.py --sites 1 5    # specific folds only
"""

import sys
import os
import argparse
import json
import glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ──────────────────────────────────────────────────────────
# Constants (match project conventions)
# ──────────────────────────────────────────────────────────
DATA_GLOB = "data/raw/pv_*.csv"
METRICS_DIR = "models/metrics/leave_one_out"

FEATURE_COLS = [
    "hour_of_day_sin", "hour_of_day_cos",
    "month_of_year_sin", "month_of_year_cos",
    "sunposition_thetaZ", "sunposition_solarAzimuth",
    "clearsky_diffuse", "clearsky_direct", "clearsky_global",
    "TemperatureAt0", "RelativeHumidityAt0",
    "SolarRadiationGlobalAt0", "SolarRadiationDirectAt0",
    "SolarRadiationDiffuseAt0", "TotalCloudCoverAt0",
]

TIME_STEPS = 8
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2
RANDOM_SEED = 42

# v1 architecture defaults (fair generalization comparison)
EMBEDDING_DIM = 4
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 50
NUM_TRAIN_SITES = 20   # Always 20 for leave-one-out
NUM_EMBED_SLOTS = 21   # 20 training + 1 unseen (true cold-start embedding)


# ──────────────────────────────────────────────────────────
# Data helpers (standalone, no circular config imports)
# ──────────────────────────────────────────────────────────
def _get_site_files():
    """Return sorted list of site CSV paths."""
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matched: {DATA_GLOB}")
    return files


def _load_site_df(path):
    """Load a single site CSV."""
    df = pd.read_csv(path, delimiter=";")
    if df.columns[-1].startswith("Unnamed"):
        df = df.iloc[:, :-1]
    if "time_idx" in df.columns:
        df = df.sort_values("time_idx")
    return df


def _create_sequences(X, y, time_steps):
    """Create LSTM sequences for one site."""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i : i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


def prepare_train_data(site_files, exclude_idx, scaler=None):
    """
    Prepare training data from all sites EXCEPT exclude_idx.
    Returns (X_feat_all, X_site_all, y_all, scaler).
    """
    X_all, site_all, y_all = [], [], []
    all_dfs = []

    new_id = 0
    for idx, path in enumerate(site_files):
        if idx == exclude_idx:
            continue
        df = _load_site_df(path)
        df["site_id"] = new_id
        all_dfs.append(df)
        new_id += 1

    # Fit scaler on all training data
    if scaler is None:
        scaler = MinMaxScaler()
        all_features = pd.concat(all_dfs)[FEATURE_COLS].ffill().bfill()
        scaler.fit(all_features)

    for df in all_dfs:
        df_feat = scaler.transform(df[FEATURE_COLS].ffill().bfill())
        y = df["power_normed"].fillna(0).values
        site_id = df["site_id"].iloc[0]

        X_seq, y_seq = _create_sequences(df_feat, y, TIME_STEPS)
        site_seq = np.full((len(y_seq), 1), site_id)

        X_all.append(X_seq)
        site_all.append(site_seq)
        y_all.append(y_seq)

    return (
        np.vstack(X_all),
        np.vstack(site_all),
        np.concatenate(y_all),
        scaler,
    )


def prepare_held_out_data(site_path, scaler):
    """
    Prepare test data for the held-out site.
    Uses the SAME scaler fitted on the training sites.
    Site ID is set to NUM_TRAIN_SITES (slot 20), which is an embedding
    slot that was NEVER updated during training — true cold start.
    """
    df = _load_site_df(site_path)
    df_feat = scaler.transform(df[FEATURE_COLS].ffill().bfill())
    y = df["power_normed"].fillna(0).values

    X_seq, y_seq = _create_sequences(df_feat, y, TIME_STEPS)

    # Use embedding slot NUM_TRAIN_SITES (=20) for the unseen site.
    # Training sites occupy slots 0..19, so slot 20 is never trained.
    # This gives a true cold-start evaluation with a random embedding.
    site_seq = np.full((len(y_seq), 1), NUM_TRAIN_SITES, dtype=int)

    # Only evaluate on the test portion (last 20%)
    split_idx = int(len(X_seq) * (1 - TEST_SPLIT))
    return (
        X_seq[split_idx:],
        site_seq[split_idx:],
        y_seq[split_idx:],
    )


# ──────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────
def train_one_fold(exclude_idx, site_files):
    """Train on 20 sites, evaluate on held-out site."""
    import tensorflow as tf
    from src.models.embedded.lstm_v3 import build_embedded_lstm
    from src.utils.metrics import save_metrics

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    site_name = f"pv_{exclude_idx + 1:02d}"

    # Prepare training data (20 sites)
    X_feat, X_site, y, scaler = prepare_train_data(site_files, exclude_idx)

    # Train/val split
    split_idx = int(len(y) * (1 - TEST_SPLIT))
    X_feat_train, X_feat_val = X_feat[:split_idx], X_feat[split_idx:]
    X_site_train, X_site_val = X_site[:split_idx], X_site[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Build model (21 embedding slots: 0..19 for training, 20 for cold-start)
    model = build_embedded_lstm(
        num_sites=NUM_EMBED_SLOTS,
        embedding_dim=EMBEDDING_DIM,
        time_steps=TIME_STEPS,
        num_features=X_feat_train.shape[2],
        lstm_units_1=LSTM_UNITS_1,
        lstm_units_2=LSTM_UNITS_2,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )

    # Train
    from tensorflow.keras.callbacks import EarlyStopping

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=0,
        ),
    ]

    model.fit(
        [X_site_train, X_feat_train], y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=([X_site_val, X_feat_val], y_val),
        callbacks=callbacks, verbose=0,
    )

    # Evaluate on held-out site
    X_feat_test, X_site_test, y_test = prepare_held_out_data(
        site_files[exclude_idx], scaler
    )

    y_pred = np.clip(model.predict([X_site_test, X_feat_test], verbose=0).flatten(), 0, 1)
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - y_pred)))

    # Also evaluate in-sample (on validation set from training sites)
    y_pred_val = np.clip(model.predict([X_site_val, X_feat_val], verbose=0).flatten(), 0, 1)
    rmse_insample = float(np.sqrt(np.mean((y_val - y_pred_val) ** 2)))
    mae_insample = float(np.mean(np.abs(y_val - y_pred_val)))

    # Save metrics
    metrics_path = os.path.join(METRICS_DIR, f"embedded_lstm_leave_out_{site_name}_metrics.json")
    save_metrics(
        model_name="embedded_lstm",
        model_version="v3_leave_one_out",
        metrics={"rmse": rmse, "mae": mae},
        output_path=metrics_path,
        extra_info={
            "model_type": "lstm",
            "scope": "leave_one_out",
            "site": site_name,
            "held_out_site": site_name,
            "held_out_site_idx": exclude_idx,
            "num_train_sites": NUM_TRAIN_SITES,
            "embedding_dim": EMBEDDING_DIM,
            "insample_rmse": rmse_insample,
            "insample_mae": mae_insample,
        },
    )

    # Clean up
    del model
    tf.keras.backend.clear_session()

    return {
        "held_out": site_name,
        "rmse": rmse,
        "mae": mae,
        "insample_rmse": rmse_insample,
        "insample_mae": mae_insample,
    }


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def run_leave_one_out(site_indices=None):
    """Run leave-one-site-out experiment across all (or selected) sites."""
    site_files = _get_site_files()
    total_sites = len(site_files)

    if site_indices is None:
        site_indices = list(range(total_sites))

    os.makedirs(METRICS_DIR, exist_ok=True)

    results = []

    for fold_num, exclude_idx in enumerate(site_indices, 1):
        site_name = f"pv_{exclude_idx + 1:02d}"
        print(f"\n{'=' * 60}")
        print(f"   FOLD {fold_num}/{len(site_indices)}: held-out = {site_name}")
        print(f"   Training on {total_sites - 1} sites, testing on {site_name}")
        print(f"{'=' * 60}")

        r = train_one_fold(exclude_idx, site_files)
        results.append(r)

        print(f"  Held-out RMSE: {r['rmse']:.4f}  MAE: {r['mae']:.4f}")
        print(f"  In-sample RMSE: {r['insample_rmse']:.4f}  MAE: {r['insample_mae']:.4f}")

    # Summary
    print(f"\n{'=' * 60}")
    print("   LEAVE-ONE-SITE-OUT COMPLETE")
    print(f"{'=' * 60}")

    avg_rmse = np.mean([r["rmse"] for r in results])
    avg_mae = np.mean([r["mae"] for r in results])
    avg_insample_rmse = np.mean([r["insample_rmse"] for r in results])

    print(f"   Folds completed: {len(results)}")
    print(f"   Avg held-out RMSE: {avg_rmse:.4f}")
    print(f"   Avg held-out MAE:  {avg_mae:.4f}")
    print(f"   Avg in-sample RMSE: {avg_insample_rmse:.4f}")
    print(f"   Generalization gap: {avg_rmse - avg_insample_rmse:.4f}")

    # Save combined summary
    summary_path = os.path.join(METRICS_DIR, "leave_one_out_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "results": results,
            "avg_held_out_rmse": avg_rmse,
            "avg_held_out_mae": avg_mae,
            "avg_insample_rmse": avg_insample_rmse,
            "generalization_gap": avg_rmse - avg_insample_rmse,
        }, f, indent=2)
    print(f"   Summary saved to: {summary_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-one-site-out experiment")
    parser.add_argument(
        "--sites", nargs="+", type=int,
        help="Site numbers to hold out (1-based, e.g. --sites 1 5 10)"
    )
    args = parser.parse_args()

    if args.sites:
        indices = [s - 1 for s in args.sites]  # Convert to 0-based
    else:
        indices = None

    run_leave_one_out(site_indices=indices)
