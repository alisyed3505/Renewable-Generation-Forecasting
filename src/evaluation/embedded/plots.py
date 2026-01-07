"""
Plots for Embedded LSTM evaluation.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Embedded LSTM Training History")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_loss.png"))
    plt.close()


def plot_predictions(y_true, y_pred, save_dir, n_samples=300):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(y_true[:n_samples], label="Actual")
    plt.plot(y_pred[:n_samples], label="Predicted")
    plt.xlabel("Time step (3h)")
    plt.ylabel("Normalized Power")
    plt.title("Embedded LSTM: Prediction vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "prediction_vs_actual.png"))
    plt.close()


def plot_error_distribution(y_true, y_pred, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    errors = y_pred - y_true

    plt.figure()
    plt.hist(errors, bins=60)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.title("Embedded LSTM Error Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "error_distribution.png"))
    plt.close()


def plot_site_mae(site_mae, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    sites = list(site_mae.keys())
    maes = list(site_mae.values())

    plt.figure(figsize=(10, 4))
    plt.bar(sites, maes)
    plt.xlabel("Site ID")
    plt.ylabel("MAE")
    plt.title("Embedded LSTM: Per-Site MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_site_mae.png"))
    plt.close()
