import sys
import os

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import joblib
import optuna
import matplotlib.pyplot as plt

# ============================================================
# Paths
# ============================================================
STUDY_PATH = "models/optuna/baseline_lstm_v2/study.pkl"
OUTPUT_DIR = "models/optuna/baseline_lstm_v2/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_optuna_plot(obj, filename):
    """
    Optuna matplotlib plots may return Figure or Axes.
    This helper safely saves either.
    """
    if hasattr(obj, "figure"):
        fig = obj.figure
    else:
        fig = obj

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 60)
    print("   OPTUNA ANALYSIS & VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load study
    # --------------------------------------------------------
    study = joblib.load(STUDY_PATH)

    # --------------------------------------------------------
    # Plot 1 — Optimization history
    # --------------------------------------------------------
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    save_optuna_plot(
        fig1,
        os.path.join(OUTPUT_DIR, "optimization_history.png")
    )

    # --------------------------------------------------------
    # Plot 2 — Hyperparameter importance
    # --------------------------------------------------------
    fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
    save_optuna_plot(
        fig2,
        os.path.join(OUTPUT_DIR, "param_importance.png")
    )

    # --------------------------------------------------------
    # Plot 3 — Parallel coordinates
    # --------------------------------------------------------
    fig3 = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    save_optuna_plot(
        fig3,
        os.path.join(OUTPUT_DIR, "parallel_coordinates.png")
    )

    print("✅ Optuna plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
