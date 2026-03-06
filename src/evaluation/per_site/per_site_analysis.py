# src/evaluation/per_site/per_site_analysis.py
"""
Per-Site Comparison Analysis

Loads per-site metrics from models/metrics/per_site/ and generates:
1. Grouped bar chart: RMSE per site per model
2. Heatmap: model x site -> RMSE
3. Site difficulty ranking
4. Summary CSV

Usage:
    python src/evaluation/per_site/per_site_analysis.py
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
METRICS_DIR = "models/metrics/per_site"
OUTPUT_DIR = "src/evaluation/per_site"
MODELS = ["naive_24h", "xgboost", "baseline_lstm"]
MODEL_LABELS = {"naive_24h": "Naive 24h", "xgboost": "XGBoost", "baseline_lstm": "Baseline LSTM"}
SITES = [f"pv_{i:02d}" for i in range(1, 22)]


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
def load_all_metrics():
    """Load all per-site metric files into a nested dict: {model: {site: metrics}}."""
    data = {}
    for model in MODELS:
        data[model] = {}
        for site in SITES:
            path = os.path.join(METRICS_DIR, f"{model}_{site}_metrics.json")
            if os.path.exists(path):
                with open(path, "r") as f:
                    raw = json.load(f)
                data[model][site] = raw.get("metrics", {})
    return data


# ──────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────
def plot_grouped_bar(data, metric="rmse"):
    """Grouped bar chart: RMSE per site per model."""
    fig, ax = plt.subplots(figsize=(16, 6))

    available_sites = sorted(
        set(s for m in MODELS for s in data.get(m, {}).keys())
    )
    n_sites = len(available_sites)
    n_models = len(MODELS)
    bar_width = 0.25
    x = np.arange(n_sites)

    for i, model in enumerate(MODELS):
        values = [data.get(model, {}).get(site, {}).get(metric, 0) for site in available_sites]
        bars = ax.bar(x + i * bar_width, values, bar_width, label=MODEL_LABELS.get(model, model))

    ax.set_xlabel("Site", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Per-Site {metric.upper()} Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(available_sites, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"per_site_{metric}_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")


def plot_heatmap(data, metric="rmse"):
    """Heatmap: model x site -> metric value."""
    available_sites = sorted(
        set(s for m in MODELS for s in data.get(m, {}).keys())
    )

    matrix = []
    for model in MODELS:
        row = [data.get(model, {}).get(site, {}).get(metric, np.nan) for site in available_sites]
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(16, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(len(available_sites)))
    ax.set_xticklabels(available_sites, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in MODELS], fontsize=10)

    # Annotate cells
    for i in range(len(MODELS)):
        for j in range(len(available_sites)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanmedian(matrix) else "black"
                ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    ax.set_title(f"Per-Site {metric.upper()} Heatmap", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label=metric.upper())
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, f"per_site_{metric}_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")


def plot_site_difficulty(data):
    """Rank sites by average RMSE across models (difficulty ranking)."""
    available_sites = sorted(
        set(s for m in MODELS for s in data.get(m, {}).keys())
    )

    avg_rmse = []
    for site in available_sites:
        values = [data.get(m, {}).get(site, {}).get("rmse", np.nan) for m in MODELS]
        avg_rmse.append(np.nanmean(values))

    # Sort by difficulty (highest avg RMSE = hardest)
    sorted_pairs = sorted(zip(available_sites, avg_rmse), key=lambda x: x[1], reverse=True)
    sorted_sites, sorted_vals = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_sites)))
    ax.barh(range(len(sorted_sites)), sorted_vals, color=colors)
    ax.set_yticks(range(len(sorted_sites)))
    ax.set_yticklabels(sorted_sites, fontsize=10)
    ax.set_xlabel("Average RMSE (across models)", fontsize=12)
    ax.set_title("Site Difficulty Ranking (hardest at top)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # Hardest at top
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "site_difficulty_ranking.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")


def generate_summary_csv(data):
    """Export a CSV table: model, site, rmse, mae."""
    import csv

    path = os.path.join(OUTPUT_DIR, "per_site_comparison.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "site", "rmse", "mae"])
        for model in MODELS:
            for site in sorted(data.get(model, {}).keys()):
                m = data[model][site]
                writer.writerow([
                    MODEL_LABELS.get(model, model),
                    site,
                    f"{m.get('rmse', ''):.6f}" if m.get("rmse") else "",
                    f"{m.get('mae', ''):.6f}" if m.get("mae") else "",
                ])
    print(f"  [OK] Saved: {path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def run_analysis():
    print("=" * 60)
    print("   PER-SITE COMPARISON ANALYSIS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_all_metrics()

    # Count available results
    total = sum(len(sites) for sites in data.values())
    print(f"\n  Found {total} metric files across {len(MODELS)} models")

    if total == 0:
        print("  [WARN] No metrics found. Run train_per_site.py first.")
        return

    print("\n  Generating plots...")
    plot_grouped_bar(data, "rmse")
    plot_grouped_bar(data, "mae")
    plot_heatmap(data, "rmse")
    plot_heatmap(data, "mae")
    plot_site_difficulty(data)
    generate_summary_csv(data)

    print(f"\n{'=' * 60}")
    print(f"  Analysis complete. All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_analysis()
