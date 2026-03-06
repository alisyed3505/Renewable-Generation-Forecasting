# src/evaluation/leave_one_out/analysis.py
"""
Leave-One-Site-Out Analysis & Visualization

Generates:
1. Bar chart of held-out RMSE per site
2. Comparison: held-out vs in-sample performance
3. Generalization gap visualization
4. Summary statistics

Usage:
    python src/evaluation/leave_one_out/analysis.py
"""

import os
import sys
import json
import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
METRICS_DIR = "models/metrics/leave_one_out"
OUTPUT_DIR = "src/evaluation/leave_one_out"


# ──────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────
def load_results():
    """Load all leave-one-out metric files."""
    pattern = os.path.join(METRICS_DIR, "embedded_lstm_leave_out_*_metrics.json")
    files = sorted(glob.glob(pattern))

    results = []
    for path in files:
        with open(path, "r") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        results.append({
            "site": data.get("held_out_site", data.get("site", "?")),
            "rmse": metrics.get("rmse", 0),
            "mae": metrics.get("mae", 0),
            "insample_rmse": data.get("insample_rmse", 0),
            "insample_mae": data.get("insample_mae", 0),
        })
    return results


# ──────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────
def plot_held_out_rmse(results):
    """Bar chart of held-out RMSE per site."""
    sites = [r["site"] for r in results]
    rmse_vals = [r["rmse"] for r in results]

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sites)))

    # Sort by RMSE
    sorted_pairs = sorted(zip(sites, rmse_vals, colors), key=lambda x: x[1])
    sites_s, rmse_s, colors_s = zip(*sorted_pairs)

    ax.bar(range(len(sites_s)), rmse_s, color=colors_s)
    ax.set_xticks(range(len(sites_s)))
    ax.set_xticklabels(sites_s, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (held-out site)", fontsize=12)
    ax.set_title("Leave-One-Site-Out: RMSE per Held-Out Site", fontsize=14, fontweight="bold")
    ax.axhline(np.mean(rmse_vals), color="red", linestyle="--", alpha=0.7, label=f"Mean: {np.mean(rmse_vals):.4f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "leave_one_out_rmse.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")


def plot_generalization_gap(results):
    """Side-by-side comparison: held-out vs in-sample RMSE per site."""
    sites = [r["site"] for r in results]
    held_out = [r["rmse"] for r in results]
    insample = [r["insample_rmse"] for r in results]

    x = np.arange(len(sites))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, insample, width, label="In-sample (20 sites)", color="#4CAF50", alpha=0.8)
    ax.bar(x + width / 2, held_out, width, label="Held-out (unseen site)", color="#FF5722", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Generalization Gap: In-Sample vs Held-Out", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "generalization_gap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")


def plot_gap_distribution(results):
    """Histogram of generalization gap (held_out - insample)."""
    gaps = [r["rmse"] - r["insample_rmse"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gaps, bins=15, color="#2196F3", alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(gaps), color="red", linestyle="--", label=f"Mean gap: {np.mean(gaps):.4f}")
    ax.set_xlabel("RMSE Gap (held-out - in-sample)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Generalization Gap", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "gap_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [OK] Saved: {path}")


def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("   LEAVE-ONE-SITE-OUT SUMMARY")
    print("=" * 70)
    print(f"{'SITE':<10} {'HELD-OUT RMSE':>14} {'IN-SAMPLE RMSE':>16} {'GAP':>10}")
    print("-" * 50)

    for r in results:
        gap = r["rmse"] - r["insample_rmse"]
        print(f"{r['site']:<10} {r['rmse']:>14.4f} {r['insample_rmse']:>16.4f} {gap:>+10.4f}")

    print("-" * 50)
    avg_rmse = np.mean([r["rmse"] for r in results])
    avg_insample = np.mean([r["insample_rmse"] for r in results])
    avg_gap = avg_rmse - avg_insample
    print(f"{'AVERAGE':<10} {avg_rmse:>14.4f} {avg_insample:>16.4f} {avg_gap:>+10.4f}")
    print()


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def run_analysis():
    print("=" * 60)
    print("   LEAVE-ONE-SITE-OUT ANALYSIS")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = load_results()
    if not results:
        print("  [WARN] No results found. Run train_leave_one_out.py first.")
        return

    print(f"  Found {len(results)} fold results")

    print_summary(results)

    print("  Generating plots...")
    plot_held_out_rmse(results)
    plot_generalization_gap(results)
    plot_gap_distribution(results)

    print(f"\n  Analysis complete. Outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    run_analysis()
