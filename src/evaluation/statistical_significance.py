# src/evaluation/statistical_significance.py
"""
Global vs Local Model Statistical Significance Testing

Compares RMSE of the global model (Embedded LSTM leave-one-out) against
local models (per-site Naive, XGBoost, Baseline LSTM) across all 21 sites.

Uses:
  - Paired Wilcoxon signed-rank test (non-parametric, no normality assumption)
  - Paired t-test (parametric, for comparison)

Both tests are paired because we compare the SAME sites across models.

Usage:
    python src/evaluation/statistical_significance.py
"""

import json
import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ──────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────
PER_SITE_SUMMARY = "models/metrics/per_site/per_site_summary.json"
LEAVE_ONE_OUT_SUMMARY = "models/metrics/leave_one_out/leave_one_out_summary.json"
OUTPUT_DIR = "src/evaluation"
RESULTS_PATH = os.path.join(OUTPUT_DIR, "significance_results.json")


# ──────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────
def load_data():
    """Load per-site and leave-one-out RMSE values aligned by site."""

    with open(PER_SITE_SUMMARY, "r") as f:
        per_site = json.load(f)

    with open(LEAVE_ONE_OUT_SUMMARY, "r") as f:
        loo = json.load(f)

    # Build aligned arrays (pv_01 .. pv_21)
    sites = sorted(per_site.keys())  # pv_01, pv_02, ..., pv_21

    naive_rmse = []
    xgboost_rmse = []
    lstm_rmse = []
    global_rmse = []

    loo_by_site = {r["held_out"]: r["rmse"] for r in loo["results"]}

    for site in sites:
        naive_rmse.append(per_site[site]["naive_24h"]["rmse"])
        xgboost_rmse.append(per_site[site]["xgboost"]["rmse"])
        lstm_rmse.append(per_site[site]["baseline_lstm"]["rmse"])
        global_rmse.append(loo_by_site[site])

    return {
        "sites": sites,
        "naive": np.array(naive_rmse),
        "xgboost": np.array(xgboost_rmse),
        "baseline_lstm": np.array(lstm_rmse),
        "global_lstm": np.array(global_rmse),
    }


# ──────────────────────────────────────────────────────────
# Statistical Tests
# ──────────────────────────────────────────────────────────
def apply_holm_bonferroni(results, alpha=0.05):
    """
    Apply Holm-Bonferroni correction for multiple comparisons.
    Adds 'holm_significant' fields to each result dict.
    """
    n = len(results)

    # Collect p-values from both tests
    for test_key in ["wilcoxon", "paired_ttest"]:
        pvals = []
        for r in results:
            p = r[test_key]["p_value"]
            pvals.append(p if p is not None else 1.0)

        # Sort indices by p-value (ascending)
        sorted_idx = sorted(range(n), key=lambda i: pvals[i])

        # Step-down: compare p[i] against alpha / (n - rank)
        for rank, idx in enumerate(sorted_idx):
            adjusted_alpha = alpha / (n - rank)
            is_sig = pvals[idx] < adjusted_alpha
            results[idx][test_key]["holm_adjusted_alpha"] = round(adjusted_alpha, 6)
            results[idx][test_key]["holm_significant"] = bool(is_sig)

    return results

def run_paired_test(a, b, name_a, name_b):
    """
    Run paired Wilcoxon signed-rank + paired t-test.
    a and b are arrays of RMSE per site.
    Lower RMSE = better.
    """
    diff = a - b  # positive means a is worse (higher RMSE)
    n = len(diff)

    # Descriptive stats
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Count wins
    wins_a = int(np.sum(diff < 0))   # a has lower RMSE
    wins_b = int(np.sum(diff > 0))   # b has lower RMSE
    ties = int(np.sum(diff == 0))

    # Paired t-test (two-sided)
    t_stat, t_pvalue = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank test (two-sided, non-parametric)
    # Only valid when there are non-zero differences
    non_zero = np.sum(diff != 0)
    if non_zero >= 10:
        w_stat, w_pvalue = stats.wilcoxon(a, b)
    else:
        w_stat, w_pvalue = float("nan"), float("nan")

    # Effect size: Cohen's d for paired samples
    if std_diff > 0:
        cohens_d = mean_diff / std_diff
    else:
        cohens_d = 0.0

    # Determine winner
    if mean_a < mean_b:
        better = name_a
    else:
        better = name_b

    result = {
        "comparison": f"{name_a} vs {name_b}",
        "n_sites": n,
        "mean_rmse_a": round(mean_a, 6),
        "mean_rmse_b": round(mean_b, 6),
        "mean_diff": round(mean_diff, 6),
        "std_diff": round(std_diff, 6),
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "paired_ttest": {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(t_pvalue), 6),
            "significant_005": bool(t_pvalue < 0.05),
            "significant_001": bool(t_pvalue < 0.01),
        },
        "wilcoxon": {
            "w_statistic": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
            "p_value": round(float(w_pvalue), 6) if not np.isnan(w_pvalue) else None,
            "significant_005": bool(w_pvalue < 0.05) if not np.isnan(w_pvalue) else None,
            "significant_001": bool(w_pvalue < 0.01) if not np.isnan(w_pvalue) else None,
        },
        "cohens_d": round(cohens_d, 4),
        "better_model": better,
    }

    return result


# ──────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────
def generate_plots(data, results):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sites = [s.replace("pv_", "") for s in data["sites"]]
    x = np.arange(len(sites))
    width = 0.2

    # ── Plot 1: RMSE comparison bar chart ──
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x - 1.5 * width, data["naive"], width, label="Naive 24h (local)", alpha=0.8, color="#e74c3c")
    ax.bar(x - 0.5 * width, data["xgboost"], width, label="XGBoost (local)", alpha=0.8, color="#3498db")
    ax.bar(x + 0.5 * width, data["baseline_lstm"], width, label="Baseline LSTM (local)", alpha=0.8, color="#2ecc71")
    ax.bar(x + 1.5 * width, data["global_lstm"], width, label="Embedded LSTM (global, LOO)", alpha=0.8, color="#9b59b6")

    ax.set_xlabel("Site", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("Global vs Local Model RMSE by Site", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sites, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "global_vs_local_rmse.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"[OK] Plot saved: {path1}")

    # ── Plot 2: Paired difference plots (Global vs each local) ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    comparisons = [
        ("naive", "Naive 24h", "#e74c3c"),
        ("xgboost", "XGBoost", "#3498db"),
        ("baseline_lstm", "Baseline LSTM", "#2ecc71"),
    ]

    for ax, (key, label, color) in zip(axes, comparisons):
        diff = data["global_lstm"] - data[key]
        ax.bar(x, diff, color=color, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Site", fontsize=10)
        ax.set_ylabel("RMSE Difference\n(Global - Local)", fontsize=10)
        ax.set_title(f"Global LSTM vs {label}", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sites, fontsize=8, rotation=45)
        ax.grid(axis="y", alpha=0.3)

        # Annotate with p-value
        for r in results:
            if key in r["comparison"].lower() or label.lower() in r["comparison"].lower():
                p = r["wilcoxon"]["p_value"] or r["paired_ttest"]["p_value"]
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                ax.text(0.98, 0.95, f"p = {p:.4f} {sig}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
                break

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "global_vs_local_differences.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"[OK] Plot saved: {path2}")


# ──────────────────────────────────────────────────────────
# Pretty Print
# ──────────────────────────────────────────────────────────
def print_results(results):
    """Print formatted significance test results."""

    print(f"\n{'=' * 70}")
    print("   GLOBAL vs LOCAL — STATISTICAL SIGNIFICANCE RESULTS")
    print(f"{'=' * 70}")

    for r in results:
        sig_t = "YES" if r["paired_ttest"]["significant_005"] else "NO"
        sig_w = "YES" if r["wilcoxon"]["significant_005"] else "NO"

        print(f"\n  {r['comparison']}")
        print(f"  {'-' * 50}")
        print(f"  Mean RMSE (A): {r['mean_rmse_a']:.6f}    Mean RMSE (B): {r['mean_rmse_b']:.6f}")
        print(f"  Mean diff:     {r['mean_diff']:+.6f}    Std diff:     {r['std_diff']:.6f}")
        print(f"  Wins A:  {r['wins_a']}    Wins B:  {r['wins_b']}    Ties: {r['ties']}")
        print(f"  Cohen's d:     {r['cohens_d']:+.4f}")
        print(f"  Paired t-test: t={r['paired_ttest']['t_statistic']:.4f}  p={r['paired_ttest']['p_value']:.6f}  sig(0.05): {sig_t}")
        print(f"  Wilcoxon:      W={r['wilcoxon']['w_statistic']}  p={r['wilcoxon']['p_value']:.6f}  sig(0.05): {sig_w}")

        # Holm-Bonferroni corrected results
        if "holm_significant" in r["wilcoxon"]:
            holm_w = "YES" if r["wilcoxon"]["holm_significant"] else "NO"
            holm_t = "YES" if r["paired_ttest"]["holm_significant"] else "NO"
            print(f"  Holm-Bonferroni: Wilcoxon={holm_w} (adj alpha={r['wilcoxon']['holm_adjusted_alpha']:.4f})  t-test={holm_t}")

        print(f"  Better model:  {r['better_model']}")

    print(f"\n{'=' * 70}")
    print("  Interpretation:")
    print("    p < 0.05  ->  statistically significant difference (*)")
    print("    p < 0.01  ->  highly significant (**)")
    print("    p < 0.001 ->  very highly significant (***)")
    print("    Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
    print("    Holm-Bonferroni: corrects for 3 simultaneous comparisons")
    print(f"{'=' * 70}\n")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = load_data()

    # Run 3 comparisons: Global LSTM vs each local model
    results = [
        run_paired_test(data["global_lstm"], data["naive"],
                        "Global LSTM (LOO)", "Naive 24h (local)"),
        run_paired_test(data["global_lstm"], data["xgboost"],
                        "Global LSTM (LOO)", "XGBoost (local)"),
        run_paired_test(data["global_lstm"], data["baseline_lstm"],
                        "Global LSTM (LOO)", "Baseline LSTM (local)"),
    ]

    # Apply Holm-Bonferroni correction for multiple comparisons
    results = apply_holm_bonferroni(results)

    print_results(results)

    # Save results to JSON
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved: {RESULTS_PATH}")

    # Generate plots
    generate_plots(data, results)

    print("\nDone!")
