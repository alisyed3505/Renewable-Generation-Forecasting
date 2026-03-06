# src/utils/experiment_registry.py
"""
Unified Experiment Registry — scans all metric JSON files and builds a
single, sortable table of every training run ever recorded.

Usage:
    python src/utils/experiment_registry.py          # print + export CSV
    python src/utils/experiment_registry.py --csv     # CSV only (quiet)
"""

import json
import os
import glob
import csv
import sys
from pathlib import Path
from datetime import datetime


# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
METRICS_ROOT = "models/metrics"
REGISTRY_CSV = os.path.join(METRICS_ROOT, "experiment_registry.csv")

# Columns for the registry table
COLUMNS = ["model", "version", "site", "rmse", "mae", "timestamp"]


# ──────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────
def discover_metric_files(root: str = METRICS_ROOT) -> list:
    """
    Recursively find all *_metrics.json files under the metrics root.
    """
    pattern = os.path.join(root, "**", "*_metrics.json")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def parse_metric_file(path: str) -> dict | None:
    """
    Read a single metrics JSON and extract registry columns.
    Returns a flat dict with the standard COLUMNS, or None on failure.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Skipping {path}: {exc}")
        return None

    metrics = data.get("metrics", {})

    return {
        "model": data.get("model_name", "unknown"),
        "version": data.get("model_version", "?"),
        "site": data.get("site", data.get("scope", "-")),
        "rmse": _fmt(metrics.get("rmse")),
        "mae": _fmt(metrics.get("mae")),
        "timestamp": _short_ts(data.get("timestamp", "")),
    }


def build_registry(root: str = METRICS_ROOT) -> list[dict]:
    """
    Scan all metric files and return a list of registry rows.
    """
    rows = []
    for path in discover_metric_files(root):
        row = parse_metric_file(path)
        if row:
            rows.append(row)

    # Sort by model name, then version
    rows.sort(key=lambda r: (r["model"], r["version"], r["site"]))
    return rows


def export_csv(rows: list[dict], output_path: str = REGISTRY_CSV):
    """
    Write registry rows to a CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def print_table(rows: list[dict]):
    """
    Pretty-print the experiment registry as a formatted table.
    """
    if not rows:
        print("No experiments found.")
        return

    # Calculate column widths
    widths = {col: len(col) for col in COLUMNS}
    for row in rows:
        for col in COLUMNS:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    # Header
    header = " | ".join(col.upper().ljust(widths[col]) for col in COLUMNS)
    separator = "-+-".join("-" * widths[col] for col in COLUMNS)

    print()
    print("=" * len(header))
    print("   EXPERIMENT REGISTRY")
    print("=" * len(header))
    print(header)
    print(separator)

    for row in rows:
        line = " | ".join(
            str(row.get(col, "")).ljust(widths[col]) for col in COLUMNS
        )
        print(line)

    print(separator)
    print(f"Total experiments: {len(rows)}")
    print()


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def _fmt(value) -> str:
    """Format a numeric value to 6 decimal places, or '-' if missing."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.6f}"
    except (ValueError, TypeError):
        return str(value)


def _short_ts(ts: str) -> str:
    """Shorten ISO timestamp to YYYY-MM-DD HH:MM."""
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return ts[:16]


# ──────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    quiet = "--csv" in sys.argv

    rows = build_registry()

    if not quiet:
        print_table(rows)

    csv_path = export_csv(rows)
    print(f"[OK] Registry exported to: {csv_path}")
    print(f"     {len(rows)} experiment(s) recorded")
