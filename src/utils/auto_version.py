"""
Automatic version detection and management utilities.

This module provides functions to automatically detect the next available version
for models, manage versioned paths, and prevent overwrites of existing results.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional


def get_next_version(model_type: str, base_dir: str = "models") -> str:
    """
    Auto-detect the next version number by scanning the models directory.
    
    Scans for existing files matching the pattern: {model_type}_v{number}.*
    and returns the next available version.
    
    Args:
        model_type: Model type name (e.g., "baseline_lstm", "embedded_lstm", "xgboost")
        base_dir: Base directory to scan (default: "models")
    
    Returns:
        Next version string, e.g., "v1", "v2", "v3"
        
    Example:
        >>> get_next_version("baseline_lstm")
        "v2"  # if v1 already exists
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return "v1"
    
    # Pattern to match: baseline_lstm_v1.keras, baseline_lstm_v2.pkl, etc.
    pattern = re.compile(rf"{re.escape(model_type)}_v(\d+)\.")
    
    max_version = 0
    
    # Scan all files in models directory
    for file in base_path.iterdir():
        if file.is_file():
            match = pattern.search(file.name)
            if match:
                version_num = int(match.group(1))
                max_version = max(max_version, version_num)
    
    # Also scan metrics subdirectory
    metrics_path = base_path / "metrics"
    if metrics_path.exists():
        for file in metrics_path.iterdir():
            if file.is_file():
                match = pattern.search(file.name)
                if match:
                    version_num = int(match.group(1))
                    max_version = max(max_version, version_num)
    
    next_version = max_version + 1
    return f"v{next_version}"


def get_versioned_paths(
    model_type: str,
    version: Optional[str] = None,
    base_dir: str = "models",
    eval_dir: str = "src/evaluation"
) -> Dict[str, str]:
    """
    Get all versioned paths for a model.
    
    Args:
        model_type: Model type name (e.g., "baseline_lstm", "embedded_lstm")
        version: Specific version string or None to auto-detect next
        base_dir: Base directory for models (default: "models")
        eval_dir: Base directory for evaluation (default: "src/evaluation")
    
    Returns:
        Dictionary with paths:
        {
            "model_path": "models/baseline_lstm_v2.keras",
            "scaler_path": "models/baseline_scaler_v2.pkl",
            "metrics_path": "models/metrics/baseline_metrics_v2.json",
            "plots_dir": "src/evaluation/baseline/v2",
            "version": "v2"
        }
    
    Example:
        >>> paths = get_versioned_paths("baseline_lstm")
        >>> print(paths["version"])
        "v2"
        >>> print(paths["plots_dir"])
        "src/evaluation/baseline/v2"
    """
    # Auto-detect version if not provided
    if version is None:
        version = get_next_version(model_type, base_dir)
    
    # Extract model family name for evaluation directory
    # baseline_lstm -> baseline, embedded_lstm -> embedded
    if "_" in model_type:
        model_family = model_type.split("_")[0]
    else:
        model_family = model_type
    
    # Determine file extension based on model type
    if "xgboost" in model_type.lower():
        model_ext = "json"
    else:
        model_ext = "keras"
    
    paths = {
        "model_path": f"{base_dir}/{model_type}_{version}.{model_ext}",
        "scaler_path": f"{base_dir}/{model_family}_scaler_{version}.pkl",
        "metrics_path": f"{base_dir}/metrics/{model_type}_{version}_metrics.json",  # Fixed: Use full model_type, not just family
        "plots_dir": f"{eval_dir}/{model_family}/{version}",
        "version": version
    }
    
    return paths


def get_existing_versions(model_type: str, base_dir: str = "models") -> list:
    """
    Get a list of all existing versions for a model type.
    
    Args:
        model_type: Model type name
        base_dir: Base directory to scan
    
    Returns:
        List of version strings, sorted (e.g., ["v1", "v2", "v3"])
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return []
    
    pattern = re.compile(rf"{re.escape(model_type)}_v(\d+)\.")
    versions = set()
    
    for file in base_path.iterdir():
        if file.is_file():
            match = pattern.search(file.name)
            if match:
                versions.add(int(match.group(1)))
    
    # Sort and format
    return [f"v{v}" for v in sorted(versions)]


def ensure_dirs(paths: Dict[str, str]) -> None:
    """
    Ensure all directories in the paths dictionary exist.
    
    Args:
        paths: Dictionary of paths from get_versioned_paths()
    """
    # Create metrics directory
    metrics_dir = os.path.dirname(paths["metrics_path"])
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Create plots directory
    os.makedirs(paths["plots_dir"], exist_ok=True)
    
    # Create model directory if needed
    model_dir = os.path.dirname(paths["model_path"])
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)


if __name__ == "__main__":
    # Test the versioning system
    print("=" * 60)
    print("Testing Auto-Versioning System")
    print("=" * 60)
    
    # Test baseline_lstm
    print("\n[baseline_lstm]")
    existing = get_existing_versions("baseline_lstm")
    print(f"Existing versions: {existing}")
    
    next_ver = get_next_version("baseline_lstm")
    print(f"Next version: {next_ver}")
    
    paths = get_versioned_paths("baseline_lstm")
    print(f"Paths for {paths['version']}:")
    for key, value in paths.items():
        if key != "version":
            print(f"  {key}: {value}")
    
    # Test embedded_lstm
    print("\n[embedded_lstm]")
    existing = get_existing_versions("embedded_lstm")
    print(f"Existing versions: {existing}")
    
    next_ver = get_next_version("embedded_lstm")
    print(f"Next version: {next_ver}")
    
    paths = get_versioned_paths("embedded_lstm")
    print(f"Paths for {paths['version']}:")
    for key, value in paths.items():
        if key != "version":
            print(f"  {key}: {value}")
