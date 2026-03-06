import json
import os
from datetime import datetime


def save_metrics(
    *,
    model_name: str,
    model_version: str,
    metrics: dict,
    output_path: str,
    extra_info: dict | None = None
):
    """
    Standardized metrics saver for all models.
    """
    payload = {
        "model_name": model_name,
        "model_version": model_version,
        "metrics": metrics,
        "timestamp": datetime.utcnow().isoformat()
    }

    if extra_info:
        payload.update(extra_info)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        """Recursively convert numpy types to Python native types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    payload = convert_numpy_types(payload)
    
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[OK] Metrics saved to: {output_path}")
