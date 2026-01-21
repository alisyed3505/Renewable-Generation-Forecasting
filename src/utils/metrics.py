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

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Metrics saved to: {output_path}")
