# Optional utility: export trials from an existing Optuna study
# Not required if optimize_lstm.py already exports trials.csv

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import joblib
import pandas as pd

STUDY_PATH = "models/optuna/baseline_lstm_v2/study.pkl"
OUTPUT_PATH = "models/optuna/baseline_lstm_v2/trials.csv"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

study = joblib.load(STUDY_PATH)

rows = []
for trial in study.trials:
    row = {
        "trial": trial.number,
        "value": trial.value,
        **trial.params
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Optuna trials exported to {OUTPUT_PATH}")
