# [LEGACY] This file was the original attempt at deep learning in this repo.
# It appears to predict solar height rather than power output and has some issues.
# Please use 'train_model.py' and 'inference.py' for the actual power forecasting system.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ... original code commented out or kept for reference ...
print("This script is legacy. Please run 'train_model.py' to train the new Random Forest model.")
print("Run 'inference.py' to generate predictions.")

"""
Original code below:
dataset = pd.read_csv('pv_01.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
...
"""