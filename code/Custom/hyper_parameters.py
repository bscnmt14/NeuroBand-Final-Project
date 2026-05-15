"""
===============================================================================
Hyperparameters Configuration File
===============================================================================
This module serves as the central configuration hub for the pipeline.
It stores the optimal parameters identified during the Grid Search phase,
ensuring reproducibility and isolating configuration from execution logic.
"""
import pandas as pd

# Optimal temporal parameters for signal windowing and feature extraction.
# These values were determined per kernel to maximize classification accuracy.
MODEL_DATA_CONFIG = {
    'linear': {
        'margin': 0,
        'window': 120,
        'step': 60,
        'zc': 1e-08,
        'ssc': 0.0001
    }
    # 'poly': {
    #     'margin': 800,
    #     'window': 400,
    #     'step': 200,
    #     'zc': 1e-08,
    #     'ssc': 1e-08
    # },
    # 'rbf': {
    #     'margin': 600,
    #     'window': 800,
    #     'step': 400,
    #     'zc': 1e-08,
    #     'ssc': 0.0001
    # },
    # 'sigmoid': {
    #     'margin': 1000,
    #     'window': 800,
    #     'step': 400,
    #     'zc': 1e-08,
    #     'ssc': 1e-06
    # }
}

# Optimal algorithmic parameters for the Support Vector Machine (SVM) models.
# 'class_weight': 'balanced' is applied globally to assist with class imbalances.
MODEL_PARAMS = {
    'linear':  {'C': 0.01,  'class_weight': 'balanced'},
#    'rbf':     {'C': 1.0,   'gamma': 'auto',  'class_weight': 'balanced'},
#    'poly':    {'C': 0.1,   'gamma': 0.1,     'degree': 3, 'class_weight': 'balanced'},
#    'sigmoid': {'C': 100.0, 'gamma': 'scale', 'class_weight': 'balanced'}
}

# In hyper_parameters.py
TARGET_CLASSES = [
    'at_rest',
    'fist',
    'like',
    'open_hand',
    'pinch',
    'pointing',
    'side_flex',
    'wrist_extension',
    'wrist_flexion'
]