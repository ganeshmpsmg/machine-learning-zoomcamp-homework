"""
Pipeline utilities and custom transformers for Bike Rental Demand Prediction.
Machine Learning Zoomcamp 2026 Midterm Project.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_COLUMNS = [
    'season', 'yr', 'mnth', 'hr', 'holiday',
    'weekday', 'workingday', 'weathersit',
    'temp', 'atemp', 'hum', 'windspeed'
]

TARGET_COLUMN = 'cnt'
LEAKAGE_COLUMNS = ['casual', 'registered']
IDENTIFIER_COLUMNS = ['instant']


class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer temporal, cyclical, and environmental
    interaction features for bike rental demand forecasting.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transforms input DataFrame or dict-compatible table into feature matrix.
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, list):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        # Cyclical encoding of hour (0-23)
        X['hr_sin'] = np.sin(2 * np.pi * X['hr'] / 24.0)
        X['hr_cos'] = np.cos(2 * np.pi * X['hr'] / 24.0)

        # Cyclical encoding of month (1-12)
        X['mnth_sin'] = np.sin(2 * np.pi * (X['mnth'] - 1) / 12.0)
        X['mnth_cos'] = np.cos(2 * np.pi * (X['mnth'] - 1) / 12.0)

        # Rush hour indicator: commute hours on working days
        # Working day peak hours: 7, 8, 9 AM and 16, 17, 18, 19 PM
        is_rush = (X['workingday'] == 1) & (X['hr'].isin([7, 8, 9, 16, 17, 18, 19]))
        X['is_rush_hour'] = is_rush.astype(int)

        # Low demand late-night hours: 0, 1, 2, 3, 4, 5
        is_night = X['hr'].isin([0, 1, 2, 3, 4, 5])
        X['is_night'] = is_night.astype(int)

        # Daytime weekend leisure hours: 10 AM - 16 PM on non-working days
        is_weekend_day = (X['workingday'] == 0) & (X['hr'].isin([10, 11, 12, 13, 14, 15, 16]))
        X['is_weekend_leisure'] = is_weekend_day.astype(int)

        # Temperature & humidity comfort index interaction
        X['temp_hum_interaction'] = X['temp'] * (1.0 - X['hum'])

        # Difference between 'feeling' temp and measured temp
        X['temp_diff'] = X['atemp'] - X['temp']

        # Adverse weather indicator (Light/Heavy snow or rain)
        X['bad_weather'] = (X['weathersit'] >= 3).astype(int)

        return X
