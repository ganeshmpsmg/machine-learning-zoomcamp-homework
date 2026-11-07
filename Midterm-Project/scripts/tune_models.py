import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin

df = pd.read_csv('data/hour.csv')
df = df.drop(columns=['casual', 'registered', 'instant'])

n = len(df)
n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_df = df.iloc[:n_train].copy()
val_df = df.iloc[n_train:n_train + n_val].copy()
test_df = df.iloc[n_train + n_val:].copy()

y_train = train_df['cnt'].values
y_val = val_df['cnt'].values
y_test = test_df['cnt'].values

X_train = train_df.drop(columns=['cnt', 'dteday'])
X_val = val_df.drop(columns=['cnt', 'dteday'])
X_test = test_df.drop(columns=['cnt', 'dteday'])

class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['hr_sin'] = np.sin(2 * np.pi * X['hr'] / 24.0)
        X['hr_cos'] = np.cos(2 * np.pi * X['hr'] / 24.0)
        X['mnth_sin'] = np.sin(2 * np.pi * (X['mnth'] - 1) / 12.0)
        X['mnth_cos'] = np.cos(2 * np.pi * (X['mnth'] - 1) / 12.0)
        
        is_rush = (X['workingday'] == 1) & (X['hr'].isin([7, 8, 9, 16, 17, 18, 19]))
        X['is_rush_hour'] = is_rush.astype(int)
        
        is_night = X['hr'].isin([0, 1, 2, 3, 4, 5])
        X['is_night'] = is_night.astype(int)
        
        X['temp_hum_interaction'] = X['temp'] * (1.0 - X['hum'])
        X['temp_diff'] = X['atemp'] - X['temp']
        X['bad_weather'] = (X['weathersit'] >= 3).astype(int)
        return X

fe = BikeFeatureEngineer()
X_train_fe = fe.transform(X_train)
X_val_fe = fe.transform(X_val)
X_test_fe = fe.transform(X_test)

def score(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }

print("=== Tuning HistGradientBoostingRegressor ===")
best_hgb = None
best_hgb_rmse = 999999
for lr in [0.05, 0.08, 0.1, 0.15]:
    for max_depth in [6, 8, 10, None]:
        for l2 in [0.0, 0.1, 1.0]:
            model = HistGradientBoostingRegressor(
                learning_rate=lr,
                max_depth=max_depth,
                l2_regularization=l2,
                max_iter=200,
                random_state=42
            )
            model.fit(X_train_fe, y_train)
            val_scores = score(y_val, model.predict(X_val_fe))
            if val_scores['rmse'] < best_hgb_rmse:
                best_hgb_rmse = val_scores['rmse']
                best_hgb = (lr, max_depth, l2, val_scores)

print(f"Best HGB params: lr={best_hgb[0]}, max_depth={best_hgb[1]}, l2={best_hgb[2]}")
print(f"Best HGB Val Scores: RMSE={best_hgb[3]['rmse']:.2f}, MAE={best_hgb[3]['mae']:.2f}, R2={best_hgb[3]['r2']:.4f}")

print("\n=== Tuning RandomForestRegressor ===")
best_rf = None
best_rf_rmse = 999999
for n_est in [50, 100, 150]:
    for md in [12, 16, 20, None]:
        for msl in [1, 2, 4]:
            model = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=md,
                min_samples_leaf=msl,
                random_state=42,
                n_jobs=2
            )
            model.fit(X_train_fe, y_train)
            val_scores = score(y_val, model.predict(X_val_fe))
            if val_scores['rmse'] < best_rf_rmse:
                best_rf_rmse = val_scores['rmse']
                best_rf = (n_est, md, msl, val_scores)

print(f"Best RF params: n_est={best_rf[0]}, max_depth={best_rf[1]}, min_samples_leaf={best_rf[2]}")
print(f"Best RF Val Scores: RMSE={best_rf[3]['rmse']:.2f}, MAE={best_rf[3]['mae']:.2f}, R2={best_rf[3]['r2']:.4f}")
