import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Load dataset
df = pd.read_csv('data/hour.csv')
df = df.drop(columns=['casual', 'registered', 'instant'])

# Chronological split
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
        'rmse': round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        'mae': round(float(mean_absolute_error(y_true, y_pred)), 2),
        'r2': round(float(r2_score(y_true, y_pred)), 4)
    }

# 1. Baseline: Train mean
baseline_mean_val = np.full_like(y_val, np.mean(y_train), dtype=float)
val_baseline = score(y_val, baseline_mean_val)

# 2. Linear Regression (OLS)
lr = LinearRegression()
lr.fit(X_train_fe, y_train)
val_lr = score(y_val, lr.predict(X_val_fe))

# 3. Ridge Regression
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_fe, y_train)
val_ridge = score(y_val, ridge.predict(X_val_fe))

# 4. Decision Tree (tuned depth)
dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42)
dt.fit(X_train_fe, y_train)
val_dt = score(y_val, dt.predict(X_val_fe))

# 5. Random Forest (tuned)
rf = RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=4, random_state=42, n_jobs=2)
rf.fit(X_train_fe, y_train)
val_rf = score(y_val, rf.predict(X_val_fe))

# 6. Gradient Boosting (Standard GBR)
gbr = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
gbr.fit(X_train_fe, y_train)
val_gbr = score(y_val, gbr.predict(X_val_fe))

# 7. Hist Gradient Boosting (tuned)
hgb = HistGradientBoostingRegressor(learning_rate=0.08, max_depth=6, l2_regularization=0.1, max_iter=200, random_state=42)
hgb.fit(X_train_fe, y_train)
val_hgb = score(y_val, hgb.predict(X_val_fe))

print("=== VALIDATION SET COMPARISON ===")
val_results = {
    "Baseline (Train Mean)": val_baseline,
    "Linear Regression (OLS)": val_lr,
    "Ridge Regression": val_ridge,
    "Decision Tree": val_dt,
    "Random Forest": val_rf,
    "Gradient Boosting (GBR)": val_gbr,
    "HistGradientBoosting": val_hgb
}

for name, res in val_results.items():
    print(f"{name:25s} | RMSE: {res['rmse']:7.2f} | MAE: {res['mae']:7.2f} | R2: {res['r2']:6.4f}")

# Best model selection:
best_model_name = min(val_results, key=lambda k: val_results[k]['rmse'])
print(f"\nBest model on validation set: {best_model_name}")

# Now evaluate on Test Set ONLY ONCE:
# Option A: Trained on Train
test_pred_train_only = hgb.predict(X_test_fe)
test_score_train_only = score(y_test, test_pred_train_only)
print(f"\n=== TEST SET EVALUATION (Trained on Train) ===")
print(f"{best_model_name} Test: RMSE: {test_score_train_only['rmse']} | MAE: {test_score_train_only['mae']} | R2: {test_score_train_only['r2']}")

# Option B: Trained on Train + Val (Standard practice in ML Zoomcamp for final deployment model)
train_val_df = pd.concat([train_df, val_df], axis=0)
X_train_val = train_val_df.drop(columns=['cnt', 'dteday'])
y_train_val = train_val_df['cnt'].values
X_train_val_fe = fe.transform(X_train_val)

final_model = HistGradientBoostingRegressor(learning_rate=0.08, max_depth=6, l2_regularization=0.1, max_iter=200, random_state=42)
final_model.fit(X_train_val_fe, y_train_val)

test_pred_final = final_model.predict(X_test_fe)
test_score_final = score(y_test, test_pred_final)
print(f"\n=== TEST SET EVALUATION (Final Model Trained on Train+Val) ===")
print(f"{best_model_name} Test: RMSE: {test_score_final['rmse']} | MAE: {test_score_final['mae']} | R2: {test_score_final['r2']}")

# Save benchmark metrics to json
summary = {
    "validation_results": val_results,
    "best_model": best_model_name,
    "test_results_train_only": test_score_train_only,
    "test_results_final_model": test_score_final
}
with open('benchmark_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Saved benchmark_results.json")
