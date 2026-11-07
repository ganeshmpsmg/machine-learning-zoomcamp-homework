import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Load data
df = pd.read_csv('data/hour.csv')

# Remove target leakage and identifiers
LEAK_COLS = ['casual', 'registered']
ID_COLS = ['instant']
df = df.drop(columns=LEAK_COLS + ID_COLS)

# Chronological split: 70% train, 15% val, 15% test
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

print("Features available:", list(X_train.columns))

# Feature engineering class
class BikeFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Cyclical transforms
        X['hr_sin'] = np.sin(2 * np.pi * X['hr'] / 24.0)
        X['hr_cos'] = np.cos(2 * np.pi * X['hr'] / 24.0)
        X['mnth_sin'] = np.sin(2 * np.pi * (X['mnth'] - 1) / 12.0)
        X['mnth_cos'] = np.cos(2 * np.pi * (X['mnth'] - 1) / 12.0)
        
        # Domain features
        # Peak commute hour on working days (7-9 AM, 16-19 PM)
        is_rush = (X['workingday'] == 1) & (X['hr'].isin([7, 8, 9, 16, 17, 18, 19]))
        X['is_rush_hour'] = is_rush.astype(int)
        
        # Late night / early morning low-demand hours
        is_night = X['hr'].isin([0, 1, 2, 3, 4, 5])
        X['is_night'] = is_night.astype(int)
        
        # Weather comfort interaction
        X['temp_hum_interaction'] = X['temp'] * (1.0 - X['hum'])
        X['temp_diff'] = X['atemp'] - X['temp']
        
        # Adverse weather indicator
        X['bad_weather'] = (X['weathersit'] >= 3).astype(int)
        
        return X

fe = BikeFeatureEngineer()
X_train_fe = fe.transform(X_train)
X_val_fe = fe.transform(X_val)
X_test_fe = fe.transform(X_test)

print("Engineered feature columns:", list(X_train_fe.columns))

def evaluate(y_true, y_pred, name="Model"):
    y_pred = np.clip(y_pred, 0, None)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:30s} | RMSE: {rmse:7.2f} | MAE: {mae:7.2f} | R2: {r2:6.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}

print("\n--- BASELINE MODELS ---")
# 1. Baseline: mean of train target
mean_pred_val = np.full_like(y_val, np.mean(y_train), dtype=float)
evaluate(y_val, mean_pred_val, "Baseline (Train Mean)")

median_pred_val = np.full_like(y_val, np.median(y_train), dtype=float)
evaluate(y_val, median_pred_val, "Baseline (Train Median)")

# Direct target vs log1p target comparison
print("\n--- DIRECT TARGET (y) vs LOG-TRANSFORMED TARGET (log1p(y)) ---")

models = {
    "Linear Regression (Ridge)": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, max_depth=8, learning_rate=0.1, random_state=42)
}

print("\n[DIRECT TARGET]:")
for name, model in models.items():
    model.fit(X_train_fe, y_train)
    preds = model.predict(X_val_fe)
    evaluate(y_val, preds, name)

print("\n[LOG1P TARGET]:")
y_train_log = np.log1p(y_train)
for name, model in models.items():
    model.fit(X_train_fe, y_train_log)
    preds = np.expm1(model.predict(X_val_fe))
    evaluate(y_val, preds, f"{name} (log1p)")
