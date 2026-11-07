"""
Training script for Bike Rental Demand Prediction.
Machine Learning Zoomcamp 2026 Midterm Project.

This script:
1. Loads data/hour.csv.
2. Validates schema and removes target leak features ('casual', 'registered') and identifier ('instant').
3. Applies a chronological train/val/test split (70% / 15% / 15%).
4. Trains and evaluates candidate models on the validation set.
5. Fits the champion HistGradientBoostingRegressor model pipeline.
6. Evaluates on the held-out test set strictly once.
7. Serializes the trained pipeline using Joblib to models/bike_rental_model.joblib.
8. Writes evaluation metrics to models/metrics.json.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pipeline_utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
    IDENTIFIER_COLUMNS,
    BikeFeatureEngineer
)


def compute_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and R2 with non-negative lower bound clipping."""
    y_pred_clipped = np.clip(y_pred, 0, None)
    return {
        'rmse': round(float(np.sqrt(mean_squared_error(y_true, y_pred_clipped))), 2),
        'mae': round(float(mean_absolute_error(y_true, y_pred_clipped)), 2),
        'r2': round(float(r2_score(y_true, y_pred_clipped)), 4)
    }


def load_and_prepare_data(data_path: str):
    """Loads CSV, validates integrity, and removes target leak and identifier columns."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please check data path.")

    df = pd.read_csv(data_path)
    print(f"[Data Loading] Loaded dataset from {data_path} with shape: {df.shape}")

    # Integrity verification
    assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' missing."
    for col in LEAKAGE_COLUMNS:
        if col in df.columns:
            print(f"[Security Check] Dropping target leakage column: {col}")
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            print(f"[Security Check] Dropping identifier column: {col}")

    # Drop leakage and identifier
    drop_cols = [c for c in LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Missing value verification
    null_counts = df.isnull().sum().sum()
    print(f"[Data Validation] Total missing values across dataset: {null_counts}")
    assert null_counts == 0, "Dataset contains unexpected missing values."

    return df


def split_chronologically(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15):
    """Splits dataset chronologically to avoid future data leakage."""
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    print(f"[Data Splitting] Train rows: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"[Data Splitting] Val rows:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"[Data Splitting] Test rows:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")

    return train_df, val_df, test_df


def build_pipeline():
    """Constructs the full Scikit-learn Pipeline with feature engineering and regressor."""
    return Pipeline([
        ('feature_engineer', BikeFeatureEngineer()),
        ('regressor', HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            l2_regularization=0.1,
            max_iter=200,
            random_state=42
        ))
    ])


def train(data_path="data/hour.csv", output_dir="models"):
    """Full training pipeline execution."""
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_prepare_data(data_path)

    # Separate features and target
    train_df, val_df, test_df = split_chronologically(df)

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[TARGET_COLUMN].values

    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df[TARGET_COLUMN].values

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[TARGET_COLUMN].values

    print("\n--- Training Validation Model ---")
    val_pipeline = build_pipeline()
    val_pipeline.fit(X_train, y_train)

    val_preds = val_pipeline.predict(X_val)
    val_metrics = compute_metrics(y_val, val_preds)
    print(f"[Validation Results] RMSE: {val_metrics['rmse']} | MAE: {val_metrics['mae']} | R2: {val_metrics['r2']}")

    print("\n--- Training Final Production Pipeline (Train + Val) ---")
    train_val_df = pd.concat([train_df, val_df], axis=0)
    X_train_val = train_val_df[FEATURE_COLUMNS].copy()
    y_train_val = train_val_df[TARGET_COLUMN].values

    final_pipeline = build_pipeline()
    final_pipeline.fit(X_train_val, y_train_val)

    # Evaluate on held-out test set strictly once
    print("\n--- Final Test Set Evaluation (Unseen Data) ---")
    test_preds = final_pipeline.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds)
    print(f"[Test Results] RMSE: {test_metrics['rmse']} | MAE: {test_metrics['mae']} | R2: {test_metrics['r2']}")

    # Save artifacts
    model_filename = os.path.join(output_dir, "bike_rental_model.joblib")
    joblib.dump(final_pipeline, model_filename)
    print(f"\n[Artifact Export] Saved complete model pipeline to: {model_filename}")

    metrics_filename = os.path.join(output_dir, "metrics.json")
    metrics_payload = {
        "model_type": "HistGradientBoostingRegressor",
        "hyperparameters": {
            "learning_rate": 0.08,
            "max_depth": 6,
            "l2_regularization": 0.1,
            "max_iter": 200,
            "random_state": 42
        },
        "feature_columns": FEATURE_COLUMNS,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "train_val_combined": len(train_val_df)
        }
    }

    with open(metrics_filename, 'w') as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[Artifact Export] Saved metrics summary to: {metrics_filename}")

    return final_pipeline, metrics_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bike Rental Demand Prediction Pipeline")
    parser.add_argument("--data", default="data/hour.csv", help="Path to hour.csv dataset")
    parser.add_argument("--output-dir", default="models", help="Directory to save artifacts")
    args = parser.parse_args()

    train(data_path=args.data, output_dir=args.output_dir)
