"""
Prediction script and inference module for Bike Rental Demand Prediction.
Machine Learning Zoomcamp 2026 Midterm Project.

Can be imported as a module or executed from CLI:
    python predict.py
    python predict.py --json '{"season": 3, "yr": 1, "mnth": 7, "hr": 17, "holiday": 0, "weekday": 3, "workingday": 1, "weathersit": 1, "temp": 0.82, "atemp": 0.7879, "hum": 0.44, "windspeed": 0.2239}'
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import sys
import json
import argparse
import joblib
import pandas as pd
import numpy as np

# Ensure pipeline_utils is accessible when unpickling
import pipeline_utils
from pipeline_utils import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "bike_rental_model.joblib")

# Sample record for quick CLI demonstration (Summer evening peak hour, 2012)
SAMPLE_INPUT = {
    "season": 3,
    "yr": 1,
    "mnth": 7,
    "hr": 17,
    "holiday": 0,
    "weekday": 3,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.82,
    "atemp": 0.7879,
    "hum": 0.44,
    "windspeed": 0.2239
}

_cached_model = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Loads and caches the serialized Scikit-learn pipeline."""
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run train.py first.")
        _cached_model = joblib.load(model_path)
    return _cached_model


def predict(input_data, model_path: str = DEFAULT_MODEL_PATH):
    """
    Predict bike rental count for a dictionary or list/DataFrame of features.

    Parameters:
        input_data: dict, list of dicts, or pandas DataFrame.
        model_path: path to joblib artifact.

    Returns:
        float or list of floats: predicted demand (cnt), non-negative.
    """
    model = load_model(model_path)

    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
        is_single = True
    elif isinstance(input_data, list):
        df = pd.DataFrame(input_data)
        is_single = False
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
        is_single = len(df) == 1
    else:
        raise TypeError("input_data must be a dict, list of dicts, or pandas DataFrame.")

    # Ensure required features exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    raw_preds = model.predict(X)
    clipped_preds = np.clip(raw_preds, 0, None)
    rounded_preds = [round(float(p), 2) for p in clipped_preds]

    return rounded_preds[0] if is_single else rounded_preds


def main():
    parser = argparse.ArgumentParser(description="Predict bike rental demand.")
    parser.add_argument("--json", dest="json_str", type=str, default=None, help="JSON string representing feature dictionary")
    parser.add_argument("--file", dest="file_path", type=str, default=None, help="Path to JSON file with input features")
    parser.add_argument("--model", dest="model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to model file")
    args = parser.parse_args()

    if args.json_str:
        payload = json.loads(args.json_str)
    elif args.file_path:
        with open(args.file_path, 'r') as f:
            payload = json.load(f)
    else:
        print("[Info] No input provided. Using sample peak-hour input:")
        payload = SAMPLE_INPUT

    prediction = predict(payload, model_path=args.model_path)
    print("\nInput features:")
    print(json.dumps(payload, indent=2))
    print(f"\n-> Predicted Bike Rental Demand (cnt): {prediction}")


if __name__ == "__main__":
    main()
