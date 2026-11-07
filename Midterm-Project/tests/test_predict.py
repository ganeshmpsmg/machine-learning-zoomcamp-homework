"""
Unit tests for bike rental demand prediction pipeline and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from pipeline_utils import BikeFeatureEngineer, FEATURE_COLUMNS
from predict import predict, load_model


@pytest.fixture
def sample_feature_dict():
    return {
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


def test_feature_engineer_transformation(sample_feature_dict):
    fe = BikeFeatureEngineer()
    df = pd.DataFrame([sample_feature_dict])
    transformed = fe.transform(df)

    # Verify cyclical features exist and are bounded between -1 and 1
    assert "hr_sin" in transformed.columns
    assert "hr_cos" in transformed.columns
    assert "mnth_sin" in transformed.columns
    assert "mnth_cos" in transformed.columns
    assert -1.0 <= transformed["hr_sin"].iloc[0] <= 1.0
    assert -1.0 <= transformed["hr_cos"].iloc[0] <= 1.0

    # Verify domain interaction features
    assert "is_rush_hour" in transformed.columns
    assert transformed["is_rush_hour"].iloc[0] == 1  # 17:00 on workingday is commute peak
    assert "temp_hum_interaction" in transformed.columns
    assert "temp_diff" in transformed.columns


def test_model_loading():
    model = load_model()
    assert model is not None
    assert hasattr(model, "predict")


def test_single_prediction(sample_feature_dict):
    pred = predict(sample_feature_dict)
    assert isinstance(pred, float)
    assert pred >= 0.0
    # Peak summer hour should be high demand (> 200)
    assert pred > 200.0


def test_batch_prediction(sample_feature_dict):
    record2 = sample_feature_dict.copy()
    record2["hr"] = 3  # 3 AM late night low demand
    record2["temp"] = 0.4
    record2["atemp"] = 0.4

    batch = [sample_feature_dict, record2]
    preds = predict(batch)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert all(p >= 0.0 for p in preds)
    # Peak commute hour demand must be higher than 3 AM demand
    assert preds[0] > preds[1]


def test_missing_feature_raises_error(sample_feature_dict):
    bad_dict = sample_feature_dict.copy()
    del bad_dict["temp"]  # Missing required feature
    with pytest.raises(ValueError, match="Input is missing required features"):
        predict(bad_dict)
