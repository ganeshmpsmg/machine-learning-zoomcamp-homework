"""
Unit and integration tests for FastAPI app endpoints and Pydantic validation schemas.
"""

import pytest
from pydantic import ValidationError
from app import (
    app,
    root,
    health_check,
    predict_endpoint,
    predict_batch_endpoint,
    BikeRentalRequest,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse
)


@pytest.fixture
def valid_payload():
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


def test_root_endpoint():
    data = root()
    assert isinstance(data, dict)
    assert "documentation" in data
    assert "health_check" in data
    assert data["documentation"] == "/docs"


def test_health_check_endpoint():
    response = health_check()
    assert isinstance(response, HealthResponse)
    assert response.status == "healthy"
    assert response.model_loaded is True
    assert response.model_type == "HistGradientBoostingRegressor"


def test_predict_endpoint_valid_payload(valid_payload):
    req = BikeRentalRequest(**valid_payload)
    response = predict_endpoint(req)
    assert isinstance(response, PredictionResponse)
    assert response.status == "success"
    assert response.predicted_demand >= 0.0
    assert response.predicted_demand_rounded >= 0
    assert response.predicted_demand_rounded == int(round(response.predicted_demand))


def test_predict_endpoint_invalid_hour(valid_payload):
    # Hour 25 is outside [0, 23]
    bad_payload = valid_payload.copy()
    bad_payload["hr"] = 25
    with pytest.raises(ValidationError) as excinfo:
        BikeRentalRequest(**bad_payload)
    errors = excinfo.value.errors()
    assert any(err["loc"][0] == "hr" for err in errors)


def test_predict_endpoint_invalid_normalized_range(valid_payload):
    # Humidity 1.5 is outside [0.0, 1.0]
    bad_payload = valid_payload.copy()
    bad_payload["hum"] = 1.50
    with pytest.raises(ValidationError) as excinfo:
        BikeRentalRequest(**bad_payload)
    errors = excinfo.value.errors()
    assert any(err["loc"][0] == "hum" for err in errors)


def test_predict_endpoint_invalid_season(valid_payload):
    # Season 5 is outside [1, 4]
    bad_payload = valid_payload.copy()
    bad_payload["season"] = 5
    with pytest.raises(ValidationError) as excinfo:
        BikeRentalRequest(**bad_payload)
    errors = excinfo.value.errors()
    assert any(err["loc"][0] == "season" for err in errors)


def test_predict_batch_endpoint(valid_payload):
    item1 = BikeRentalRequest(**valid_payload)
    item2_payload = valid_payload.copy()
    item2_payload["hr"] = 4  # 4 AM
    item2_payload["temp"] = 0.25
    item2 = BikeRentalRequest(**item2_payload)

    response = predict_batch_endpoint([item1, item2])
    assert isinstance(response, BatchPredictionResponse)
    assert response.status == "success"
    assert response.count == 2
    assert len(response.predictions) == 2
    # Commute peak hour demand should exceed 4 AM early morning demand
    assert response.predictions[0].predicted_demand > response.predictions[1].predicted_demand
