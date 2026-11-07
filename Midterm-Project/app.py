"""
FastAPI application for Bike Rental Demand Prediction.
Machine Learning Zoomcamp 2026 Midterm Project.

Provides:
- GET /health: Health check endpoint
- POST /predict: Single observation prediction with Pydantic schema validation
- POST /predict/batch: Batch prediction endpoint
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict

import pipeline_utils
from predict import predict, load_model, DEFAULT_MODEL_PATH

API_VERSION = "1.0.0"
MODEL_TYPE = "HistGradientBoostingRegressor"


app = FastAPI(
    title="Bike Rental Demand Prediction API",
    description="Production-ready Machine Learning API predicting hourly bike rental demand using the UCI Bike Sharing Dataset.",
    version=API_VERSION
)


class BikeRentalRequest(BaseModel):
    season: int = Field(..., ge=1, le=4, description="Season (1: springer, 2: summer, 3: fall, 4: winter)")
    yr: int = Field(..., ge=0, le=1, description="Year (0: 2011, 1: 2012)")
    mnth: int = Field(..., ge=1, le=12, description="Month of year (1 to 12)")
    hr: int = Field(..., ge=0, le=23, description="Hour of the day (0 to 23)")
    holiday: int = Field(..., ge=0, le=1, description="Holiday flag (1: holiday, 0: not holiday)")
    weekday: int = Field(..., ge=0, le=6, description="Day of week (0 to 6, 0=Sunday)")
    workingday: int = Field(..., ge=0, le=1, description="Working day flag (1: neither weekend nor holiday, 0: otherwise)")
    weathersit: int = Field(..., ge=1, le=4, description="Weather category (1: Clear, 2: Mist, 3: Light Rain/Snow, 4: Heavy Rain/Snow)")
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature in Celsius (t / 41)")
    atemp: float = Field(..., ge=0.0, le=1.0, description="Normalized feeling temperature in Celsius (t / 50)")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized relative humidity (hum / 100)")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized wind speed (windspeed / 67)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponse(BaseModel):
    predicted_demand: float = Field(..., description="Continuous predicted bike rental count (cnt)")
    predicted_demand_rounded: int = Field(..., description="Integer rounded rental count")
    model_version: str = Field(default=API_VERSION)
    status: str = Field(default="success")


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    status: str = Field(default="success")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    api_version: str
    model_path: str


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to Bike Rental Demand Prediction API",
        "documentation": "/docs",
        "health_check": "/health",
        "predict_endpoint": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    loaded = False
    try:
        model = load_model()
        loaded = model is not None
    except Exception:
        loaded = False

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_type=MODEL_TYPE,
        api_version=API_VERSION,
        model_path=DEFAULT_MODEL_PATH
    )


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict_endpoint(payload: BikeRentalRequest):
    try:
        pred_value = predict(payload.model_dump())
        return PredictionResponse(
            predicted_demand=pred_value,
            predicted_demand_rounded=int(round(pred_value)),
            model_version=API_VERSION,
            status="success"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
def predict_batch_endpoint(payload: List[BikeRentalRequest]):
    try:
        data = [item.model_dump() for item in payload]
        preds = predict(data)
        responses = [
            PredictionResponse(
                predicted_demand=p,
                predicted_demand_rounded=int(round(p)),
                model_version=API_VERSION,
                status="success"
            )
            for p in preds
        ]
        return BatchPredictionResponse(
            predictions=responses,
            count=len(responses),
            status="success"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(exc)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
