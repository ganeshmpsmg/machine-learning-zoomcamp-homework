# Bike Rental Demand Prediction Using Machine Learning

**DataTalks.Club Machine Learning Zoomcamp 2026 — Midterm Project**  
*Covers Modules 1 through 6: Problem Description, Exploratory Data Analysis, Feature Engineering, Linear & Tree Models, Hyperparameter Tuning, Model Selection, and Containerized Web Service Deployment.*

---

## 1. Problem Statement & Business Context

Bike-sharing systems automate bike rental and return across hundreds of stations in modern metropolitan areas. For municipal mobility managers and commercial operators like **Capital Bikeshare** in Washington, D.C., operating efficiency depends on **fleet balancing**:
- **Preventing Empty Stations**: Morning commuters arriving at suburban transit hubs need bikes to reach downtown offices. A station with zero bicycles causes transit disruption and lost revenue.
- **Preventing Full Stations**: Commuters arriving at commercial hubs require open docking points. Inability to dock bicycles causes customer frustration and service penalties.
- **Minimizing Operational Rebalancing Costs**: Hauling bicycles via diesel rebalancing vans between stations is costly and carbon-intensive. Accurate demand forecasting allows proactive, route-optimized rebalancing.

### Objective
Build a supervised machine learning regression pipeline that predicts total hourly bike rental demand (`cnt`) using historical environmental, seasonal, and calendar features.

---

## 2. Dataset Source & License

- **Dataset Source**: [UCI Machine Learning Repository — Bike Sharing Dataset (ID: 275)](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
- **Primary Data File**: `data/hour.csv` (17,379 hourly records spanning 731 days from January 1, 2011 to December 31, 2012).
- **Attribution & License**:
  The dataset was contributed by Hadi Fanaee-T and João Gama (Laboratory of Artificial Intelligence and Decision Support, University of Porto / INESC Porto).
  > **Citation**: Fanaee-T, Hadi, and Gama, Joao, *"Event labeling combining ensemble detectors and background knowledge"*, Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, [doi:10.1007/s13748-013-0040-3](https://doi.org/10.1007/s13748-013-0040-3).

---

## 3. Feature Descriptions & Target Leakage Prevention

### 3.1 Strict Target Leakage Rule
In the raw dataset, the target column `cnt` satisfies the exact identity:
$$\text{cnt} = \text{casual} + \text{registered}$$
- `casual`: Count of unregistered, walk-up users.
- `registered`: Count of annual or monthly subscription members.

Because `casual` and `registered` are recorded **after** rentals take place, they are unavailable at prediction time. **Using `casual` or `registered` as input features constitutes complete target leakage and is strictly prohibited.** This pipeline validates and drops both columns immediately upon ingestion.

### 3.2 Identifier Exclusion
- `instant`: Arbitrary sequential integer row index ($1, 2, \dots, 17379$). Dropped to prevent index memorization.

### 3.3 Feature Schema
| Feature | Type | Range / Values | Description |
| :--- | :--- | :--- | :--- |
| `season` | Categorical / Ordinal | `1, 2, 3, 4` | Season (1: Spring, 2: Summer, 3: Fall, 4: Winter) |
| `yr` | Binary | `0, 1` | Year (0: 2011, 1: 2012) |
| `mnth` | Discrete | `1 – 12` | Calendar month of the year |
| `hr` | Discrete | `0 – 23` | Hour of the day |
| `holiday` | Binary | `0, 1` | 1 if official public holiday; 0 otherwise |
| `weekday` | Discrete | `0 – 6` | Day of the week (0: Sunday, ..., 6: Saturday) |
| `workingday`| Binary | `0, 1` | 1 if neither weekend nor holiday; 0 otherwise |
| `weathersit`| Categorical | `1, 2, 3, 4` | 1: Clear/Few clouds, 2: Mist/Cloudy, 3: Light Rain/Snow, 4: Severe Rain/Ice Pellets |
| `temp` | Continuous | `[0.0, 1.0]` | Normalized temperature in Celsius: $T / 41$ |
| `atemp` | Continuous | `[0.0, 1.0]` | Normalized feeling temperature in Celsius: $T_{\text{feel}} / 50$ |
| `hum` | Continuous | `[0.0, 1.0]` | Normalized relative humidity: $H / 100$ |
| `windspeed` | Continuous | `[0.0, 1.0]` | Normalized wind speed: $W / 67$ |
| **`cnt`** | **Target** | **`[1, 977]`** | **Total hourly bike rental count (Supervised Regression Target)** |

---

## 4. Methodology & Pipeline Architecture

```
[ data/hour.csv ]
        │
        ▼
[ Security & Schema Check ] ──> Drops 'casual', 'registered' (leakage) & 'instant' (ID)
        │
        ▼
[ Chronological Split ] ──────> 70% Train (12,165 hrs) | 15% Val (2,606 hrs) | 15% Test (2,608 hrs)
        │
        ▼
[ Feature Engineering ] ──────> Cyclical encoding (hr, mnth), rush-hour flag, weather comfort index
        │
        ▼
[ Model Benchmarking ] ───────> Baseline, Linear Regression, Ridge, Decision Tree, Random Forest, HistGBR
        │
        ▼
[ Validation Selection ] ─────> Best model on Val: HistGradientBoostingRegressor (RMSE: 60.12, R²: 0.9275)
        │
        ▼
[ Test Evaluation ] ──────────> Held-out Test Set evaluated ONCE (RMSE: 66.57, R²: 0.9042)
        │
        ▼
[ Artifact Export ] ──────────> models/bike_rental_model.joblib
        │
        ▼
[ Deployment ] ───────────────> FastAPI (app.py) + Docker Container + Docker Compose
```

### 4.1 Chronological Splitting Strategy
Random K-Fold splitting on time-series records leads to **lookahead data leakage** (using future records to predict the past). To guarantee realistic generalization:
- **Training Set (70%)**: 12,165 hours (2011-01-01 to 2012-05-27)
- **Validation Set (15%)**: 2,606 hours (2012-05-27 to 2012-09-12) — used for hyperparameter tuning & model selection
- **Test Set (15%)**: 2,608 hours (2012-09-12 to 2012-12-31) — evaluated strictly **once** at the end

### 4.2 Feature Engineering (`pipeline_utils.BikeFeatureEngineer`)
1. **Cyclical Transforms**:
   $$\text{hr\_sin} = \sin\left(\frac{2\pi \cdot \text{hr}}{24}\right), \quad \text{hr\_cos} = \cos\left(\frac{2\pi \cdot \text{hr}}{24}\right)$$
   $$\text{mnth\_sin} = \sin\left(\frac{2\pi \cdot (\text{mnth}-1)}{12}\right), \quad \text{mnth\_cos} = \cos\left(\frac{2\pi \cdot (\text{mnth}-1)}{12}\right)$$
2. **Commute Rush Hour Flag**: Binary flag indicating hours $[7, 8, 9, 16, 17, 18, 19]$ on working days (`workingday == 1`).
3. **Late Night Flag**: Binary flag indicating hours $[0, 1, 2, 3, 4, 5]$.
4. **Weekend Leisure Flag**: Binary flag indicating hours $[10, 11, 12, 13, 14, 15, 16]$ on non-working days.
5. **Weather Comfort Index**: $\text{temp} \times (1.0 - \text{hum})$.
6. **Feeling Temperature Differential**: $\text{atemp} - \text{temp}$.
7. **Adverse Weather Indicator**: Binary indicator when $\text{weathersit} \ge 3$.

---

## 5. Actual Evaluation Results & Benchmark

All metrics were computed directly from `data/hour.csv` without fabricated numbers:

### 5.1 Validation Set Benchmark (Hyperparameter Tuning)
| Model | RMSE (bikes) | MAE (bikes) | $R^2$ Score | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Train Mean)** | 253.93 | 194.08 | -0.2939 | Predicts constant training mean ($\bar{y} = 153.7$) |
| **Linear Regression (OLS)** | 136.04 | 108.69 | 0.6286 | Standard Ordinary Least Squares |
| **Ridge Regression** | 136.69 | 109.18 | 0.6250 | L2 regularized ($\alpha = 10.0$) |
| **Decision Tree Regressor** | 74.77 | 49.65 | 0.8878 | Max depth = 10, min samples leaf = 5 |
| **Random Forest Regressor** | 69.93 | 46.30 | 0.9019 | 150 trees, max depth = 16, min samples leaf = 4 |
| **Gradient Boosting (GBR)** | 67.01 | 45.75 | 0.9099 | 150 estimators, max depth = 5, learning rate = 0.08 |
| **HistGradientBoosting (Champion)** | **60.12** | **40.24** | **0.9275** | **Tuned: lr=0.08, depth=6, l2=0.1, max_iter=200** |

### 5.2 Champion Model Selection & Unseen Test Set Evaluation
The **HistGradientBoostingRegressor** outperformed all candidate models across RMSE, MAE, and $R^2$.

Following ML Zoomcamp best practice, the champion pipeline was retrained on the combined **Train + Validation** set (14,771 hours) and evaluated **strictly once** on the held-out **Test Set** (2,608 hours):

| Metric | Champion Validation Score | Final Unseen Test Score |
| :--- | :---: | :---: |
| **RMSE (Root Mean Squared Error)** | **60.12** | **66.57** |
| **MAE (Mean Absolute Error)** | **40.24** | **43.45** |
| **$R^2$ (Coefficient of Determination)** | **0.9275** | **0.9042** |

The final model explains **90.4% of the variance** on completely unseen future data with an average absolute error of only ~43 bicycles per hour.

---

## 6. Project Directory Structure

```
.
├── data/
│   ├── hour.csv                     # Primary dataset (17,379 records)
│   ├── day.csv                      # Daily aggregated dataset
│   └── Readme.txt                   # UCI dataset documentation
├── models/
│   ├── bike_rental_model.joblib     # Serialized end-to-end model pipeline
│   └── metrics.json                 # Training & test evaluation metrics
├── tests/
│   ├── conftest.py                  # Pytest configuration and sys.path setup
│   ├── test_predict.py              # Unit tests for inference & feature engineering
│   └── test_api.py                  # Integration tests for FastAPI endpoints & validation
├── notebook.ipynb                   # Executed Jupyter Notebook with EDA, plots & benchmark
├── pipeline_utils.py                # Reusable custom scikit-learn feature transformers
├── train.py                         # Standalone reproducible training script
├── predict.py                       # Standalone prediction script & CLI interface
├── app.py                           # Production FastAPI web service
├── requirements.txt                 # Pinned dependencies
├── Dockerfile                       # Multi-stage container definition
├── docker-compose.yml               # Container orchestration configuration
├── benchmark_results.json           # Raw benchmark metrics JSON
└── README.md                        # Project documentation (this file)
```

---

## 7. Installation & Quickstart

### 7.1 Clone and Environment Setup
```bash
# Navigate to the project directory
cd Emergency_network

# Create and activate a Python virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 7.2 Train the Pipeline
To re-run the complete training workflow from scratch:
```bash
python train.py
```
This script will:
1. Load `data/hour.csv`.
2. Verify target leakage columns and drop them.
3. Perform the 70/15/15 chronological split.
4. Train the champion pipeline on Train + Validation.
5. Evaluate on the Test set.
6. Export `models/bike_rental_model.joblib` and `models/metrics.json`.

---

## 8. Prediction & API Usage

### 8.1 CLI Prediction (`predict.py`)
Run inference with the default summer peak-hour sample:
```bash
python predict.py
```

Pass a custom JSON payload:
```bash
python predict.py --json "{\"season\": 3, \"yr\": 1, \"mnth\": 7, \"hr\": 17, \"holiday\": 0, \"weekday\": 3, \"workingday\": 1, \"weathersit\": 1, \"temp\": 0.82, \"atemp\": 0.7879, \"hum\": 0.44, \"windspeed\": 0.2239}"
```
**Output**:
```
Input features:
{
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

-> Predicted Bike Rental Demand (cnt): 774.94
```

---

### 8.2 Serving the FastAPI Application (`app.py`)
Start the FastAPI server locally:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

#### Health Check Endpoint
```bash
curl -X GET http://localhost:8000/health
```
**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "HistGradientBoostingRegressor",
  "api_version": "1.0.0",
  "model_path": "models/bike_rental_model.joblib"
}
```

#### Prediction Endpoint (`POST /predict`)
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```
**Response**:
```json
{
  "predicted_demand": 774.94,
  "predicted_demand_rounded": 775,
  "model_version": "1.0.0",
  "status": "success"
}
```

#### Python Client Example
```python
import requests

url = "http://localhost:8000/predict"
payload = {
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

response = requests.post(url, json=payload)
print(response.json())
# Output: {'predicted_demand': 774.94, 'predicted_demand_rounded': 775, 'model_version': '1.0.0', 'status': 'success'}
```

---

## 9. Containerization with Docker & Docker Compose

### 9.1 Build and Run with Docker
```bash
# Build the Docker image
docker build -t bike-rental-demand-api:latest .

# Run the container mapping port 8000
docker run -d -p 8000:8000 --name bike-rental-service bike-rental-demand-api:latest

# Check container logs
docker logs -f bike-rental-service

# Test the container health check
curl -f http://localhost:8000/health
```

### 9.2 Orchestration with Docker Compose
```bash
# Start service in the background
docker-compose up --build -d

# View status
docker-compose ps

# Stop service
docker-compose down
```

---

## 10. Automated Testing

All prediction routines, pipeline transformers, schema validations, and API endpoints are tested using `pytest`.

Run the test suite:
```bash
python -m pytest tests/ -v
```

**Actual Test Results**:
```
tests/test_api.py::test_root_endpoint PASSED                             [  8%]
tests/test_api.py::test_health_check_endpoint PASSED                     [ 16%]
tests/test_api.py::test_predict_endpoint_valid_payload PASSED            [ 25%]
tests/test_api.py::test_predict_endpoint_invalid_hour PASSED             [ 33%]
tests/test_api.py::test_predict_endpoint_invalid_normalized_range PASSED [ 41%]
tests/test_api.py::test_predict_endpoint_invalid_season PASSED           [ 50%]
tests/test_api.py::test_predict_batch_endpoint PASSED                    [ 58%]
tests/test_predict.py::test_feature_engineer_transformation PASSED       [ 66%]
tests/test_predict.py::test_model_loading PASSED                         [ 75%]
tests/test_predict.py::test_single_prediction PASSED                     [ 83%]
tests/test_predict.py::test_batch_prediction PASSED                      [ 91%]
tests/test_predict.py::test_missing_feature_raises_error PASSED          [100%]

====================== 12 passed in 6.49s =======================
```

---

## 11. Project Limitations & Future Improvements

1. **System Expansion & Year Drift**: The dataset exhibits strong year-over-year growth from 2011 to 2012 due to system expansion (adding stations and bikes). While the model captures the `yr` feature, deploying to subsequent years (e.g. 2013+) requires either dynamic station-count normalization or ongoing retuning.
2. **Station-Level Granularity**: The current model predicts aggregate system-wide hourly demand. In future iterations, hierarchical forecasting or station-cluster modeling (e.g., residential origin stations vs commercial destination stations) would enable pin-point rebalancing instructions.
3. **Real-Time Weather Feed Integration**: In production, connecting the API to a live OpenWeatherMap or National Weather Service API would allow fully automated 24-hour lookahead forecasting.
4. **Monitoring & Drift Detection**: Integrating Evidently AI or Prometheus metrics to monitor prediction distribution drift against incoming request data.
