# Project 4: Predictive Maintenance ML Pipeline

A production-ready ML engineering pipeline for equipment failure prediction, built with FastAPI, XGBoost, MLflow, and automated CI/CD retraining.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  API Request │────▶│  FastAPI App │────▶│  MLflow Model   │
│  (JSON)      │     │  (main.py)   │     │  (XGBoost)      │
└─────────────┘     └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │   Metrics    │
                    │  Dashboard   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │    Drift     │
                    │  Detector    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Automated   │
                    │  Retraining  │
                    └──────────────┘
```

## Features

- **FastAPI REST API** — `/predict`, `/health`, `/metrics` endpoints with real-time latency tracking
- **Live Monitoring Dashboard** — CLI dashboard polling metrics every 5 seconds
- **Drift Detection** — Kolmogorov-Smirnov statistical test across all features
- **Automated Retraining** — XGBoost model retraining with MLflow registry
- **CI/CD Pipelines** — GitHub Actions for testing (on push/PR) and weekly retraining
- **Load Testing** — 100-request stress test with latency percentiles

## Project Structure

```
predictive_maintenance_project/
├── .github/workflows/
│   ├── test.yml              # CI: run tests on push/PR
│   └── retrain.yml           # Weekly drift check + retrain
├── api/
│   └── main.py               # FastAPI app with /predict, /health, /metrics
├── tests/
│   └── test_model.py         # Pytest unit tests
├── requirements.txt          # Python dependencies
├── drift_detector.py         # KS-test drift detection
├── test_drift.py             # Drift simulation (no drift + drift scenarios)
├── retrain_pipeline.py       # Automated XGBoost retraining with MLflow
├── monitor_dashboard.py      # Live CLI metrics dashboard
├── test_api.py               # API integration tests (normal + high-risk)
└── load_test.py              # 100-request load test
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start MLflow Tracking Server
```bash
mlflow ui
```

### 3. Launch the FastAPI App
```bash
uvicorn api.main:app --reload
```

### 4. View Metrics Live (separate terminal)
```bash
python monitor_dashboard.py
```

### 5. Run Tests
```bash
python test_api.py        # Integration tests
python test_drift.py      # Drift detection simulation
python load_test.py       # 100-request load test
pytest tests/             # Unit tests
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and docs link |
| `/health` | GET | Health check (verifies model loaded) |
| `/predict` | POST | Predict equipment failure probability |
| `/metrics` | GET | Real-time usage metrics |

### Example Prediction Request
```json
POST /predict
{
  "temperature": 85.0,
  "vibration": 0.7,
  "pressure": 110.0,
  "rpm": 1500.0,
  "age_days": 200
}
```

### Example Response
```json
{
  "will_fail": true,
  "probability": 0.847,
  "recommendation": "Schedule maintenance",
  "latency_ms": 12.34,
  "timestamp": "2025-01-15T10:30:00"
}
```

## CI/CD Workflows

- **test.yml** — Runs on every push/PR to `main`. Installs deps and runs `pytest tests/`.
- **retrain.yml** — Runs weekly (Sunday midnight) or manually. Checks drift, retrains model, uploads drift report.

## Model

- **Algorithm:** XGBoost Classifier
- **Features:** temperature, vibration, pressure, rpm, age_days
- **Target:** binary failure prediction
- **Registry:** MLflow Model Registry (`PredictiveMaintenance` / `Production` stage)
