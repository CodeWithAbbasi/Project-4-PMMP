from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.pyfunc
import pandas as pd
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title='Predictive Maintenance API',
    description='ML API for equipment failure prediction',
    version='1.0.0'
)

# Global metrics storage
metrics = {
    'total_requests': 0,
    'predictions': [],
    'latencies': [],
    'failures_predicted': 0,
    'errors': 0
}

# Load production model
MODEL_NAME = 'PredictiveMaintenance'
MODEL_STAGE = 'Production'
try:
    model_uri = f'models://{MODEL_NAME}/{MODEL_STAGE}'
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f'Loaded model: {model_uri}')
except Exception as e:
    logger.error(f'Failed to load model: {e}')
    model = None


# Request validation model
class PredictionRequest(BaseModel):
    temperature: float = Field(..., description='Temperature in °C')
    vibration: float = Field(..., description='Vibration in mm/s')
    pressure: float = Field(..., description='Pressure in PSI')
    rpm: float = Field(..., description='Rotations per minute')
    age_days: int = Field(..., description='Days since last maintenance')

    class Config:
        json_schema_extra = {
            'example': {
                'temperature': 85.0,
                'vibration': 0.7,
                'pressure': 110.0,
                'rpm': 1500.0,
                'age_days': 200
            }
        }


# Response evaluation model
class PredictionResponse(BaseModel):
    will_fail: bool
    probability: float
    recommendation: str
    latency_ms: float
    timestamp: str


@app.get('/')
def root():
    return {
        'message': 'Predictive Maintenance API',
        'version': '1.0.0',
        'docs': '/docs'
    }


@app.get('/health')
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    return {'status': 'healthy', 'model_loaded': True}


@app.post('/predict', response_model=PredictionResponse)
def predict(request: PredictionRequest):
    global metrics
    if model is None:
        metrics['errors'] += 1
        raise HTTPException(status_code=503, detail='Model not available')

    try:
        start_time = time.time()

        # Prepare input payload for inference
        input_data = pd.DataFrame([{
            'temperature': request.temperature,
            'vibration': request.vibration,
            'rpm': request.rpm,
            'pressure': request.pressure,
            'age_days': request.age_days
        }])

        # ML model prediction inference
        prediction = model.predict(input_data)[0]
        # In case the model outputs a binary assignment directly, get probability if possible
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1])
        else:
            probability = float(prediction)

        will_fail = bool(probability >= 0.5)
        latency_ms = (time.time() - start_time) * 1000

        # Update system metrics live tracking
        metrics['total_requests'] += 1
        metrics['latencies'].append(latency_ms)
        metrics['predictions'].append(probability)
        if will_fail:
            metrics['failures_predicted'] += 1

        logger.info(f'Prediction: {will_fail}, Prob: {probability:.3f}, Latency: {latency_ms:.2f}ms')

        return PredictionResponse(
            will_fail=will_fail,
            probability=probability,
            recommendation='Schedule maintenance' if will_fail else 'Continue operation',
            latency_ms=round(latency_ms, 2),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        metrics['errors'] += 1
        logger.error(f'Prediction error: {e}')
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/metrics')
def get_metrics():
    avg_latency = sum(metrics['latencies']) / len(metrics['latencies']) if metrics['latencies'] else 0
    failure_rate = metrics['failures_predicted'] / metrics['total_requests'] if metrics['total_requests'] > 0 else 0
    return {
        'total_requests': metrics['total_requests'],
        'failures_predicted': metrics['failures_predicted'],
        'failure_rate': round(failure_rate, 3),
        'avg_latency_ms': round(avg_latency, 2),
        'errors': metrics['errors']
    }
