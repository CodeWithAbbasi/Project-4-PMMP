import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrain_model():
    logger.info('Starting automated retraining execution sequence...')
    mlflow.set_experiment('predictive-maintenance')

    n_samples = 10000
    np.random.seed(int(datetime.now().timestamp()))

    # Emulate incoming drift target variants
    temperature = np.random.normal(85, 15, n_samples)
    vibration = np.random.normal(0.6, 0.2, n_samples)
    pressure = np.random.normal(110, 20, n_samples)
    rpm = np.random.normal(1500, 200, n_samples)
    age_days = np.random.randint(0, 365, n_samples)

    failure_score = (
        (temperature > 90) * 0.2 +
        (vibration > 0.8) * 0.3 +
        (pressure > 130) * 0.3 +
        (age_days > 300) * 0.2
    )
    failure_prob = failure_score + np.random.normal(0, 0.1, n_samples)
    failure = (failure_prob > 0.5).astype(int)

    data = pd.DataFrame({
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure,
        'rpm': rpm,
        'age_days': age_days,
        'failure': failure
    })

    logger.info(f'New ingestion profile processed: {len(data)} samples, failure rate: {data.failure.mean():.2%}')

    X = data.drop('failure', axis=1)
    y = data['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info('Training updated XGBoost structural parameters...')
    with mlflow.start_run(run_name='auto_retrain'):
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        new_accuracy = accuracy_score(y_test, y_pred)
        new_roc_auc = roc_auc_score(y_test, y_pred_proba)

        mlflow.log_param('retrain_date', datetime.now().isoformat())
        mlflow.log_metric('accuracy', new_accuracy)
        mlflow.log_metric('roc_auc', new_roc_auc)

        logger.info(f'Retrained metrics profile - Accuracy: {new_accuracy:.4f}, ROC AUC: {new_roc_auc:.4f}')

        run_id = mlflow.active_run().info.run_id
        mlflow.sklearn.log_model(model, 'model', registered_model_name='PredictiveMaintenance')
        logger.info(f'Model registered with run_id: {run_id}')

    return new_roc_auc


if __name__ == '__main__':
    retrain_model()
