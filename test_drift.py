import numpy as np
import pandas as pd
from drift_detector import DriftDetector


def run_drift_simulation():
    np.random.seed(42)
    n_samples = 1000

    # Establish synthetic dataset template baseline distribution
    reference_data = pd.DataFrame({
        'temperature': np.random.normal(75, 15, n_samples),
        'vibration': np.random.normal(0.5, 0.2, n_samples),
        'pressure': np.random.normal(100, 20, n_samples)
    })

    detector = DriftDetector(reference_data)

    # Test Case 1: Validate stability parameters
    print('Test 1: No Drift Expected')
    current_data_no_drift = pd.DataFrame({
        'temperature': np.random.normal(75, 15, 500),
        'vibration': np.random.normal(0.5, 0.2, 500),
        'pressure': np.random.normal(100, 20, 500)
    })
    drift, results = detector.check_all_features(current_data_no_drift)
    print(f'Drift Detected: {drift}\n')

    # Test Case 2: Induce target data mean mutations
    print('Test 2: Drift Expected')
    current_data_drift = pd.DataFrame({
        'temperature': np.random.normal(95, 15, 500),  # Altered Mean
        'vibration': np.random.normal(0.8, 0.2, 500),   # Altered Mean
        'pressure': np.random.normal(100, 20, 500)
    })
    drift, results = detector.check_all_features(current_data_drift)
    print(f'Drift Detected: {drift}')
    print('\nDrift Details:')
    for feature, result in results.items():
        if result['drift_detected']:
            print(f" - {feature}: DRIFT DETECTED (p={result['p_value']:.4f})")

    detector.save_report()
    print('\nDrift report saved to drift_report.json')


if __name__ == '__main__':
    run_drift_simulation()
