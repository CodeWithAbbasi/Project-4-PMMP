import requests
import json

API_URL = 'http://localhost:8000'


def run_tests():
    print('Testing /health endpoint...')
    try:
        health_res = requests.get(f'{API_URL}/health')
        print(f'Status: {health_res.status_code}')
        print(f'Response: {health_res.json()}\n')
    except Exception as e:
        print(f"Health check failed: {e}\n")

    test_cases = [
        {
            'name': 'Normal Operation',
            'data': {
                'temperature': 70.0, 'vibration': 0.4, 'pressure': 95.0, 'rpm': 1500.0, 'age_days': 100
            }
        },
        {
            'name': 'High Risk',
            'data': {
                'temperature': 95.0, 'vibration': 0.9, 'pressure': 135.0, 'rpm': 1500.0, 'age_days': 320
            }
        }
    ]

    for test in test_cases:
        print(f"Testing: {test['name']}")
        try:
            response = requests.post(f'{API_URL}/predict', json=test['data'])
            result = response.json()
            print(f' Will Fail:   {result.get("will_fail")}')
            print(f' Probability: {result.get("probability"):.3f}')
            print(f' Latency:     {result.get("latency_ms")} ms\n')
        except Exception as e:
            print(f"Failed handling input profile {test['name']}: {e}")


if __name__ == '__main__':
    run_tests()
