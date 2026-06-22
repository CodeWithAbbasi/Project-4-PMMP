import requests
import time
import os
from datetime import datetime

API_URL = 'http://localhost:8000'


def display_metrics():
    try:
        response = requests.get(f'{API_URL}/metrics')
        if response.status_code == 200:
            metrics = response.json()
            os.system('cls' if os.name == 'nt' else 'clear')
            print('\n' + '=' * 50)
            print(f'METRICS DASHBOARD - {datetime.now().strftime("%H:%M:%S")}')
            print('=' * 50)
            print(f'Total Requests:     {metrics["total_requests"]}')
            print(f'Failures Predicted: {metrics["failures_predicted"]}')
            print(f'Failure Rate:       {metrics["failure_rate"]:.1%}')
            print(f'Avg Latency:        {metrics["avg_latency_ms"]:.2f} ms')
            print(f'Errors Encountered: {metrics["errors"]}')
            print('=' * 50)
        else:
            print(f"Failed to fetch metrics. Status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("Waiting for API connection... Make sure uvicorn is running.")


if __name__ == '__main__':
    print('Starting Monitoring Dashboard Engine...')
    while True:
        try:
            display_metrics()
            time.sleep(5)
        except KeyboardInterrupt:
            print('\nMonitoring stopped cleanly.')
            break
