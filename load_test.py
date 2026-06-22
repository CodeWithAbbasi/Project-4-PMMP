import requests
import time
import random

API_URL = 'http://localhost:8000'
NUM_REQUESTS = 100


def generate_random_request():
    return {
        'temperature': random.uniform(60.0, 100.0),
        'vibration': random.uniform(0.3, 1.0),
        'pressure': random.uniform(80.0, 140.0),
        'rpm': random.uniform(1400.0, 1600.0),
        'age_days': random.randint(0, 365)
    }


def run_load_test():
    print(f'Starting load test: sending {NUM_REQUESTS} randomized requests...')
    start_time = time.time()
    successes = 0
    failures = 0
    latencies = []

    for i in range(NUM_REQUESTS):
        try:
            req_payload = generate_random_request()
            req_start = time.time()
            response = requests.post(f'{API_URL}/predict', json=req_payload)
            req_time = (time.time() - req_start) * 1000
            latencies.append(req_time)

            if response.status_code == 200:
                successes += 1
            else:
                failures += 1
        except Exception as e:
            failures += 1
            print(f'Error on request {i}: {e}')

        if (i + 1) % 20 == 0:
            print(f'Progress: {i + 1}/{NUM_REQUESTS}')

    total_time = time.time() - start_time
    print('\n' + '=' * 50)
    print('LOAD TEST RESULTS')
    print('=' * 50)
    print(f'Total Requests: {NUM_REQUESTS}')
    print(f'Successful:     {successes}')
    print(f'Failed:         {failures}')
    print(f'Success Rate:   {(successes / NUM_REQUESTS):.1%}')
    print(f'Total Time:     {total_time:.2f}s')
    print(f'Requests/sec:   {NUM_REQUESTS / total_time:.2f}')
    if latencies:
        print(f'Avg Latency:    {sum(latencies) / len(latencies):.2f} ms')
        print(f'Min Latency:    {min(latencies):.2f} ms')
        print(f'Max Latency:    {max(latencies):.2f} ms')
    print('=' * 50)


if __name__ == '__main__':
    run_load_test()
