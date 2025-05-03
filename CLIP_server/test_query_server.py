import requests
import random
import string
import time
import concurrent.futures

# Server configuration
SERVER_URL = "http://localhost:8000" # Adjust if your server runs on a different port
QUERY_ENDPOINT = f"{SERVER_URL}/query_embedding"
NUM_REQUESTS = 10_000
MAX_WORKERS = 1000

def generate_random_text(length=20):
    """Generates a random string of fixed length."""
    letters = string.ascii_lowercase + string.digits + ' '
    return ''.join(random.choice(letters) for i in range(length))

def send_request(request_id):
    """Sends a single request to the query server."""
    random_text = generate_random_text()
    params = {"text": random_text}
    try:
        response = requests.get(QUERY_ENDPOINT, params=params, timeout=15) # Added timeout
        if response.status_code == 200:
            # print(f"Request {request_id}: Success (Status {response.status_code}) for text: '{random_text}'")
            return True, request_id, None
        else:
            print(f"Request {request_id}: Failed (Status {response.status_code}) for text: '{random_text}'. Response: {response.text}")
            return False, request_id, response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Request {request_id}: Error sending request for text '{random_text}': {e}")
        return False, request_id, str(e)

if __name__ == "__main__":
    print(f"Sending {NUM_REQUESTS} requests to {QUERY_ENDPOINT} using up to {MAX_WORKERS} workers...")
    start_time = time.time()
    success_count = 0
    failure_count = 0

    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create futures for all requests
        futures = [executor.submit(send_request, i + 1) for i in range(NUM_REQUESTS)]

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            success, req_id, error_info = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1

    end_time = time.time()
    duration = end_time - start_time

    print("\n--- Test Summary ---")
    print(f"Total requests sent: {NUM_REQUESTS}")
    print(f"Successful requests: {success_count}")
    print(f"Failed requests: {failure_count}")
    print(f"Total time taken: {duration:.2f} seconds")
    if duration > 0:
        print(f"Requests per second: {NUM_REQUESTS / duration:.2f}")