import json
import os

import requests

backend_host = os.getenv("BACKEND_HOST", "localhost")
backend_port = os.getenv("BACKEND_PORT", "8000")
backend_url = f"http://{backend_host}:{backend_port}/api/v1/actions/update_action"


with open("72_raw_results.json", "r") as f:
    data = json.load(f)

response = requests.put(backend_url, json=data)
print(response.json())
