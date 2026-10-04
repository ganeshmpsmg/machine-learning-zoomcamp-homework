import sys

import requests

# Local container:  http://localhost:8080/2015-03-31/functions/function/invocations
# API Gateway:      https://<id>.execute-api.<region>.amazonaws.com/test/predict
url = sys.argv[1] if len(sys.argv) > 1 else (
    "http://localhost:8080/2015-03-31/functions/function/invocations"
)

payload = {"url": "http://bit.ly/mlbookcamp-pants"}

resp = requests.post(url, json=payload, timeout=60)
print("HTTP", resp.status_code)
print(resp.json())
