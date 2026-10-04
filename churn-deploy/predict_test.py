"""Send a test customer to the service. Works locally and on AWS.

Usage:
    python predict_test.py                      # local (http://localhost:9696)
    python predict_test.py <your-eb-url>        # e.g. churn-serving-env.eba-xxxx.ap-south-1.elasticbeanstalk.com
"""
import sys

import requests

host = sys.argv[1] if len(sys.argv) > 1 else "localhost:9696"
if not host.startswith("http"):
    host = "http://" + host
url = f"{host}/predict"

customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}

response = requests.post(url, json=customer, timeout=30).json()
print(response)

if response["churn"]:
    print("Customer is likely to churn - send a promo email")
else:
    print("Customer is not likely to churn")
