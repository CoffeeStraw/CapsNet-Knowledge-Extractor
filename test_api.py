"""
Simple test file that performs some requests to the API
"""
import requests

# API / getModels
response = requests.get("http://127.0.0.1:5000/api/getModels")
print(response.text)

# API / computeTrainingStep
response = requests.post(
    "http://127.0.0.1:5000/api/computeStep",
    json={"model": "Simple", "step": "trained", "dataset": "MNIST", "index": 0},
)
print(response.text)

# API / computeTrainingStep
response = requests.post(
    "http://127.0.0.1:5000/api/computeStep",
    json={"model": "Original", "step": "trained", "dataset": "MNIST", "index": 42},
)
print(response.text)
