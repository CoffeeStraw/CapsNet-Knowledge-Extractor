"""
Simple test file that performs some requests to the API
"""
import json
import requests

# API / getModels
response = requests.get("http://127.0.0.1:5000/api/getModels")
print(json.loads(response.text))

# API / setImage
response = requests.post(
    "http://127.0.0.1:5000/api/setImage", json={"dataset": "MNIST", "index": 1}
)
print(response.text)

# API / computeTrainingStep
response = requests.post(
    "http://127.0.0.1:5000/api/computeStep", json={"model": "Simple", "step": "trained"}
)
print(response.text)

# API / computeTrainingStep
response = requests.post(
    "http://127.0.0.1:5000/api/computeStep",
    json={"model": "Original", "step": "trained"},
)
print(response.text)
