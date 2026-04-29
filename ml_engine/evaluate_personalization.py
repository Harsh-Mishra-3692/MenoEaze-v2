import requests
import numpy as np
import random

BASE_URL = "http://localhost:8000"
USER_ID = "eval_user_v2"

def generate_sequence():
    return np.random.uniform(0.2, 0.8, (5, 11)).tolist()

base_errors = []
adapted_errors = []

for i in range(30):
    sequence = generate_sequence()

    # simulate real variation
    actual = random.uniform(0.3, 0.9)

    res = requests.post(f"{BASE_URL}/predict", json={
        "user_id": USER_ID,
        "sequence": sequence
    }).json()

    base = res["base_severity"]
    adapted = res["severity"]
    pred_id = res["prediction_id"]

    base_errors.append(abs(actual - base))
    adapted_errors.append(abs(actual - adapted))

    requests.post(f"{BASE_URL}/feedback", json={
        "user_id": USER_ID,
        "prediction_id": pred_id,
        "actual_severity": actual
    })

print("\nMAE BASE:", np.mean(base_errors))
print("MAE ADAPTED:", np.mean(adapted_errors))
