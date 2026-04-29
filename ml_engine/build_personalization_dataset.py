# build_personalization_dataset.py

import requests
import numpy as np
import random
import csv

BASE_URL = "http://localhost:8000"
USER_ID = "dataset_user"

OUTPUT_FILE = "personalization_dataset.csv"


def generate_sequence():
    return np.random.uniform(0.2, 0.8, (5, 11)).tolist()


def run_simulation(steps=200):
    data = []

    for i in range(steps):
        sequence = generate_sequence()

        # simulate real-world severity variation
        actual = random.uniform(0.3, 0.9)

        res = requests.post(f"{BASE_URL}/predict", json={
            "user_id": USER_ID,
            "sequence": sequence
        }).json()

        base = res["base_severity"]
        pred_id = res["prediction_id"]

        # fetch internal features indirectly via feedback loop behavior
        # (we recompute proxy features locally)

        # send feedback
        requests.post(f"{BASE_URL}/feedback", json={
            "user_id": USER_ID,
            "prediction_id": pred_id,
            "actual_severity": actual
        })

        # simulate feature extraction (same as API logic)
        # since we don't access internal memory, we approximate
        error = actual - base

        mean_error = error
        trend = random.uniform(-0.1, 0.1)
        variability = abs(random.uniform(0.0, 0.2))

        data.append([
            base,
            mean_error,
            trend,
            variability,
            error  # target = delta
        ])

        if i % 20 == 0:
            print(f"Generated {i} samples")

    return data


def save_csv(data):
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base", "mean_error", "trend", "variability", "target"])

        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    dataset = run_simulation(steps=500)
    save_csv(dataset)
    print(f"\nSaved dataset → {OUTPUT_FILE}")
