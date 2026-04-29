# test_flow.py

import requests
import json
import random
import time
import statistics

BASE_URL = "http://localhost:8000"
USER_ID = "test_user_001"

NUM_STEPS = 15
TIMEOUT = 5


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
def generate_sequence():
    return [
        [round(random.uniform(0.2, 0.8), 3) for _ in range(11)]
        for _ in range(5)
    ]


def simulate_actual(base_prediction):
    noise = random.uniform(-0.2, 0.4)
    actual = base_prediction + noise
    return max(0.0, min(1.0, actual))


# ─────────────────────────────────────────────
# SAFE REQUEST
# ─────────────────────────────────────────────
def safe_post(url, payload):
    try:
        res = requests.post(url, json=payload, timeout=TIMEOUT)
        if res.status_code != 200:
            print(f"❌ HTTP {res.status_code}: {res.text}")
            return None
        return res.json()
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN TEST
# ─────────────────────────────────────────────
def run_test():
    print("\n🚀 Starting SYSTEM TEST (ML + Adaptation + API)\n")

    errors = []
    latencies = []
    adaptation_flags = []

    for step in range(NUM_STEPS):
        sequence = generate_sequence()

        # ── PREDICT ─────────────────────────────
        start = time.time()

        res = safe_post(f"{BASE_URL}/predict", {
            "user_id": USER_ID,
            "sequence": sequence
        })

        latency = round((time.time() - start) * 1000, 2)

        if not res:
            print("❌ Prediction failed, stopping test")
            return

        print(f"\n--- STEP {step + 1} ---")
        print(json.dumps(res, indent=2))

        # Validate schema
        required_keys = ["severity", "prediction_id"]
        for key in required_keys:
            if key not in res:
                print(f"❌ Missing key: {key}")
                return

        pred_id = res["prediction_id"]
        base = res.get("base_severity", 0.5)
        adapted = res.get("severity", 0.5)

        latencies.append(latency)
        adaptation_flags.append(res.get("adapted", False))

        # ── SIMULATE FEEDBACK ───────────────────
        actual = simulate_actual(base)

        print(f"Simulated actual: {round(actual, 4)}")

        # ── SEND FEEDBACK ───────────────────────
        fb = safe_post(f"{BASE_URL}/feedback", {
            "user_id": USER_ID,
            "prediction_id": pred_id,
            "actual_severity": actual
        })

        if not fb:
            print("❌ Feedback failed")
            return

        print("Feedback:", fb)

        # ── ERROR TRACKING ──────────────────────
        error = abs(actual - adapted)
        errors.append(error)

        print(f"Error: {round(error, 4)} | Latency: {latency} ms")

        time.sleep(0.3)

    # ─────────────────────────────────────────────
    # FINAL ANALYSIS
    # ─────────────────────────────────────────────
    print("\n📊 FINAL ANALYSIS\n")

    if errors:
        print(f"Mean Error (MAE): {round(statistics.mean(errors), 4)}")
        print(f"Std Dev Error: {round(statistics.stdev(errors), 4)}")
        print(f"Min Error: {round(min(errors), 4)}")
        print(f"Max Error: {round(max(errors), 4)}")

    if latencies:
        print(f"\nLatency Avg: {round(statistics.mean(latencies), 2)} ms")
        print(f"Latency Max: {round(max(latencies), 2)} ms")

    print(f"\nAdaptation triggered: {sum(adaptation_flags)} / {NUM_STEPS}")

    # Convergence check
    if len(errors) > 5:
        first_half = statistics.mean(errors[:len(errors)//2])
        second_half = statistics.mean(errors[len(errors)//2:])

        print(f"\nError Trend:")
        print(f"  First Half MAE: {round(first_half, 4)}")
        print(f"  Second Half MAE: {round(second_half, 4)}")

        if second_half < first_half:
            print("✅ Model is adapting (improving)")
        else:
            print("⚠️ Model not improving (possible instability)")

    print("\n✅ Test completed\n")


# ─────────────────────────────────────────────
# EDGE CASE TESTS
# ─────────────────────────────────────────────
def run_edge_tests():
    print("\n🧪 Running edge case tests...\n")

    # Invalid sequence
    res = safe_post(f"{BASE_URL}/predict", {
        "user_id": USER_ID,
        "sequence": []
    })
    print("Empty sequence response:", res)

    # Missing user_id
    res = safe_post(f"{BASE_URL}/predict", {
        "sequence": generate_sequence()
    })
    print("Missing user_id response:", res)

    print("\n✅ Edge tests done\n")


if __name__ == "__main__":
    run_test()
    run_edge_tests()
