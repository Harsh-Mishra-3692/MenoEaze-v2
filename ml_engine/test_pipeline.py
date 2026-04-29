"""
test_pipeline.py
================
End-to-end test for the MenoEaze Health Intelligence System.

Simulates a user flow:
1. Health check
2. Base prediction (no user history)
3. Prediction with guidance (RAG)
4. Submit feedback
5. Adapted prediction (after feedback)
6. Verify all responses

Usage:
    # Start API first: uvicorn api:app --port 8000
    python test_pipeline.py
"""

import sys
import json
import time
import uuid
import requests
import numpy as np

API_BASE = "http://localhost:8000"
TEST_USER_ID = f"test-user-{uuid.uuid4().hex[:8]}"


def separator(title: str):
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


def generate_test_sequence() -> list:
    """
    Generate a realistic 5-day symptom sequence.
    Shape: (5, 11) — all values in [0, 1]
    
    Features: hot_flash, night_sweats, sleep_quality, mood,
              fatigue, anxiety, activity, stress, caffeine, age, bmi
    """
    np.random.seed(42)

    days = []
    for d in range(5):
        base = 0.4 + 0.05 * d  # gradually worsening
        day = [
            np.clip(base + np.random.normal(0, 0.1), 0, 1),  # hot_flash
            np.clip(base + np.random.normal(0, 0.1), 0, 1),  # night_sweats
            np.clip(1 - base + np.random.normal(0, 0.1), 0, 1),  # sleep
            np.clip(1 - base + np.random.normal(0, 0.1), 0, 1),  # mood
            np.clip(base + np.random.normal(0, 0.1), 0, 1),  # fatigue
            np.clip(base * 0.8 + np.random.normal(0, 0.1), 0, 1),  # anxiety
            np.clip(0.5 - base * 0.3 + np.random.normal(0, 0.1), 0, 1),  # activity
            np.clip(base * 0.7 + np.random.normal(0, 0.1), 0, 1),  # stress
            np.clip(0.5 + np.random.normal(0, 0.15), 0, 1),  # caffeine
            0.5,  # age (normalized)
            0.45,  # bmi (normalized)
        ]
        days.append([round(v, 4) for v in day])

    return days


def test_health():
    separator("TEST 1: Health Check")

    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        data = resp.json()

        print(f"  Status        : {data.get('status')}")
        print(f"  Model loaded  : {data.get('model_loaded')}")
        print(f"  Scalers loaded: {data.get('scalers_loaded')}")
        print(f"  RAG ready     : {data.get('rag_ready')}")
        print(f"  Groq available: {data.get('groq_available')}")
        print(f"  Device        : {data.get('device')}")

        assert data["status"] == "ok", "Health check failed"
        assert data["model_loaded"], "Model not loaded"
        assert data["scalers_loaded"], "Scalers not loaded"

        print("  ✅ PASSED")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_base_prediction():
    separator("TEST 2: Base Prediction (no user history)")

    sequence = generate_test_sequence()

    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            json={
                "sequence": sequence,
                "request_guidance": False,
            },
            timeout=10,
        )

        data = resp.json()

        severity = data.get("severity")
        base = data.get("base_severity")
        latency = data.get("latency_ms")

        print(f"  Severity      : {severity}")
        print(f"  Base severity : {base}")
        print(f"  Adapted       : {data.get('adapted')}")
        print(f"  User bias     : {data.get('user_bias')}")
        print(f"  Latency       : {latency}ms")

        assert severity is not None, "No severity returned"
        assert 0 <= severity <= 1, f"Severity {severity} out of [0,1]"
        assert latency is not None, "No latency reported"

        print("  ✅ PASSED")
        return data

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return None


def test_prediction_with_guidance():
    separator("TEST 3: Prediction with RAG Guidance")

    sequence = generate_test_sequence()

    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            json={
                "user_id": TEST_USER_ID,
                "sequence": sequence,
                "request_guidance": True,
            },
            timeout=30,
        )

        data = resp.json()

        print(f"  Severity      : {data.get('severity')}")
        print(f"  Severity level: {data.get('severity_level', 'N/A')}")
        print(f"  Prediction ID : {data.get('prediction_id', 'N/A')}")
        print(f"  Latency       : {data.get('latency_ms')}ms")

        guidance = data.get("guidance", "")
        sources = data.get("sources", [])

        print(f"  Sources       : {len(sources)}")
        for s in sources[:3]:
            print(f"    - {s.get('title', 'Unknown')}")

        if guidance:
            print(f"  Guidance      : {guidance[:200]}...")
        else:
            print("  Guidance      : (none — RAG may not be initialized)")

        assert data.get("severity") is not None
        print("  ✅ PASSED")
        return data

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return None


def test_feedback(prediction_id: str):
    separator("TEST 4: Submit Feedback")

    if not prediction_id:
        print("  ⏭ SKIPPED (no prediction_id from previous test)")
        return False

    try:
        resp = requests.post(
            f"{API_BASE}/feedback",
            json={
                "user_id": TEST_USER_ID,
                "prediction_id": prediction_id,
                "actual_severity": 0.65,
            },
            timeout=10,
        )

        data = resp.json()

        print(f"  Status  : {data.get('status')}")
        print(f"  Message : {data.get('message')}")

        print("  ✅ PASSED")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_adapted_prediction():
    separator("TEST 5: Adapted Prediction (after feedback)")

    sequence = generate_test_sequence()

    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            json={
                "user_id": TEST_USER_ID,
                "sequence": sequence,
                "request_guidance": False,
            },
            timeout=10,
        )

        data = resp.json()

        print(f"  Severity      : {data.get('severity')}")
        print(f"  Base severity : {data.get('base_severity')}")
        print(f"  Adapted       : {data.get('adapted')}")
        print(f"  User bias     : {data.get('user_bias')}")
        print(f"  Latency       : {data.get('latency_ms')}ms")

        assert data.get("severity") is not None
        print("  ✅ PASSED")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_backward_compat():
    separator("TEST 6: Backward-Compatible Endpoint (/api/ml-predict)")

    sequence = generate_test_sequence()

    try:
        resp = requests.post(
            f"{API_BASE}/api/ml-predict",
            json={"sequence": sequence},
            timeout=10,
        )

        data = resp.json()
        print(f"  Severity : {data.get('severity')}")
        print(f"  Latency  : {data.get('latency_ms')}ms")

        assert data.get("severity") is not None
        print("  ✅ PASSED")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("  MenoEaze Health Intelligence — E2E Test Suite")
    print(f"  API: {API_BASE}")
    print(f"  User: {TEST_USER_ID}")
    print("=" * 60)

    results = {}

    results["health"] = test_health()

    if not results["health"]:
        print("\n❌ API not reachable. Start it with:")
        print("   uvicorn api:app --port 8000 --reload")
        sys.exit(1)

    results["base_predict"] = test_base_prediction()

    pred_data = test_prediction_with_guidance()
    results["guidance"] = pred_data is not None

    prediction_id = pred_data.get("prediction_id") if pred_data else None
    results["feedback"] = test_feedback(prediction_id)
    results["adapted"] = test_adapted_prediction()
    results["compat"] = test_backward_compat()

    # ── Summary ────────────────────────────────────────────
    separator("SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, ok in results.items():
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")

    print(f"\n  {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed! System is demo-ready.")
    else:
        print("\n  ⚠️  Some tests failed. Check configuration.")

    print("=" * 60)


if __name__ == "__main__":
    main()
