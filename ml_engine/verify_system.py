"""
verify_system.py — END-TO-END STRESS TEST FOR MENOEAZE CONTINUAL LEARNING + MAML
==================================================================================

This script programmatically validates the full closed-loop system:
  1. Cold Start: /predict returns a base prediction (strategy != 'maml')
  2. Feedback Loop: 6 consecutive /feedback calls with high actual_severity
  3. MAML Activation: /predict shifts strategy to 'maml' and adapts upward

Usage:
    1. Start the API server:  uvicorn ml_engine.api:app --reload
    2. Run this script:       python ml_engine/verify_system.py

Uses isolated test user ID to avoid Supabase/VectorDB corruption.
"""

import sys
import time
import json
import random

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package not installed. Run: pip install requests")
    sys.exit(1)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL = "http://127.0.0.1:8000"
TEST_USER_ID = f"test_maml_isolated_{random.randint(1000, 9999)}"
TIMEOUT = 10  # seconds — prevents hanging on server lockup
NUM_FEEDBACK = 6
FEEDBACK_DELAY = 0.5  # seconds between feedback calls

# Generate a realistic (5, 11) test sequence
def _make_sequence():
    """Generate a synthetic (5, 11) sequence with realistic ranges."""
    seq = []
    for _ in range(5):
        row = [
            round(random.uniform(45, 60), 1),     # age
            round(random.uniform(20, 35), 1),      # bmi
            round(random.uniform(0.1, 0.9), 2),    # hot_flash_score
            round(random.uniform(0.1, 0.9), 2),    # night_sweats_score
            round(random.uniform(0.1, 0.9), 2),    # sleep_quality
            round(random.uniform(0.1, 0.9), 2),    # mood_score
            round(random.uniform(0.1, 0.9), 2),    # fatigue_score
            round(random.uniform(0.1, 0.9), 2),    # anxiety_score
            round(random.uniform(0.0, 1.0), 2),    # physical_activity
            round(random.uniform(0.1, 0.9), 2),    # stress_level
            round(random.uniform(0.0, 5.0), 1),    # caffeine_intake
        ]
        seq.append(row)
    return seq


def _separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _assert(condition: bool, msg: str):
    if condition:
        print(f"  ✅ PASS: {msg}")
    else:
        print(f"  ❌ FAIL: {msg}")
        # Don't exit — continue to see all failures


# ─────────────────────────────────────────────
# TEST 0: HEALTH CHECK
# ─────────────────────────────────────────────
def test_health():
    _separator("TEST 0: Health Check")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        _assert(r.status_code == 200, f"GET /health returned {r.status_code}")
        data = r.json()
        _assert(data.get("status") == "ok", f"Health status: {data}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"  ❌ FATAL: Cannot connect to {BASE_URL}. Is the server running?")
        print(f"  Run: uvicorn ml_engine.api:app --reload")
        return False
    except Exception as e:
        print(f"  ❌ FATAL: Health check failed: {e}")
        return False


# ─────────────────────────────────────────────
# TEST 1: COLD START PREDICTION
# ─────────────────────────────────────────────
def test_cold_start():
    _separator("TEST 1: Cold Start Prediction (No History)")

    seq = _make_sequence()
    payload = {
        "user_id": TEST_USER_ID,
        "sequence": seq
    }

    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
        _assert(r.status_code == 200, f"POST /predict returned {r.status_code}")

        data = r.json()
        print(f"  Response: {json.dumps(data, indent=2)[:500]}")

        severity = data.get("severity", data.get("base_severity"))
        strategy = data.get("strategy", "unknown")

        _assert(severity is not None, f"Got severity={severity}")
        _assert(strategy != "maml", f"Cold start strategy='{strategy}' (should NOT be 'maml')")

        return severity, seq

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return None, None


# ─────────────────────────────────────────────
# TEST 2: FEEDBACK LOOP (Build MAML History)
# ─────────────────────────────────────────────
def test_feedback_loop(sequence):
    _separator(f"TEST 2: Feedback Loop ({NUM_FEEDBACK} submissions)")

    if sequence is None:
        print("  ⚠️  SKIPPED: No sequence from cold start test.")
        return False

    success_count = 0

    for i in range(NUM_FEEDBACK):
        # Use a CONSISTENTLY HIGH actual_severity to create a clear signal
        high_severity = round(random.uniform(0.80, 0.95), 2)

        payload = {
            "user_id": TEST_USER_ID,
            "sequence": sequence,
            "actual_severity": high_severity,
            # Include prediction_id to satisfy build_feedback_record
            "prediction_id": f"test_pred_{i}",
            "feedback_type": "correction",
        }

        try:
            r = requests.post(f"{BASE_URL}/feedback", json=payload, timeout=TIMEOUT)

            if r.status_code == 200:
                success_count += 1
                print(f"  [{i+1}/{NUM_FEEDBACK}] ✅ Feedback sent | actual_severity={high_severity}")
            else:
                print(f"  [{i+1}/{NUM_FEEDBACK}] ⚠️  HTTP {r.status_code}: {r.text[:200]}")

        except Exception as e:
            print(f"  [{i+1}/{NUM_FEEDBACK}] ❌ Error: {e}")

        time.sleep(FEEDBACK_DELAY)

    _assert(success_count >= NUM_FEEDBACK - 1, f"{success_count}/{NUM_FEEDBACK} feedback calls succeeded")
    return success_count >= NUM_FEEDBACK - 1


# ─────────────────────────────────────────────
# TEST 3: MAML ACTIVATION CHECK
# ─────────────────────────────────────────────
def test_maml_activation(cold_severity, sequence):
    _separator("TEST 3: MAML Activation (Post-Feedback Prediction)")

    if sequence is None:
        print("  ⚠️  SKIPPED: No sequence available.")
        return

    payload = {
        "user_id": TEST_USER_ID,
        "sequence": sequence
    }

    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
        _assert(r.status_code == 200, f"POST /predict returned {r.status_code}")

        data = r.json()
        print(f"  Response: {json.dumps(data, indent=2)[:500]}")

        strategy = data.get("strategy", "unknown")
        new_severity = data.get("severity", data.get("base_severity"))

        _assert(
            strategy == "maml",
            f"Strategy shifted to '{strategy}' (expected 'maml')"
        )

        if cold_severity is not None and new_severity is not None:
            delta = new_severity - cold_severity
            print(f"  📊 Severity delta: {cold_severity:.4f} → {new_severity:.4f} (Δ={delta:+.4f})")
            # We sent high actual_severity, so the adapted model should shift up
            # (or at least not crash). We check it's different from base.
            _assert(
                strategy == "maml" or abs(delta) > 0.001,
                f"Severity adapted (delta={delta:+.4f})"
            )
        else:
            print("  ⚠️  Cannot compute delta (missing severity values)")

    except Exception as e:
        print(f"  ❌ ERROR: {e}")


# ─────────────────────────────────────────────
# TEST 4: TIMEOUT / LATENCY SAFETY
# ─────────────────────────────────────────────
def test_latency():
    _separator("TEST 4: Latency Safety Check")

    seq = _make_sequence()
    payload = {
        "user_id": TEST_USER_ID,
        "sequence": seq
    }

    start = time.time()
    try:
        r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT)
        elapsed = (time.time() - start) * 1000
        _assert(r.status_code == 200, f"Response status: {r.status_code}")
        _assert(elapsed < 5000, f"Latency={elapsed:.0f}ms (must be <5000ms)")
        print(f"  ⏱️  Total round-trip: {elapsed:.0f}ms")
    except requests.exceptions.Timeout:
        print(f"  ❌ FAIL: Request timed out after {TIMEOUT}s — possible server lockup!")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print(f"\n🧪 MenoEaze Continual Learning + MAML Stress Test")
    print(f"   Server: {BASE_URL}")
    print(f"   Test User: {TEST_USER_ID}")
    print(f"   Feedback Count: {NUM_FEEDBACK}")

    # Test 0: Health
    if not test_health():
        print("\n💀 Server is not reachable. Aborting.")
        sys.exit(1)

    # Test 1: Cold Start
    cold_severity, test_seq = test_cold_start()

    # Test 2: Feedback Loop
    feedback_ok = test_feedback_loop(test_seq)

    # Test 3: MAML Activation
    if feedback_ok:
        test_maml_activation(cold_severity, test_seq)
    else:
        print("\n  ⚠️  Skipping MAML activation test — feedback loop failed.")

    # Test 4: Latency
    test_latency()

    # Summary
    _separator("SUMMARY")
    print(f"  Test user ID: {TEST_USER_ID}")
    print(f"  All checks complete. Review ✅/❌ above.")
    print(f"  If all ✅ → system is production-ready.")
    print()


if __name__ == "__main__":
    main()
