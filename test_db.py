"""
MenoEaze — End-to-End System Validation Script
Tests: DB insert/fetch, sequence build, and /run endpoint
"""
import sys
import time
import requests
from supabase import create_client
from datetime import datetime, timezone

# ── CONFIG ──
SUPABASE_URL = "https://wjxliokqbwqkypvceucg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndqeGxpb2txYndxa3lwdmNldWNnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NzcwNTU1MiwiZXhwIjoyMDkzMjgxNTUyfQ.Xc3kKDzTGi_EYyaTI_sqfFoGx90Yt0s91xzTzyOJ1TU"

DEMO_USER_ID = "00000000-0000-0000-0000-000000000000"
API_URL = "http://localhost:8000"

client = create_client(SUPABASE_URL, SUPABASE_KEY)

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        result = fn()
        if result:
            print(f"  ✅ {name}")
            passed += 1
        else:
            print(f"  ❌ {name} — returned False")
            failed += 1
    except Exception as e:
        print(f"  ❌ {name} — {e}")
        failed += 1

# ═══════════════════════════════════════════════
# PHASE 0: Ensure demo user exists
# ═══════════════════════════════════════════════
print("\n═══ PHASE 0: Ensure Demo User ═══")

def ensure_demo_user():
    # Check if user exists
    res = client.table("users").select("id").eq("id", DEMO_USER_ID).execute()
    if res.data and len(res.data) > 0:
        print("  ℹ️  Demo user already exists")
        return True

    # Try to insert via auth.users first (needed for FK)
    # If RLS blocks this, we need service_role to create in auth.users
    try:
        res = client.table("users").insert({
            "id": DEMO_USER_ID,
            "email": "demo@menoeaze.local",
        }).execute()
        return bool(res.data)
    except Exception as e:
        print(f"  ⚠️  Could not create demo user in public.users: {e}")
        print("  ℹ️  You may need to create this user via Supabase dashboard or SQL:")
        print(f"     INSERT INTO auth.users (id, email) VALUES ('{DEMO_USER_ID}', 'demo@menoeaze.local');")
        print(f"     INSERT INTO public.users (id, email) VALUES ('{DEMO_USER_ID}', 'demo@menoeaze.local');")
        return False

test("Demo user exists", ensure_demo_user)

# ═══════════════════════════════════════════════
# PHASE 1: DB Insert Tests (5 logs)
# ═══════════════════════════════════════════════
print("\n═══ PHASE 1: Insert 5 Symptom Logs ═══")

sample_vectors = [
    [3, 2, 6, 6, 3, 3, 5, 4, 2, 50, 25],
    [5, 4, 4, 4, 5, 5, 3, 6, 1, 50, 25],
    [2, 1, 7, 7, 2, 2, 6, 3, 3, 50, 25],
    [4, 3, 5, 5, 4, 4, 4, 5, 2, 50, 25],
    [6, 5, 3, 3, 6, 6, 2, 7, 1, 50, 25],
]

for i, vec in enumerate(sample_vectors):
    def insert_log(v=vec, idx=i):
        res = client.table("symptom_logs").insert({
            "user_id": DEMO_USER_ID,
            "feature_vector": v,
            "notes": f"Test log {idx + 1}",
            "emoji": "🙂",
        }).execute()
        return bool(res.data)

    test(f"Insert log {i + 1}", insert_log)
    time.sleep(0.3)

# ═══════════════════════════════════════════════
# PHASE 2: DB Fetch Test
# ═══════════════════════════════════════════════
print("\n═══ PHASE 2: Fetch Recent Logs ═══")

def fetch_logs():
    res = client.table("symptom_logs")\
        .select("*")\
        .eq("user_id", DEMO_USER_ID)\
        .order("created_at", desc=True)\
        .limit(10)\
        .execute()
    count = len(res.data) if res.data else 0
    print(f"  ℹ️  Found {count} logs for demo user")
    return count >= 5

test("Fetch ≥ 5 logs", fetch_logs)

# ═══════════════════════════════════════════════
# PHASE 3: API /run Tests
# ═══════════════════════════════════════════════
print("\n═══ PHASE 3: API /run Endpoint Tests ═══")

def test_log_api():
    r = requests.post(f"{API_URL}/run", json={
        "action": "log",
        "payload": {
            "user_id": DEMO_USER_ID,
            "feature_vector": [3, 2, 6, 6, 3, 3, 5, 4, 2, 50, 25],
            "notes": "API test log",
            "emoji": "🔬",
        }
    }, timeout=10)
    data = r.json()
    if r.status_code != 200:
        print(f"  ℹ️  Status: {r.status_code}, Body: {data}")
    return r.status_code == 200 and data.get("status") == "ok"

test("POST /run action=log", test_log_api)

def test_predict_api():
    r = requests.post(f"{API_URL}/run", json={
        "action": "predict",
        "payload": {
            "user_id": DEMO_USER_ID,
            "symptoms": "I am experiencing hot flashes and night sweats with difficulty sleeping",
        }
    }, timeout=30)
    data = r.json()
    if r.status_code != 200:
        print(f"  ℹ️  Status: {r.status_code}, Body: {data}")
        return False

    pred = data.get("prediction", {})
    sev = pred.get("severity")
    conf = pred.get("confidence")
    answer = data.get("rag", {}).get("answer", "")

    print(f"  ℹ️  Severity: {sev}, Confidence: {conf}")
    print(f"  ℹ️  RAG answer: {answer[:100]}...")

    return (
        data.get("action") == "predict" and
        isinstance(sev, (int, float)) and
        isinstance(conf, (int, float))
    )

test("POST /run action=predict", test_predict_api)

# ═══════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════
print(f"\n═══ RESULTS: {passed} passed, {failed} failed ═══")
sys.exit(0 if failed == 0 else 1)