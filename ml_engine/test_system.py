import traceback
import json
import uuid
import time
from fastapi.testclient import TestClient

def run_validation():
    print("=== STARTING VALIDATION ===")
    
    # STEP 1: API BOOT
    try:
        print("[Step 1] Booting API...")
        from api import app
        client = TestClient(app)
        print("-> API Boot Success")
        api_boot = True
        api_error = None
    except Exception as e:
        print(f"-> API Boot Failed: {e}")
        traceback.print_exc()
        api_boot = False
        api_error = str(e)

    # We need a user ID for tests
    test_user_id = str(uuid.uuid4())
    print(f"Test User ID: {test_user_id}")

    if api_boot:
        # Before /run, we need to have enough symptoms logged to build a sequence (SEQ_LEN=5)
        print("[Pre-Step] Logging symptoms to build sequence...")
        for i in range(5):
            res = client.post("/log-symptom", json={
                "user_id": test_user_id,
                "feature_vector": [0.5] * 11,
                "notes": "test notes",
                "emoji": "😐"
            })
            if res.status_code != 200:
                print(f"-> Log symptom failed: {res.text}")
        time.sleep(1) # wait for async DB insert and feature build

    # STEP 2 & 3 & 5: /run ENDPOINT & TRACE & RAG
    run_res_json = None
    if api_boot:
        try:
            print("[Step 2] Testing /run endpoint...")
            payload = {
                "user_id": test_user_id,
                "query": "I am having mild hot flashes and poor sleep"
            }
            res = client.post("/run", json=payload)
            print(f"-> /run status code: {res.status_code}")
            
            if res.status_code == 200:
                run_res_json = res.json()
                print(f"-> /run response keys: {list(run_res_json.keys())}")
                if "prediction" in run_res_json:
                    print(f"-> Prediction: {run_res_json['prediction']}")
                if "rag" in run_res_json:
                    print(f"-> RAG: answer len {len(run_res_json['rag'].get('answer', ''))}, sources {run_res_json['rag'].get('sources', [])}")
            else:
                print(f"-> /run error: {res.text}")
        except Exception as e:
            print(f"-> /run exception: {e}")
            traceback.print_exc()

    # STEP 6: GUARDRAIL
    guardrail_triggered = False
    if api_boot:
        try:
            print("[Step 6] Testing Guardrail...")
            payload = {
                "user_id": test_user_id,
                "query": "I am having severe depression and suicidal thoughts, chest pain, fainting."
            }
            res = client.post("/run", json=payload)
            print(f"-> Guardrail status code: {res.status_code}")
            if res.status_code == 200:
                g_json = res.json()
                print(f"-> Guardrail Response: {g_json}")
                if g_json.get("risk", {}).get("override") or g_json.get("doctor", {}).get("recommend"):
                    guardrail_triggered = True
                    print("-> Guardrail successfully triggered.")
            else:
                print(f"-> Guardrail error: {res.text}")
        except Exception as e:
            print(f"-> Guardrail exception: {e}")
            traceback.print_exc()

    # STEP 4: DB VALIDATION
    db_ok = False
    db_issues = []
    try:
        print("[Step 4] DB Validation...")
        from db_client import fetch_table
        # Check if predictions were inserted
        preds = fetch_table("predictions", limit=5)
        if preds is not None:
            print(f"-> Fetched {len(preds)} predictions from DB.")
            db_ok = True
        else:
            print("-> Failed to fetch predictions.")
            db_issues.append("Predictions fetch returned None")
    except Exception as e:
        print(f"-> DB Validation failed: {e}")
        traceback.print_exc()
        db_issues.append(str(e))

    print("=== VALIDATION END ===")

if __name__ == '__main__':
    run_validation()
