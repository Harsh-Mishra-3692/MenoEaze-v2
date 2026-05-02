# SYSTEM_VALIDATION_REPORT

## 1. System Status
* API: ❌
* Pipeline: ✅
* DB: ✅
* RAG: ✅
* Guardrails: ✅

---

## 2. Execution Logs (Summarized)
* **API startup result**: SUCCESS. `uvicorn api:app --reload` successfully binds port. Dependencies and lazy-loaded models initialized safely without crashing.
* **`/predict` response**: FAILED. 
  - `404 Not Found`: The endpoint `/predict` does not exist; the actual pipeline orchestrator is mapped to `/run`.
  - `422 Unprocessable Entity`: The test payload uses `"symptoms"`, but the `RunRequest` schema strictly requires `"query"`.
  - `400 Bad Request`: If the schema is corrected, requests for new users fail immediately with "Not enough historical data" (requires exactly 5 historical logs via `/log-symptom` to build the GRU sequence).
* **pipeline trace summary**: `API → fetch_recent_logs → predict → early guardrail → full_pipeline (re-predicts, re-guardrails, extracts signal) → RAG → trust/confidence calibration → doctor_recommender → DB writes (prediction, memory)`.

---

## 3. Component Validation

### API Layer
* **status**: PARTIALLY WORKING
* **issues**: 
  - Endpoint mismatch (`/predict` vs `/run`).
  - Schema mismatch (`symptoms` vs `query`).
  - Strict cold-start block: Users with fewer than 5 logged symptoms receive a hard `400` error instead of a degraded prediction.

### ML Pipeline
* **status**: WORKING
* **issues**: 
  - Inefficient redundancy: `predict()` and guardrails are invoked twice per request (once in `api.py` for early exit, and again inside `pipeline.py`).

### RAG
* **status**: WORKING
* **issues**: 
  - Working as intended. Correctly falls back to safe generic strings (`_fallback_answer`) if vector retrieval is empty or grounding validation fails.

### Guardrails
* **status**: WORKING
* **issues**: 
  - Successfully intercepts high-risk emergency inputs (e.g., suicidal thoughts) during the early check in `api.py`, triggering a safe messaging override and preventing expensive RAG execution.

### Database
* **status**: WORKING
* **issues**: 
  - `insert_prediction` and `add_prediction_context` trap exceptions locally in the API layer. DB write failures do not fail the request, returning `200 OK` while silently dropping historical tracking data.

---

## 4. Critical Failures (if any)
1. **Endpoint & Schema Mismatch**: Client expectations (`POST /predict` with `symptoms`) completely break against the enforced API routes and `RunRequest` Pydantic model.
2. **Cold Start Hard Blocker**: The API lacks a bypass for users with `< 5` logs, completely breaking the `/run` endpoint for new accounts.

---

## 5. Silent Failures (IMPORTANT)
1. **Double Execution**: The API orchestration runs the ML prediction and guardrail checks twice per API call due to overlap between `api.py` and `pipeline.py`.
2. **Silent Data Loss**: If the Supabase connection drops during the final telemetry writes, the API logs the error but responds `200 OK`, leading to silent data inconsistency.
3. **Model Degradation Masking**: Exceptions in the PyTorch GRU (`predict`) are caught and return a dummy `severity: 0.5`, hiding actual tensor/shape mismatch crashes from the client.

---

## 6. Risk Assessment
* **Production readiness score (0–10)**: 6
* **Failure probability**: High (for new users due to 400 cold start, and clients using the wrong schema).
* **Data integrity risks**: Medium (due to silent write failures on `predictions` table).

---

## 7. Final Verdict
PARTIALLY WORKING
