# api.py — PRODUCTION-GRADE ORCHESTRATOR (HARDENED + STABLE)

import time
import math
import asyncio
from typing import Dict, List, Optional

import torch
import numpy as np
import threading

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager

# Internal modules
from ml_engine.pipeline import full_pipeline
from ml_engine.data_collector import build_feedback_record
from ml_engine.db_client import insert_feedback, check_connection, fetch_table, insert_bulk
from ml_engine.continual_train import train_incremental

import ml_engine.pipeline as pipeline
from ml_engine.adaptation_worker import learning_buffer, start_worker_thread
from ml_engine.model_loader import load_model as ml_load_model

# NEW: use centralized logger
from ml_engine.logger import get_logger, init_logging, set_request_id, clear_request_id

# ─────────────────────────────────────────────
# INIT LOGGING
# ─────────────────────────────────────────────
init_logging()
logger = get_logger("menoeaze.api")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN = 5
FEATURES = 11
MAX_MEMORY = 20

# ─────────────────────────────────────────────
# DB INTEGRATION HELPERS (PHASE 2.4)
# ─────────────────────────────────────────────
async def safe_db_call(fn, *args, **kwargs):
    try:
        res = await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=2.0)
        return True, res
    except asyncio.TimeoutError:
        logger.error("DB timeout")
        return False, "DB timeout"
    except Exception as e:
        logger.error(f"DB call failed: {e}")
        return False, "DB error"

async def _get_db_sequence(user_id: str):
    success, logs = await safe_db_call(fetch_table, "symptom_logs", limit=5, filters={"user_id": user_id}, order_by="created_at")
    if not success:
        return False, logs
        
    if logs and len(logs) >= 3:
        logs.reverse()  # chronological order
        seq = []
        for r in logs:
            vec = r.get("feature_vector", [])
            if isinstance(vec, list) and len(vec) == FEATURES and all(isinstance(v, (int, float)) and math.isfinite(v) for v in vec):
                seq.append(vec)
        if len(seq) == len(logs):
            return True, seq
        return False, "Invalid DB sequence"
    return False, "Not enough data"

async def _load_user_memory(user_id: str):
    success, rows = await safe_db_call(fetch_table, "user_memory", limit=1, filters={"user_id": user_id})
    if success and rows and "data" in rows[0]:
        return True, rows[0]["data"][-MAX_MEMORY:]
    return success, rows if not success else []

async def _save_user_memory(user_id: str, memory_list: list):
    success, res = await safe_db_call(insert_bulk, "user_memory", [{"user_id": user_id, "data": memory_list}], True, "user_id")
    if not success:
        logger.error(f"user_memory save failed: {res}")
    return success

# ─────────────────────────────────────────────
# MODEL HOT SWAP
# ─────────────────────────────────────────────
_current_model_timestamp: int = 0

async def _periodic_model_refresh():
    global _current_model_timestamp
    while True:
        try:
            model, metadata = ml_load_model()
            new_ts = metadata.get("timestamp") or 0

            if new_ts > _current_model_timestamp:
                logger.info(f"Hot-swapping model → ts={new_ts}")

                pipeline._model = model
                pipeline._model.to(pipeline.DEVICE)
                pipeline._model.eval()

                _current_model_timestamp = new_ts

        except Exception as e:
            logger.error(f"Model refresh failed: {e}")

        await asyncio.sleep(60)

# ─────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting...")

    start_worker_thread(check_interval_seconds=60, min_batch_size=50)
    asyncio.create_task(_periodic_model_refresh())

    yield

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = FastAPI(title="MenoEaze AI", lifespan=lifespan)

# ⚠️ FIXED: restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# REQUEST MIDDLEWARE (TRACE ID)
# ─────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(time.time())
    set_request_id(request_id)

    try:
        response = await call_next(request)
        return response
    finally:
        clear_request_id()

# ─────────────────────────────────────────────
# VALIDATION UTIL
# ─────────────────────────────────────────────
def sanitize_sequence(seq):
    arr = np.array(seq, dtype=np.float32)

    if arr.shape != (SEQ_LEN, FEATURES):
        raise ValueError(f"Invalid shape {arr.shape}, expected {(SEQ_LEN, FEATURES)}")

    if not np.isfinite(arr).all():
        logger.error("NaN/Inf detected in input sequence")
        raise ValueError("Sequence contains NaN or Inf values")

    return arr

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class RunRequest(BaseModel):
    user_id: str
    query: str
    sequence: Optional[List[List[float]]] = None

    @field_validator("sequence")
    def validate(cls, v):
        return v if v is None else sanitize_sequence(v).tolist()


class PredictRequest(BaseModel):
    user_id: str
    sequence: List[List[float]]

    @field_validator("sequence")
    def validate(cls, v):
        return sanitize_sequence(v).tolist()


class QueryRequest(BaseModel):
    user_id: str
    query: str

# ─────────────────────────────────────────────
# RESPONSE FORMATTER (BACKWARD COMPATIBILITY)
# ─────────────────────────────────────────────
def format_response(request: Request, response_dict: dict):
    if request.headers.get("x-api-v2", "").lower() == "true":
        return response_dict
    # Legacy fallback
    return response_dict.get("data", response_dict)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    model_loaded = pipeline._model is not None
    db_connected = check_connection()
    llm_ready = pipeline._llm and getattr(pipeline._llm, "api_key", None)
    
    issues = []
    if not model_loaded: issues.append("Model missing")
    if not db_connected: issues.append("DB disconnected")
    if not llm_ready: issues.append("LLM missing")

    return {
        "status": "degraded" if issues else "ok",
        "model": "loaded" if model_loaded else "missing",
        "db": "connected" if db_connected else "down",
        "llm": "ready" if llm_ready else "missing",
        "issues": issues
    }

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
@app.post("/run")
async def run_pipeline(data: RunRequest, request: Request):
    start = time.time()
    logger.info(f"POST /run started | user={data.user_id}")

    try:
        db_error = None
        
        success, res = await _get_db_sequence(data.user_id)
        db_seq = res if success else None
        if not success and res != "Not enough data":
            db_error = res

        mem_success, mem_res = await _load_user_memory(data.user_id)
        history = mem_res if mem_success else []
        if not mem_success:
            db_error = mem_res

        final_seq = np.array(db_seq) if db_seq else (np.array(data.sequence) if data.sequence else None)

        result = full_pipeline(
            user_id=data.user_id,
            query=data.query,
            sequence=final_seq,
            user_history={
                "predictions": [h.get("pred") for h in history],
                "actuals": [h.get("actual") for h in history],
                "feedback_logs": [h for h in history if "sequence" in h],
            }
        )

        entry = {
            "ts": time.time(),
            "query": data.query,
        }

        if result.get("prediction"):
            entry["pred"] = result["prediction"]["severity"]

        history.append(entry)
        history = history[-MAX_MEMORY:]
        save_success, _ = await _save_user_memory(data.user_id, history)

        prediction = result.get("prediction", {})
        rag = result.get("rag", {})
        
        status = "success"
        reason = "OK"
        if "error" in prediction:
            status, reason = "degraded", "Pipeline partially failed (model missing)"
        elif db_error or not save_success:
            status, reason = "degraded", db_error or "Database memory write failed"
        elif "error" in rag or rag.get("fallback"):
            status, reason = "degraded", "Pipeline partially failed (LLM missing)"

        result["latency_ms"] = round((time.time() - start) * 1000, 2)
        logger.info(f"POST /run success | latency={result['latency_ms']}ms")

        return format_response(request, {"status": status, "reason": reason, "data": result})

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})
    except Exception as e:
        logger.exception("POST /run failed")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(data: PredictRequest, request: Request):
    logger.info(f"POST /predict started | user={data.user_id}")
    try:
        db_error = None
        
        success, res = await _get_db_sequence(data.user_id)
        db_seq = res if success else None
        if not success and res != "Not enough data":
            db_error = res

        mem_success, mem_res = await _load_user_memory(data.user_id)
        history = mem_res if mem_success else []
        if not mem_success:
            db_error = mem_res

        final_seq = np.array(db_seq) if db_seq else np.array(data.sequence)

        result = full_pipeline(
            user_id=data.user_id,
            query="",
            sequence=final_seq,
            user_history={"feedback_logs": history}
        )

        prediction = result.get("prediction", {})
        rag = result.get("rag", {})
        
        status = "success"
        reason = "OK"
        if "error" in prediction:
            status, reason = "degraded", "Pipeline partially failed (model missing)"
        elif db_error:
            status, reason = "degraded", db_error
        elif "error" in rag or rag.get("fallback"):
            status, reason = "degraded", "Pipeline partially failed (LLM missing)"

        logger.info(f"POST /predict success | user={data.user_id}")
        return format_response(request, {"status": status, "reason": reason, "data": prediction})

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})
    except Exception as e:
        logger.exception("POST /predict failed")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})

# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────
@app.post("/query")
async def query(data: QueryRequest, request: Request):
    logger.info(f"POST /query started | user={data.user_id}")
    try:
        result = full_pipeline(
            user_id=data.user_id,
            query=data.query,
            sequence=None,
            user_history=None
        )
        rag = result.get("rag", {})
        if "fallback" in rag or "error" in rag:
            return format_response(request, {"status": "degraded", "reason": rag.get("error", "LLM missing or failed"), "data": rag})

        logger.info(f"POST /query success | user={data.user_id}")
        return format_response(request, {"status": "success", "reason": "OK", "data": rag})

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})
    except Exception as e:
        logger.exception("POST /query failed")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})

# ─────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────
_training_lock = asyncio.Lock()
_last_training_ts: float = 0.0

@app.post("/feedback")
async def feedback(data: dict, request: Request):
    global _last_training_ts
    user_id = data.get("user_id", "unknown")
    logger.info(f"POST /feedback started | user={user_id}")
    try:
        # DB write (with timeout to ensure truthfulness but prevent hangs)
        record = build_feedback_record(
            user_id=data.get("user_id", ""),
            predicted=data.get("predicted", 0.0),
            actual=data.get("actual", data.get("actual_severity", 0.0)),
            error=data.get("error", 0.0),
        )
        
        try:
            success = await asyncio.wait_for(
                asyncio.to_thread(insert_feedback, record),
                timeout=2.0
            )
            if not success:
                logger.error(f"Feedback record failed for user {user_id}")
                return format_response(request, {"status": "degraded", "reason": "Database write failed", "data": {}})
        except asyncio.TimeoutError:
            logger.error(f"Feedback DB write timeout for user {user_id}")
            return format_response(request, {"status": "degraded", "reason": "Database write timeout", "data": {}})

        # sanitize
        sequence = data.get("sequence")
        actual = data.get("actual_severity")

        if sequence is not None and actual is not None:
            seq = sanitize_sequence(sequence)
            actual = float(actual) if math.isfinite(float(actual)) else 0.0

            # Prevent consecutive duplicates
            seq_t = torch.tensor(seq, dtype=torch.float32)
            y_t = torch.tensor([actual], dtype=torch.float32)
            is_dup = False
            if learning_buffer.buffer:
                last_x, last_y = learning_buffer.buffer[-1]
                if torch.allclose(last_x, seq_t) and torch.allclose(last_y, y_t):
                    is_dup = True
            
            if not is_dup:
                learning_buffer.add_sample(seq_t, y_t)
                if len(learning_buffer.buffer) > 50:
                    learning_buffer.buffer.pop(0)  # drop oldest

            # memory update
            mem_success, mem_res = await _load_user_memory(data.get("user_id"))
            history = mem_res if mem_success else []
                
            history.append({
                "sequence": seq.tolist(),
                "actual": actual,
                "ts": time.time()
            })
            save_ok = await _save_user_memory(data.get("user_id"), history)
            if not save_ok:
                logger.error(f"Feedback memory write failed for {user_id}")
                return format_response(request, {"status": "degraded", "reason": "Database memory write failed", "data": {}})
            
            # Phase 2.4: Safe Training Trigger
            if len(learning_buffer.buffer) >= 5:
                now = time.time()
                if now - _last_training_ts > 5.0:
                    async def _run_train():
                        global _last_training_ts
                        async with _training_lock:
                            try:
                                x_b, y_b = learning_buffer.get_batch(5)
                                if x_b.numel() > 0 and not torch.isnan(x_b).any() and not torch.isnan(y_b).any():
                                    await asyncio.to_thread(train_incremental, (x_b, y_b))
                                    _last_training_ts = time.time()
                                    logger.info(f"Triggered lightweight adapt() on {x_b.size(0)} samples.")
                            except Exception as e:
                                logger.error(f"Lightweight adapt failed: {e}")
                    asyncio.create_task(_run_train())

        logger.info(f"POST /feedback success | user={user_id}")
        return format_response(request, {"status": "success", "reason": "OK", "data": {}})

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})
    except Exception as e:
        logger.exception("POST /feedback failed")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})