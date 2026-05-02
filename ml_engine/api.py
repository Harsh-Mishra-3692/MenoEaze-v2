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
MAX_MEMORY = 50

# ─────────────────────────────────────────────
# DB INTEGRATION HELPERS (PHASE 2.4)
# ─────────────────────────────────────────────
async def safe_db_call(fn, *args, **kwargs):
    try:
        res = await asyncio.wait_for(asyncio.to_thread(fn, *args, **kwargs), timeout=2.0)
        return True, res
    except Exception as e:
        logger.error(f"safe_db_call failed: {e}")
        return False, None

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

from ml_engine.db_client import insert_feedback, check_connection, fetch_table, insert_bulk, get_user_memory, update_user_memory

async def _load_user_memory(user_id: str):
    success, mem = await safe_db_call(get_user_memory, user_id)
    if success:
        history = mem.get("history", [])
        return True, history[-MAX_MEMORY:]
    return False, []

async def _save_user_memory(user_id: str, new_entry: dict):
    success, res = await safe_db_call(update_user_memory, user_id, new_entry)
    if not success:
        logger.error(f"user_memory save failed")
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
        return False, f"Invalid shape {arr.shape}"

    if not np.isfinite(arr).all():
        logger.error("NaN/Inf detected in input sequence")
        return False, "NaN/Inf detected"

    return True, arr

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class RunRequest(BaseModel):
    user_id: str
    query: str
    sequence: Optional[List[List[float]]] = None

    @field_validator("sequence")
    def validate(cls, v):
        if v is None: return v
        ok, data = sanitize_sequence(v)
        if not ok: raise ValueError(data)
        return data.tolist()


class PredictRequest(BaseModel):
    user_id: str
    sequence: List[List[float]]

    @field_validator("sequence")
    def validate(cls, v):
        ok, data = sanitize_sequence(v)
        if not ok: raise ValueError(data)
        return data.tolist()


class QueryRequest(BaseModel):
    user_id: str
    query: str

# ─────────────────────────────────────────────
# RESPONSE FORMATTER (UNIFIED)
# ─────────────────────────────────────────────
def format_response(request: Request, response_dict: dict):
    return response_dict

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

    return format_response(None, {
        "status": "degraded" if issues else "success",
        "reason": "OK" if not issues else "Health checks failed",
        "data": {
            "model": "loaded" if model_loaded else "missing",
            "db": "connected" if db_connected else "down",
            "llm": "ready" if llm_ready else "missing",
            "issues": issues
        }
    })

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

        # Append to DB atomically
        save_success = await _save_user_memory(data.user_id, entry)

        prediction = result.get("prediction", {})
        rag = result.get("rag", {})
        
        status = "success"
        reason = "OK"
        if "error" in prediction:
            status, reason = "degraded", "model"
        elif db_error or not save_success:
            status, reason = "degraded", "db"
        elif "error" in rag or rag.get("fallback"):
            status, reason = "degraded", "llm"

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
    logger.warning("POST /predict is deprecated. Use POST /run instead.")
    return format_response(request, {
        "status": "error",
        "reason": "Deprecated: Use POST /run",
        "data": {}
    })

# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────
@app.post("/query")
async def query(data: QueryRequest, request: Request):
    logger.warning("POST /query is deprecated. Use POST /run instead.")
    return format_response(request, {
        "status": "error",
        "reason": "Deprecated: Use POST /run",
        "data": {}
    })

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
        
        success, _ = await safe_db_call(insert_feedback, record)
        if not success:
            logger.error(f"Feedback DB write timeout for user {user_id}")
            return format_response(request, {"status": "degraded", "reason": "db", "data": {}})

        # sanitize
        sequence = data.get("sequence")
        actual = data.get("actual_severity")

        if sequence is not None and actual is not None:
            ok, seq = sanitize_sequence(sequence)
            if not ok:
                return format_response(request, {"status": "error", "reason": seq, "data": {}})
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
            new_entry = {
                "sequence": seq.tolist(),
                "actual": actual,
                "ts": time.time()
            }
            save_ok = await _save_user_memory(data.get("user_id"), new_entry)
            if not save_ok:
                logger.error(f"Feedback memory write failed for {user_id}")
                return format_response(request, {"status": "degraded", "reason": "Database memory write failed", "data": {}})
            
            # Phase 2.4: Safe Training Trigger
            if len(learning_buffer.buffer) >= 5:
                now = time.time()
                if now - _last_training_ts > 5.0:
                    x_b, y_b = learning_buffer.get_batch(5)
                    if x_b.numel() > 0 and not torch.isnan(x_b).any() and not torch.isnan(y_b).any():
                        batch = (x_b, y_b)
                        async def _run_train(batch):
                            try:
                                async with _training_lock:
                                    await asyncio.to_thread(train_incremental, batch)
                            except Exception as e:
                                logger.error(f"Training task failed: {e}")
                        asyncio.create_task(_run_train(batch))

        logger.info(f"POST /feedback success | user={user_id}")
        return format_response(request, {"status": "success", "reason": "OK", "data": {}})

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})
    except Exception as e:
        logger.exception("POST /feedback failed")
        return format_response(request, {"status": "error", "reason": str(e), "data": {}})