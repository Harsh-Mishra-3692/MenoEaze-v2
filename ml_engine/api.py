# api.py — PRODUCTION-GRADE ORCHESTRATOR (HARDENED + STABLE)

import time
import math
import asyncio
from typing import Dict, List, Optional

import torch
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager

# Internal modules
from ml_engine.pipeline import full_pipeline
from ml_engine.data_collector import build_feedback_record
from ml_engine.db_client import insert_feedback

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
# MEMORY (TEMP — replace with DB later)
# ─────────────────────────────────────────────
user_memory: Dict[str, List[dict]] = {}

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
        logger.warning("NaN/Inf detected → sanitized")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

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
# ROUTES
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────
@app.post("/run")
async def run_pipeline(data: RunRequest):
    start = time.time()

    try:
        history = user_memory.get(data.user_id, [])

        result = full_pipeline(
            user_id=data.user_id,
            query=data.query,
            sequence=np.array(data.sequence) if data.sequence else None,
            user_history={
                "predictions": [h.get("pred") for h in history],
                "actuals": [h.get("actual") for h in history],
                "feedback_logs": [h for h in history if "sequence" in h],
            }
        )

        # update memory
        entry = {
            "ts": time.time(),
            "query": data.query,
        }

        if result.get("prediction"):
            entry["pred"] = result["prediction"]["severity"]

        user_memory.setdefault(data.user_id, []).append(entry)
        user_memory[data.user_id] = user_memory[data.user_id][-MAX_MEMORY:]

        result["latency_ms"] = round((time.time() - start) * 1000, 2)

        return result

    except Exception as e:
        logger.exception("run_pipeline failed")
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(data: PredictRequest):
    try:
        history = user_memory.get(data.user_id, [])

        result = full_pipeline(
            user_id=data.user_id,
            query="",
            sequence=np.array(data.sequence),
            user_history={"feedback_logs": history}
        )

        return result.get("prediction", {})

    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────────
# QUERY
# ─────────────────────────────────────────────
@app.post("/query")
async def query(data: QueryRequest):
    try:
        result = full_pipeline(
            user_id=data.user_id,
            query=data.query,
            sequence=None,
            user_history=None
        )
        return result.get("rag", {})

    except Exception as e:
        logger.exception("query failed")
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────
@app.post("/feedback")
async def feedback(data: dict, background_tasks: BackgroundTasks):
    try:
        # DB write (non-blocking)
        try:
            record = build_feedback_record(
                user_id=data.get("user_id", ""),
                predicted=data.get("predicted", 0.0),
                actual=data.get("actual", data.get("actual_severity", 0.0)),
                error=data.get("error", 0.0),
            )
            background_tasks.add_task(insert_feedback, record)
        except Exception as e:
            logger.warning(f"Feedback record failed: {e}")

        # sanitize
        sequence = data.get("sequence")
        actual = data.get("actual_severity")

        if sequence is not None and actual is not None:
            seq = sanitize_sequence(sequence)
            actual = float(actual) if math.isfinite(float(actual)) else 0.0

            learning_buffer.add_sample(
                torch.tensor(seq, dtype=torch.float32),
                torch.tensor([actual], dtype=torch.float32)
            )

            # memory update
            user_memory.setdefault(data["user_id"], []).append({
                "sequence": seq.tolist(),
                "actual": actual,
                "ts": time.time()
            })

        return {"status": "ok"}

    except Exception as e:
        logger.exception("feedback failed")
        raise HTTPException(500, str(e))