# api.py — ELITE (PRODUCTION-GRADE ORCHESTRATOR + CONTINUAL LEARNING)

import time
import asyncio
import logging
from typing import Dict, List, Optional

import torch
import numpy as np

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager

from ml_engine.pipeline import full_pipeline
from ml_engine.data_collector import build_feedback_record
from ml_engine.db_client import insert_feedback

# NEW: Phase 3 — Continual Learning imports
import ml_engine.pipeline as pipeline
from ml_engine.adaptation_worker import learning_buffer, start_worker_thread
from ml_engine.model_loader import load_model as ml_load_model

logger = logging.getLogger("menoeaze.api")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN = 5
FEATURES = 11
MAX_MEMORY = 20

# ─────────────────────────────────────────────
# MEMORY
# ─────────────────────────────────────────────
user_memory: Dict[str, List[dict]] = {}

# ─────────────────────────────────────────────
# MODEL HOT-SWAP (NON-BLOCKING)
# ─────────────────────────────────────────────
_current_model_timestamp: int = 0

async def _periodic_model_refresh():
    """Non-blocking background loop: hot-swap the pipeline model if a new version exists."""
    global _current_model_timestamp
    while True:
        try:
            model, metadata = ml_load_model()
            new_ts = metadata.get("timestamp") or 0
            if new_ts > _current_model_timestamp:
                logger.info(f"[API] New model version detected (ts={new_ts}). Hot-swapping.")
                # GIL-safe pointer swap — no lock needed for reference assignment
                pipeline._model = model
                pipeline._model.to(pipeline.DEVICE)
                pipeline._model.eval()
                _current_model_timestamp = new_ts
        except Exception as e:
            logger.error(f"[API] Model refresh failed (non-fatal): {e}")
        await asyncio.sleep(60)

# ─────────────────────────────────────────────
# LIFECYCLE
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[API] Started")
    # NEW: Start the immortal background adaptation worker
    start_worker_thread(check_interval_seconds=60, min_batch_size=50)
    logger.info("[API] Adaptation worker thread started.")
    # NEW: Start the non-blocking model refresh heartbeat
    asyncio.create_task(_periodic_model_refresh())
    logger.info("[API] Model refresh heartbeat started.")
    yield

app = FastAPI(title="MenoEaze Elite AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class RunRequest(BaseModel):
    user_id: str
    query: str
    sequence: Optional[List[List[float]]] = None

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v):
        if v is None:
            return v
        arr = np.array(v, dtype=np.float32)
        if arr.shape != (SEQ_LEN, FEATURES):
            raise ValueError(f"Expected shape {(SEQ_LEN, FEATURES)}")
        return v


class PredictRequest(BaseModel):
    user_id: str
    sequence: List[List[float]]

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v):
        arr = np.array(v, dtype=np.float32)
        if arr.shape != (SEQ_LEN, FEATURES):
            raise ValueError(f"Expected shape {(SEQ_LEN, FEATURES)}")
        return v


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
        user_id = data.user_id
        query = data.query

        seq_array = None
        if data.sequence is not None:
            seq_array = np.array(data.sequence, dtype=np.float32)

        history = user_memory.get(user_id, [])

        result = full_pipeline(
            user_id=user_id,
            query=query,
            sequence=seq_array,
            user_history={
                "predictions": [h["pred"] for h in history if "pred" in h],
                "actuals": [h["actual"] for h in history if "actual" in h],
                # Phase 5: MAML needs raw feedback logs with sequence + actual_severity
                "feedback_logs": [
                    h for h in history
                    if "sequence" in h and "actual_severity" in h
                ],
            }
        )

        # ── Update memory ──────────────────
        entry = {
            "query": query,
            "ts": time.time(),
        }

        if result.get("prediction"):
            entry["pred"] = result["prediction"]["severity"]

        user_memory.setdefault(user_id, []).append(entry)
        user_memory[user_id] = user_memory[user_id][-MAX_MEMORY:]

        latency = (time.time() - start) * 1000
        result["api_latency_ms"] = round(latency, 2)

        return result

    except Exception as e:
        logger.exception("[API] run failed")
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# PREDICT ONLY
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(data: PredictRequest):

    try:
        # Phase 5: Gather user feedback history for MAML fast adaptation
        history = user_memory.get(data.user_id, [])

        result = full_pipeline(
            user_id=data.user_id,
            query="",
            sequence=np.array(data.sequence, dtype=np.float32),
            user_history={
                "predictions": [h["pred"] for h in history if "pred" in h],
                "actuals": [h["actual"] for h in history if "actual" in h],
                "feedback_logs": [
                    h for h in history
                    if "sequence" in h and "actual_severity" in h
                ],
            }
        )

        return result.get("prediction", {})

    except Exception as e:
        logger.exception("[API] predict failed")
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# QUERY ONLY
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
        logger.exception("[API] query failed")
        raise HTTPException(500, str(e))


# ─────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────
@app.post("/feedback")
async def feedback(data: dict, background_tasks: BackgroundTasks):

    try:
        # build_feedback_record expects exactly (user_id, predicted, actual, error).
        # The request may contain extra keys (sequence, actual_severity, etc.)
        # so we extract only what the legacy function needs.
        try:
            record = build_feedback_record(
                user_id=data.get("user_id", ""),
                predicted=data.get("predicted", 0.0),
                actual=data.get("actual", data.get("actual_severity", 0.0)),
                error=data.get("error", 0.0),
            )
            background_tasks.add_task(insert_feedback, record)
        except Exception as rec_err:
            logger.warning(f"[API] build_feedback_record failed (non-fatal): {rec_err}")

        # NEW: Pipe feedback data into the continual learning buffer
        sequence = data.get("sequence")
        actual_severity = data.get("actual_severity")

        if sequence is not None and actual_severity is not None:
            try:
                x_tensor = torch.tensor(sequence, dtype=torch.float32)
                y_tensor = torch.tensor([float(actual_severity)], dtype=torch.float32)
                learning_buffer.add_sample(x_tensor, y_tensor)
                logger.info("[API] Feedback piped to online learning buffer.")
            except Exception as buf_err:
                logger.warning(f"[API] Buffer injection failed (non-fatal): {buf_err}")

        # Phase 5: Store feedback in user_memory for MAML fast adaptation
        user_id = data.get("user_id")
        if user_id and sequence is not None and actual_severity is not None:
            fb_entry = {
                "sequence": sequence,
                "actual_severity": float(actual_severity),
                "actual": float(actual_severity),
                "ts": time.time(),
            }
            user_memory.setdefault(user_id, []).append(fb_entry)
            user_memory[user_id] = user_memory[user_id][-MAX_MEMORY:]
            logger.info(f"[API] Feedback stored in user_memory for MAML | user={user_id}")

        return {"status": "ok"}

    except Exception as e:
        logger.exception("[API] feedback failed")
        raise HTTPException(500, str(e))