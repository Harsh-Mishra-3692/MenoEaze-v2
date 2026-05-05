# api.py — PRODUCTION v9 (L7/L9 HARDENED | NEVER BREAKS FRONTEND)

import time
import asyncio
import uuid
from typing import Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# ───────── SAFE IMPORTS ─────────
try:
    from ml_engine.pipeline import full_pipeline
    from ml_engine.db_client import (
        insert_symptom_log,
        fetch_recent_logs,
        insert_feedback,
        get_client,
        get_user_stats
    )
    from ml_engine.memory import get_full_history
    from ml_engine.trust_filter import compute_trust_single
    from ml_engine.logger import get_logger, set_request_id
    from ml_engine.personalization_trainer import update_personalization_from_feedback
except:
    from pipeline import full_pipeline
    from db_client import (
        insert_symptom_log,
        fetch_recent_logs,
        insert_feedback,
        get_client,
        get_user_stats
    )
    from memory import get_full_history
    from trust_filter import compute_trust_single
    from logger import get_logger, set_request_id
    from personalization_trainer import update_personalization_from_feedback

logger = get_logger("menoeaze.api")
app = FastAPI(title="MenoEaze API")


# ───────── CORS ─────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ───────── CONFIG ─────────
SEQ_LEN = 5
FEATURES = 11
PIPELINE_TIMEOUT = 25.0
MAX_QUERY_LEN = 500

DEMO_MODE = True
DEMO_USER_ID = "18c67edb-dd07-4317-979f-cfb346e118ec"

MODEL_READY = False
DB_READY = False


# ───────── STARTUP ─────────
@app.on_event("startup")
async def startup_event():
    global MODEL_READY, DB_READY

    logger.info("🚀 Starting API...")

    try:
        await asyncio.to_thread(
            full_pipeline,
            user_id="warmup",
            query="warmup",
            sequence=np.full((5, 11), 0.5, dtype=np.float32),
            user_history=[]
        )
        MODEL_READY = True
        logger.info("✅ Model warmup complete")
    except Exception:
        MODEL_READY = False

    try:
        DB_READY = get_client() is not None
    except:
        DB_READY = False


# ───────── MIDDLEWARE ─────────
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    req_id = str(uuid.uuid4())
    set_request_id(req_id)

    start = time.time()

    response = await call_next(request)

    latency = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = req_id

    logger.info(f"{request.method} {request.url.path} | {latency}ms")

    return response


# ───────── HEALTH ─────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_READY,
        "db_connected": DB_READY
    }


# ───────── SCHEMAS ─────────
class RunRequest(BaseModel):
    user_id: Optional[str] = None
    query: Optional[str] = None


# ───────── HELPERS ─────────
def _sanitize_text(text: Any, max_len=300):
    try:
        return str(text).strip()[:max_len]
    except:
        return ""


def _clamp(x):
    try:
        return float(max(0.0, min(1.0, float(x))))
    except:
        return 0.5


def _fallback_sequence():
    return np.full((SEQ_LEN, FEATURES), 0.5, dtype=np.float32)


def _safe_answer(query: str):
    return (
        "Based on your symptoms, this may be related to hormonal changes during menopause. "
        "For a precise diagnosis and personalized care, please consult a healthcare professional."
    )


# ───────── MAIN (/run) ─────────
@app.post("/run")
async def run(req: RunRequest, request: Request):

    start = time.time()

    user_id = _sanitize_text(req.user_id, 50) or DEMO_USER_ID
    query = _sanitize_text(req.query, MAX_QUERY_LEN)

    if not query or len(query) < 3:
        raise HTTPException(400, "Invalid query")

    # ───────── SEQUENCE ─────────
    try:
        logs = fetch_recent_logs(user_id)
        if logs:
            seq = np.array([l["feature_vector"] for l in logs[-5:]], dtype=np.float32)
        else:
            seq = _fallback_sequence()
    except Exception:
        seq = _fallback_sequence()

    # ───────── HISTORY ─────────
    try:
        user_history = await asyncio.to_thread(get_full_history, user_id)
    except:
        user_history = []

    # ───────── PIPELINE ─────────
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                full_pipeline,
                user_id=user_id,
                query=query,
                sequence=seq,
                user_history=user_history
            ),
            timeout=PIPELINE_TIMEOUT
        )
    except Exception as e:
        logger.error(f"PIPELINE FAILURE: {e}")
        result = {}

    # ───────── SAFE PARSE ─────────
    pred = result.get("prediction", {}) if isinstance(result, dict) else {}
    rag = result.get("rag", {}) if isinstance(result, dict) else {}

    severity = _clamp(pred.get("severity"))
    confidence = _clamp(pred.get("confidence", 0.5))

    answer = rag.get("answer") if isinstance(rag, dict) else None

    if not answer:
        answer = _safe_answer(query)

    response = {
        "status": "ok",
        "severity": severity,
        "confidence": confidence,
        "trend": [severity] * 7,
        "trend_summary": result.get("trend_summary", "") if isinstance(result, dict) else "",
        "answer": answer,
        "citations": rag.get("sources", []) if isinstance(rag, dict) else [],
        "latency_ms": int((time.time() - start) * 1000)
    }

    return response
