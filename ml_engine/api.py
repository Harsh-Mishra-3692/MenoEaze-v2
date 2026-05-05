# api.py — PRODUCTION v7 (RAILWAY SAFE | FULL FEATURE PRESERVED)

import time
import asyncio
import uuid
from typing import Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# 🔥 SAFE IMPORT (works locally + Railway)
try:
    from ml_engine.pipeline import full_pipeline
    from ml_engine.db_client import (
        insert_symptom_log,
        fetch_recent_logs,
        insert_prediction,
        insert_feedback,
        insert_guardrail_log,
        get_client,
        get_user_stats
    )
    from ml_engine.memory import add_prediction_context, get_full_history
    from ml_engine.trust_filter import compute_trust_single
    from ml_engine.logger import get_logger, set_request_id
    from ml_engine.personalization_trainer import update_personalization_from_feedback
except:
    from pipeline import full_pipeline
    from db_client import (
        insert_symptom_log,
        fetch_recent_logs,
        insert_prediction,
        insert_feedback,
        insert_guardrail_log,
        get_client,
        get_user_stats
    )
    from memory import add_prediction_context, get_full_history
    from trust_filter import compute_trust_single
    from logger import get_logger, set_request_id
    from personalization_trainer import update_personalization_from_feedback

logger = get_logger("menoeaze.api")
app = FastAPI(title="MenoEaze API")

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN = 5
FEATURES = 11
PIPELINE_TIMEOUT = 8.0
MAX_QUERY_LEN = 500
MIN_LOGS_REQUIRED = 5

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# DEMO MODE
DEMO_MODE = True
DEMO_USER_ID = "18c67edb-dd07-4317-979f-cfb346e118ec"

# ─────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────
MODEL_READY = False
DB_READY = False

# ─────────────────────────────────────────────
# STARTUP (PRELOAD MODEL)
# ─────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global MODEL_READY, DB_READY

    logger.info("Starting API...")

    try:
        await asyncio.to_thread(
            full_pipeline,
            user_id="warmup",
            query="warmup",
            sequence=np.full((5, 11), 0.5, dtype=np.float32),
            user_history=[]
        )
        MODEL_READY = True
        logger.info("Model warmup complete")
    except Exception:
        logger.exception("Model warmup failed")
        MODEL_READY = False

    try:
        DB_READY = get_client() is not None
    except:
        DB_READY = False

# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    req_id = str(uuid.uuid4())
    set_request_id(req_id)

    start = time.time()

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("[CRASH]")
        raise

    latency = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = req_id

    logger.info(f"{request.method} {request.url.path} | {latency}ms")

    return response

# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_READY,
        "db_connected": DB_READY
    }

# ─────────────────────────────────────────────
# AUTH (unchanged)
# ─────────────────────────────────────────────
def extract_user_id(request: Request) -> str:
    if DEMO_MODE:
        return DEMO_USER_ID

    from jose import jwt

    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "")
    decoded = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])
    return decoded.get("sub")

# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class RunRequest(BaseModel):
    user_id: Optional[str] = None
    query: Optional[str] = None
    sequence: Optional[Any] = None
    action: Optional[str] = None
    payload: Optional[Any] = None

class LogRequest(BaseModel):
    user_id: Optional[str] = None
    feature_vector: Optional[list] = None
    notes: Optional[str] = ""
    emoji: Optional[str] = ""

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = None
    prediction_id: Optional[str] = None
    predicted: Optional[float] = None
    actual: Optional[float] = None
    rating: Optional[int] = 5

class StatsRequest(BaseModel):
    user_id: Optional[str] = None

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# MAIN ENDPOINT (/run)
# ─────────────────────────────────────────────
_last_valid_cache = {}

@app.post("/run")
async def run(req: RunRequest, request: Request):

    start = time.time()
    user_id = _sanitize_text(req.user_id, 50) or DEMO_USER_ID

    # 🔁 ACTION ROUTING (PRESERVED)
    if req.action == "log":
        return await log_symptom(LogRequest(**(req.payload or {})), request)

    if req.action == "feedback":
        return await feedback(FeedbackRequest(**(req.payload or {})), request)

    if req.action == "stats":
        return await stats(StatsRequest(**(req.payload or {})), request)

    query = _sanitize_text(req.query, MAX_QUERY_LEN)

    if not query or len(query) < 3:
        raise HTTPException(400, "Invalid query")

    # Build sequence
    logs = fetch_recent_logs(user_id)
    seq = None

    if logs:
        try:
            seq = np.array([l["feature_vector"] for l in logs[-5:]], dtype=np.float32)
        except:
            seq = None

    if seq is None or seq.shape != (5, 11):
        seq = _fallback_sequence()

    try:
        user_history = await asyncio.to_thread(get_full_history, user_id)
    except:
        user_history = []

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
    except Exception:
        logger.exception("Pipeline failure")
        if user_id in _last_valid_cache:
            return _last_valid_cache[user_id]
        raise HTTPException(500, "Pipeline failed")

    pred = result.get("prediction", {}) or {}
    rag = result.get("rag", {}) or {}

    sev = _clamp(pred.get("severity"))
    conf = _clamp(pred.get("confidence", 0.5))

    trend = [sev] * 7

    final = {
        "status": "ok",
        "severity": sev,
        "confidence": conf,
        "trend": trend,
        "trend_summary": result.get("trend_summary", ""),
        "answer": rag.get("answer", ""),
        "citations": rag.get("sources", []),

        "prediction": {
            "severity": sev,
            "confidence": conf
        },
        "rag": rag,
        "latency_ms": int((time.time() - start) * 1000)
    }

    _last_valid_cache[user_id] = final
    return final

# ─────────────────────────────────────────────
# /log
# ─────────────────────────────────────────────
@app.post("/log")
async def log_symptom(req: LogRequest, request: Request):
    user_id = req.user_id or DEMO_USER_ID

    if not req.feature_vector or len(req.feature_vector) != FEATURES:
        raise HTTPException(400, "Invalid feature_vector")

    ok = insert_symptom_log(
        user_id=user_id,
        feature_vector=req.feature_vector,
        raw_text=req.notes,
        emoji=req.emoji
    )

    if not ok:
        raise HTTPException(503, "DB write failed")

    return {"status": "ok"}

# ─────────────────────────────────────────────
# /feedback
# ─────────────────────────────────────────────
@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request):
    user_id = req.user_id or DEMO_USER_ID

    trust = compute_trust_single({
        "predicted": _clamp(req.predicted),
        "actual": _clamp(req.actual),
        "rating": req.rating or 5
    })

    ok = insert_feedback(
        user_id=user_id,
        prediction_id=req.prediction_id,
        predicted=req.predicted,
        actual=req.actual,
        rating=req.rating,
        trust_score=trust
    )

    if not ok:
        raise HTTPException(503, "Feedback write failed")

    asyncio.create_task(
        asyncio.to_thread(update_personalization_from_feedback, user_id)
    )

    return {"status": "ok", "trust_score": trust}

# ─────────────────────────────────────────────
# /stats
# ─────────────────────────────────────────────
@app.post("/stats")
async def stats(req: StatsRequest, request: Request):
    user_id = req.user_id or DEMO_USER_ID
    stats = get_user_stats(user_id)
    return {"status": "ok", "stats": stats}