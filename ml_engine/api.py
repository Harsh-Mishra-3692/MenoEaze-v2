# api.py — PRODUCTION v8 (DEBUG ENABLED | RAILWAY SAFE)

import time
import asyncio
import uuid
from typing import Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# ─────────────────────────────────────────────
# SAFE IMPORTS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN = 5
FEATURES = 11
PIPELINE_TIMEOUT = 25.0   # 🔥 Increased for Railway
MAX_QUERY_LEN = 500

SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

DEMO_MODE = True
DEMO_USER_ID = "18c67edb-dd07-4317-979f-cfb346e118ec"

MODEL_READY = False
DB_READY = False

# ─────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────
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
    except Exception as e:
        logger.exception("❌ Model warmup failed")
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
    except Exception as e:
        logger.exception("🔥 REQUEST CRASH")
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
# AUTH
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
# MAIN (/run)
# ─────────────────────────────────────────────
@app.post("/run")
async def run(req: RunRequest, request: Request):

    start = time.time()
    user_id = _sanitize_text(req.user_id, 50) or DEMO_USER_ID
    query = _sanitize_text(req.query, MAX_QUERY_LEN)

    if not query or len(query) < 3:
        raise HTTPException(400, "Invalid query")

    logger.info(f"🧠 Running pipeline | user={user_id} | query={query}")

    # Sequence
    try:
        logs = fetch_recent_logs(user_id)
        if logs:
            seq = np.array([l["feature_vector"] for l in logs[-5:]], dtype=np.float32)
        else:
            seq = _fallback_sequence()
    except Exception as e:
        logger.exception("⚠️ Sequence build failed")
        seq = _fallback_sequence()

    # History
    try:
        user_history = await asyncio.to_thread(get_full_history, user_id)
    except:
        user_history = []

    # 🔥 PIPELINE EXECUTION
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

    except asyncio.TimeoutError:
        logger.error("⏱️ PIPELINE TIMEOUT")
        return {
            "status": "error",
            "error": "Pipeline timeout",
            "stage": "timeout"
        }

    except Exception as e:
        logger.exception("🚨 PIPELINE FAILURE")
        return {
            "status": "error",
            "error": str(e),
            "stage": "pipeline"
        }

    # ───────── RESULT PARSE ─────────
    pred = result.get("prediction", {}) or {}
    rag = result.get("rag", {}) or {}

    sev = _clamp(pred.get("severity"))
    conf = _clamp(pred.get("confidence", 0.5))

    final = {
        "status": "ok",
        "severity": sev,
        "confidence": conf,
        "trend": [sev] * 7,
        "trend_summary": result.get("trend_summary", ""),
        "answer": rag.get("answer", ""),
        "citations": rag.get("sources", []),
        "latency_ms": int((time.time() - start) * 1000)
    }

    logger.info("✅ Pipeline success")
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
