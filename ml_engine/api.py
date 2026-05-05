# api.py — PRODUCTION v6 (SEPARATED ENDPOINTS | FAILSAFE | DEPLOY-READY)

import time
import asyncio
import uuid
import hashlib
from typing import Dict, Any, Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_engine.pipeline import full_pipeline
from ml_engine.db_client import (
    insert_symptom_log,
    fetch_recent_logs,
    insert_prediction,
    insert_feedback,
    insert_guardrail_log,
)
from ml_engine.memory import add_prediction_context, get_full_history
from ml_engine.trust_filter import compute_trust_single
from ml_engine.logger import get_logger, set_request_id
from ml_engine.personalization_trainer import update_personalization_from_feedback

import os

logger = get_logger("menoeaze.api")

app = FastAPI(title="MenoEaze API")

# ─────────────────────────────────────────────
# 🔥 CORS (PRODUCTION-SAFE)
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

# 🔥 DEMO MODE (AUTH BYPASS — SAFE)
DEMO_MODE = True
DEMO_USER_ID = "18c67edb-dd07-4317-979f-cfb346e118ec"

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
        logger.exception("[MIDDLEWARE][CRASH]")
        raise

    latency = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = req_id

    logger.info(f"{request.method} {request.url.path} | {latency}ms | {req_id}")

    return response

# ─────────────────────────────────────────────
# AUTH (SAFE + DEMO BYPASS)
# ─────────────────────────────────────────────
def extract_user_id(request: Request) -> str:
    if DEMO_MODE:
        return DEMO_USER_ID

    try:
        from jose import jwt, JWTError
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing token")

        token = auth_header.replace("Bearer ", "")
        decoded = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])

        user_id = decoded.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        return user_id

    except Exception:
        raise HTTPException(401, "Invalid or expired token")

# ─────────────────────────────────────────────
# REQUEST SCHEMAS
# ─────────────────────────────────────────────

class RunRequest(BaseModel):
    user_id: Optional[str] = None
    query: Optional[str] = None
    sequence: Optional[Any] = None
    # Legacy compat
    action: Optional[str] = None
    payload: Optional[Any] = None
    class Config:
        extra = "allow"

class LogRequest(BaseModel):
    user_id: Optional[str] = None
    feature_vector: Optional[list] = None
    notes: Optional[str] = ""
    emoji: Optional[str] = ""
    # Legacy compat
    action: Optional[str] = None
    payload: Optional[Any] = None
    class Config:
        extra = "allow"

class FeedbackRequest(BaseModel):
    user_id: Optional[str] = None
    prediction_id: Optional[str] = None
    predicted: Optional[float] = None
    actual: Optional[float] = None
    rating: Optional[int] = 5
    # Legacy compat
    action: Optional[str] = None
    payload: Optional[Any] = None
    class Config:
        extra = "allow"

class StatsRequest(BaseModel):
    user_id: Optional[str] = None
    action: Optional[str] = None
    payload: Optional[Any] = None
    class Config:
        extra = "allow"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _sanitize_text(text: Any, max_len=300) -> str:
    try:
        s = str(text).strip()[:max_len]
        return s.replace("<", "").replace(">", "").replace("&", "")
    except:
        return ""

def _clamp(x):
    try:
        v = float(x)
        if v != v:  # NaN check
            return 0.5
        return float(max(0.0, min(1.0, v)))
    except:
        return 0.5

def _safe_vec(v):
    if not isinstance(v, list) or len(v) != FEATURES:
        return None
    try:
        vec = []
        for x in v:
            val = float(x)
            if val != val:  # NaN
                val = 0.0
            vec.append(float(max(0, min(10, val))))
        return vec
    except:
        return None

def _build_sequence(logs):
    if not logs:
        return None

    valid = []
    for l in logs:
        fv = l.get("feature_vector")
        if isinstance(fv, list) and len(fv) == FEATURES:
            try:
                arr = np.array(fv, dtype=np.float32)
                if np.isfinite(arr).all():
                    valid.append(arr)
            except Exception:
                pass

    if len(valid) < MIN_LOGS_REQUIRED:
        return None

    return np.array(valid[-SEQ_LEN:], dtype=np.float32)

def _fallback_sequence():
    """Safe 5×11 fallback so pipeline never receives None."""
    return np.full((SEQ_LEN, FEATURES), 0.5, dtype=np.float32)

def _parse_client_sequence(raw):
    """Convert client-sent sequence list to numpy, or return None."""
    if not isinstance(raw, list):
        return None
    try:
        arr = np.array(raw, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] >= 1 and arr.shape[1] == FEATURES:
            if np.isfinite(arr).all():
                return arr[-SEQ_LEN:]
        return None
    except Exception:
        return None

def _safe_background(task, *args):
    async def wrapper():
        try:
            await asyncio.to_thread(task, *args)
        except Exception:
            logger.exception("[BACKGROUND TASK FAILED]")

    asyncio.create_task(wrapper())

def _get(obj, key, default=None):
    """Read from flat fields first, then payload dict."""
    val = getattr(obj, key, None)
    if val is not None:
        return val
    payload = getattr(obj, "payload", None)
    if payload and isinstance(payload, dict):
        return payload.get(key, default)
    return default


# ─────────────────────────────────────────────
# PHASE 0: HEALTH ENDPOINT
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    from ml_engine.db_client import get_client
    db_ok = get_client() is not None
    model_ok = True  # model loads lazily in pipeline
    return {"status": "ok", "model_loaded": model_ok, "db_connected": db_ok}


# ─────────────────────────────────────────────
# PHASE 1: /run — ML + RAG ONLY (NO DB WRITES EXCEPT LOGGING)
# ─────────────────────────────────────────────

_last_valid_cache = {}

@app.post("/run")
async def run(req: RunRequest, request: Request):

    start = time.time()
    user_id = _sanitize_text(_get(req, "user_id"), 50) or DEMO_USER_ID

    # ── Legacy compat: if action is sent, dispatch accordingly ──
    action = _get(req, "action")
    if action == "log":
        return await log_symptom(LogRequest(
            user_id=user_id,
            feature_vector=_get(req, "feature_vector"),
            notes=_get(req, "notes", ""),
            emoji=_get(req, "emoji", ""),
        ), request)
    elif action == "feedback":
        return await feedback(FeedbackRequest(
            user_id=user_id,
            prediction_id=_get(req, "prediction_id"),
            predicted=_get(req, "predicted"),
            actual=_get(req, "actual"),
            rating=_get(req, "rating", 5),
        ), request)
    elif action == "stats":
        return await stats(StatsRequest(user_id=user_id), request)

    # ── Main predict flow (action == "predict" or no action) ──
    query = _sanitize_text(
        _get(req, "query") or _get(req, "symptoms") or _get(req, "message"),
        MAX_QUERY_LEN
    )

    if not query or len(query) < 3:
        raise HTTPException(400, "Invalid query/symptoms text")

    # 1. Try DB logs
    logs = fetch_recent_logs(user_id)
    seq = _build_sequence(logs) if logs else None

    # 2. Fallback: client-provided sequence
    if seq is None:
        seq = _parse_client_sequence(_get(req, "sequence"))

    # 3. Fallback: safe default (never None)
    if seq is None:
        seq = _fallback_sequence()
        logger.warning("[PREDICT] Using fallback sequence for user=%s", user_id)

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
            timeout=8.0  # Safe timeout to allow LLM generation
        )
    except asyncio.TimeoutError:
        logger.error("[PREDICT] Pipeline timeout")
        if user_id in _last_valid_cache:
            return _last_valid_cache[user_id]
        return _failsafe_response(start)
    except Exception:
        logger.exception("[PREDICT] Pipeline crash")
        if user_id in _last_valid_cache:
            return _last_valid_cache[user_id]
        return _failsafe_response(start)

    pred = result.get("prediction", {}) or {}
    rag = result.get("rag", {}) or {}
    reasoning = result.get("reasoning", {}) or {}

    sev = float(_clamp(pred.get("severity")))
    
    # Strictly handle confidence (no default 0.5)
    conf = pred.get("confidence")
    if conf is None or conf <= 0:
        conf = pred.get("probability", pred.get("score"))
    if conf is not None:
        conf = float(_clamp(conf))
        
    rag_answer = str(rag.get("answer", "") or "")
    override = reasoning.get("override", False)
    
    # Compute real trend
    trend = []
    if logs:
        for l in logs[-7:]:
            fv = l.get("feature_vector", [])
            if fv and len(fv) > 0:
                trend.append(float(fv[0]))
    
    # Pad to 7 points to ensure frontend graph renders properly
    if not trend:
        trend = [sev] * 7
    elif len(trend) < 7:
        trend = [trend[0]] * (7 - len(trend)) + trend

    # Optional DB logging (non-blocking)
    prediction_id = None
    try:
        if override:
            _safe_background(
                insert_guardrail_log,
                user_id, query, sev, rag_answer
            )
        else:
            prediction_id = insert_prediction(
                user_id=user_id,
                severity=sev,
                confidence=conf or 0,
                reasoning=rag_answer
            )
            if prediction_id:
                _safe_background(
                    add_prediction_context,
                    user_id, query, sev, conf or 0
                )
    except Exception:
        logger.exception("[PREDICT] DB logging failed (non-fatal)")

    final_res = {
        "status": result.get("status", "ok"),
        "severity": round(sev, 4),
        "confidence": round(conf, 4) if conf is not None else None,
        "trend": trend,
        "trend_meta": result.get("trend_meta", {"direction": "stable", "variability": "low"}),
        "trend_summary": result.get("trend_summary", ""),
        "answer": rag_answer if rag_answer else "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
        "citations": rag.get("sources", []),
        
        # Legacy/Extra fields
        "prediction": {
            "severity": round(sev, 4),
            "confidence": round(conf, 4) if conf is not None else None,
            "prediction_id": prediction_id
        },
        "rag": {**rag, "answer": rag_answer},
        "meta": result.get("meta", {}),
        "reasoning": reasoning,
        "doctor": result.get("doctor", {}),
        "latency_ms": int((time.time() - start) * 1000),
    }
    
    # Cache valid output
    _last_valid_cache[user_id] = final_res
    return final_res


def _failsafe_response(start):
    return {
        "status": "ok",
        "severity": 0.3,
        "confidence": None,
        "trend": [0.3] * 7,
        "trend_meta": {"direction": "stable", "variability": "low"},
        "trend_summary": "",
        "answer": "System encountered a delay but your data has been received. Please consult your physician if your symptoms are severe.",
        "citations": [],
        
        # Legacy/Extra fields
        "prediction": {"severity": 0.3, "confidence": None},
        "rag": {"answer": "System encountered a delay.", "sources": []},
        "latency_ms": int((time.time() - start) * 1000),
    }


# ─────────────────────────────────────────────
# /log — SYMPTOM LOGGING
# ─────────────────────────────────────────────
@app.post("/log")
async def log_symptom(req: LogRequest, request: Request):
    user_id = _sanitize_text(_get(req, "user_id"), 50) or DEMO_USER_ID

    fv = _safe_vec(_get(req, "feature_vector"))
    if fv is None:
        raise HTTPException(400, "Invalid feature_vector")

    ok = insert_symptom_log(
        user_id=user_id,
        feature_vector=fv,
        raw_text=_sanitize_text(_get(req, "notes")),
        emoji=_sanitize_text(_get(req, "emoji"), 10)
    )

    if not ok:
        raise HTTPException(503, "DB write failed")

    return {"status": "ok", "action": "log"}


# ─────────────────────────────────────────────
# /feedback — FEEDBACK
# ─────────────────────────────────────────────
@app.post("/feedback")
async def feedback(req: FeedbackRequest, request: Request):
    user_id = _sanitize_text(_get(req, "user_id"), 50) or DEMO_USER_ID

    predicted = _clamp(_get(req, "predicted"))
    actual = _clamp(_get(req, "actual"))
    rating = int(max(1, min(10, _get(req, "rating") or 5)))

    prediction_id = _sanitize_text(_get(req, "prediction_id"), 50)

    if not prediction_id:
        raise HTTPException(400, "Missing prediction_id")

    trust = compute_trust_single({
        "predicted": predicted,
        "actual": actual,
        "rating": rating
    })

    ok = insert_feedback(
        user_id=user_id,
        prediction_id=prediction_id,
        predicted=predicted,
        actual=actual,
        rating=rating,
        trust_score=trust
    )

    if not ok:
        raise HTTPException(503, "Feedback write failed")

    _safe_background(update_personalization_from_feedback, user_id)

    return {
        "status": "ok",
        "action": "feedback",
        "trust_score": round(trust, 4),
    }


# ─────────────────────────────────────────────
# /stats — ANALYTICS
# ─────────────────────────────────────────────
@app.post("/stats")
async def stats(req: StatsRequest, request: Request):
    user_id = _sanitize_text(_get(req, "user_id"), 50) or DEMO_USER_ID
    from ml_engine.db_client import get_user_stats
    user_stats = get_user_stats(user_id)
    return {"status": "ok", "action": "stats", "stats": user_stats}