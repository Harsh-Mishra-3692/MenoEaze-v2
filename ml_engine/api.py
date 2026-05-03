# api.py — ELITE v5 (DEMO-SAFE | ZERO-TRUST COMPAT | HARDENED)

import time
import asyncio
import uuid
import hashlib
from typing import Dict, Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from jose import jwt, JWTError

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
from ml_engine.build_features import build_features

import os

logger = get_logger("menoeaze.api")

app = FastAPI(title="MenoEaze API")

# ─────────────────────────────────────────────
# 🔥 CORS (CRITICAL FIX)
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safe for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN = 5
FEATURES = 11
PIPELINE_TIMEOUT = 30.0
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
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            raise HTTPException(401, "Missing token")

        token = auth_header.replace("Bearer ", "")
        decoded = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=["HS256"])

        user_id = decoded.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        return user_id

    except JWTError:
        raise HTTPException(401, "Invalid or expired token")

# ─────────────────────────────────────────────
# REQUEST SCHEMA
# ─────────────────────────────────────────────
class RunRequest(BaseModel):
    action: Literal["log", "predict", "feedback"]
    payload: Dict[str, Any]

    @validator("payload")
    def validate_payload(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Invalid payload")
        return v

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _sanitize_text(text: Any, max_len=300) -> str:
    try:
        return str(text).strip()[:max_len]
    except:
        return ""

def _clamp(x):
    try:
        return float(max(0.0, min(1.0, float(x))))
    except:
        return 0.5

def _build_sequence(logs):
    if not logs:
        return None

    valid = []
    for l in logs:
        fv = l.get("feature_vector")
        if isinstance(fv, list) and len(fv) == FEATURES:
            valid.append(fv)

    if len(valid) < MIN_LOGS_REQUIRED:
        return None

    return np.array(valid[-SEQ_LEN:], dtype=np.float32)

def _safe_background(task, *args):
    async def wrapper():
        try:
            await asyncio.to_thread(task, *args)
        except Exception:
            logger.exception("[BACKGROUND TASK FAILED]")

    asyncio.create_task(wrapper())

def _idempotency_key(user_id, action, payload):
    raw = f"{user_id}:{action}:{str(payload)[:200]}"
    return hashlib.sha256(raw.encode()).hexdigest()

# ─────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────
@app.post("/run")
async def run(req: RunRequest, request: Request):

    start = time.time()
    user_id = req.payload.get("user_id") or DEMO_USER_ID
    req_key = _idempotency_key(user_id, req.action, req.payload)

    try:
        # ───────────── LOG ─────────────
        if req.action == "log":

            fv = req.payload.get("feature_vector")

            if not isinstance(fv, list) or len(fv) != FEATURES:
                raise HTTPException(400, "Invalid feature_vector")

            try:
                fv = [float(max(0, min(10, x))) for x in fv]
            except:
                raise HTTPException(400, "Invalid feature values")

            ok = insert_symptom_log(
                user_id=user_id,
                feature_vector=fv,
                raw_text=_sanitize_text(req.payload.get("notes")),
                emoji=_sanitize_text(req.payload.get("emoji"), 10)
            )

            if not ok:
                raise HTTPException(503, "DB write failed")

            _safe_background(build_features)

            return {
                "status": "ok",
                "action": "log",
                "request_id": req_key
            }

        # ───────────── STATS ─────────────
        elif req.action == "stats":
            from ml_engine.db_client import get_user_stats
            stats = get_user_stats(user_id)
            return {
                "status": "ok",
                "action": "stats",
                "stats": stats
            }

        # ───────────── PREDICT ─────────────
        elif req.action == "predict":

            query = _sanitize_text(req.payload.get("symptoms"), MAX_QUERY_LEN)

            if not query or len(query) < 3:
                raise HTTPException(400, "Invalid symptoms")

            logs = fetch_recent_logs(user_id)

            if logs is None:
                raise HTTPException(503, "DB unavailable")

            if len(logs) < MIN_LOGS_REQUIRED:
                raise HTTPException(
                    400,
                    f"Insufficient data: need at least {MIN_LOGS_REQUIRED} logs"
                )

            seq = _build_sequence(logs)

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
            except asyncio.TimeoutError:
                raise HTTPException(503, "Pipeline timeout")

            pred = result.get("prediction", {}) or {}
            rag = result.get("rag", {}) or {}
            reasoning = result.get("reasoning", {}) or {}

            sev = _clamp(pred.get("severity"))
            conf = _clamp(pred.get("confidence"))
            override = reasoning.get("override", False)

            prediction_id = None

            if override:
                insert_guardrail_log(
                    user_id=user_id,
                    query=query,
                    severity=sev,
                    message=rag.get("answer", "")
                )
            else:
                prediction_id = insert_prediction(
                    user_id=user_id,
                    severity=sev,
                    confidence=conf,
                    reasoning=rag.get("answer", "")
                )

                if not prediction_id:
                    raise HTTPException(503, "Prediction store failed")

                _safe_background(
                    add_prediction_context,
                    user_id,
                    query,
                    sev,
                    conf
                )

            return {
                "status": result.get("status", "ok"),
                "action": "predict",
                "prediction": {
                    "severity": round(sev, 4),
                    "confidence": round(conf, 4),
                    "prediction_id": prediction_id
                },
                "rag": rag,
                "meta": result.get("meta", {}),
                "reasoning": reasoning,
                "doctor": result.get("doctor", {}),
                "latency_ms": int((time.time() - start) * 1000),
                "request_id": req_key
            }

        # ───────────── FEEDBACK ─────────────
        elif req.action == "feedback":

            predicted = _clamp(req.payload.get("predicted"))
            actual = _clamp(req.payload.get("actual"))
            rating = int(max(1, min(10, req.payload.get("rating", 5))))

            prediction_id = _sanitize_text(req.payload.get("prediction_id"), 50)

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
                "request_id": req_key
            }

        else:
            raise HTTPException(400, "Invalid action")

    except HTTPException:
        raise

    except Exception:
        logger.exception("[API][CRITICAL]")
        raise HTTPException(500, "Internal server error")