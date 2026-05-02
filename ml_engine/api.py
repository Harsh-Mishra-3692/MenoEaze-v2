# api.py — FINAL ELITE v2 (FULL PRODUCTION ORCHESTRATOR)

import time
import asyncio
import uuid
import hashlib
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, validator, Field

from ml_engine.pipeline import full_pipeline
from ml_engine.db_client import (
    insert_symptom_log,
    fetch_recent_logs,
    insert_prediction,
    insert_feedback,
    fetch_prediction_history,
    fetch_feedback_history,
    insert_guardrail_log
)
from ml_engine.memory import add_prediction_context, get_full_history
from ml_engine.trust_filter import compute_trust_single
from ml_engine.logger import get_logger, set_request_id
from ml_engine.personalization_trainer import update_personalization_from_feedback
from ml_engine.build_features import build_features

logger = get_logger("menoeaze.api")

SEQ_LEN = 5
FEATURES = 11
PIPELINE_TIMEOUT = 6.0
DB_TIMEOUT = 2.0
MAX_QUERY_LEN = 500
MIN_USER_ID_LEN = 10

app = FastAPI(title="MenoEaze API")


# ─────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────
@app.middleware("http")
async def add_request_context(request: Request, call_next):
    req_id = str(uuid.uuid4())
    set_request_id(req_id)

    start = time.time()
    response = await call_next(request)
    latency = round((time.time() - start) * 1000, 2)

    response.headers["X-Request-ID"] = req_id

    logger.info(f"{request.method} {request.url.path} | {latency}ms | {req_id}")

    return response


# ─────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────
class LogSymptomRequest(BaseModel):
    user_id: str
    feature_vector: List[float]
    notes: Optional[str] = ""
    emoji: Optional[str] = ""

    @validator("feature_vector")
    def validate_vector(cls, v):
        if len(v) != FEATURES:
            raise ValueError("feature_vector must be length 11")
        if any([not np.isfinite(x) for x in v]):
            raise ValueError("Invalid numeric values")
        return [float(max(-5, min(10, x))) for x in v]


class PredictRequest(BaseModel):
    user_id: str
    symptoms: str

    @validator("user_id")
    def validate_user(cls, v):
        if not v or len(v) < MIN_USER_ID_LEN:
            raise ValueError("Invalid user_id")
        return v

    @validator("symptoms")
    def validate_symptoms(cls, v):
        v = v.strip()
        if not v or len(v) < 3:
            raise ValueError("Invalid symptoms text")
        return v[:MAX_QUERY_LEN]


class FeedbackRequest(BaseModel):
    user_id: str
    prediction_id: str
    predicted: float
    actual: float
    rating: int


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _sanitize_text(text: str, max_len=300):
    try:
        return str(text).strip()[:max_len]
    except:
        return ""


def build_sequence(logs):
    if not logs:
        return None
    seq = [
        l.get("feature_vector")
        for l in logs
        if isinstance(l.get("feature_vector"), list)
        and len(l.get("feature_vector")) == FEATURES
    ]
    if len(seq) < 3:
        return None
    return np.array(seq[-SEQ_LEN:], dtype=np.float32)


def safe_prediction(pred: dict):
    try:
        return (
            max(0.0, min(1.0, float(pred.get("severity", 0.5)))),
            max(0.0, min(1.0, float(pred.get("confidence", 0.5))))
        )
    except:
        return 0.5, 0.5


def _idempotency_key(user_id, query):
    return hashlib.md5(f"{user_id}:{query}".encode()).hexdigest()


# ─────────────────────────────────────────────
# LOG SYMPTOM
# ─────────────────────────────────────────────
@app.post("/log-symptom")
async def log_symptom(data: LogSymptomRequest):

    ok = insert_symptom_log(
        user_id=data.user_id,
        feature_vector=data.feature_vector,
        raw_text=_sanitize_text(data.notes),
        emoji=_sanitize_text(data.emoji, 10)
    )

    if not ok:
        raise HTTPException(503, "Failed to store symptom log in database")

    asyncio.create_task(asyncio.to_thread(build_features))

    return {"status": "logged"}


# ─────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────
@app.post("/predict")
async def run_predict(data: PredictRequest):

    start = time.time()
    request_key = _idempotency_key(data.user_id, data.symptoms)

    try:
        logs = fetch_recent_logs(data.user_id)
        if logs is None:
            raise Exception("DB Fetch Failed")
    except Exception as e:
        logger.error(f"[API] DB Error on fetch_recent_logs: {e}")
        raise HTTPException(503, "Database unavailable")

    seq = build_sequence(logs)

    try:
        user_history = await asyncio.to_thread(get_full_history, data.user_id)
    except Exception:
        user_history = []

    # ───────── EXACTLY ONE PIPELINE EXECUTION
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                full_pipeline,
                user_id=data.user_id,
                query=data.symptoms,
                sequence=seq,
                user_history=user_history
            ),
            timeout=PIPELINE_TIMEOUT
        )
    except Exception as e:
        logger.error(f"[API] pipeline failed: {e}")
        result = {
            "prediction": {"severity": 0.5, "confidence": 0.3},
            "rag": {"answer": "System is currently degraded. Please consult a healthcare professional.", "sources": []}
        }

    sev, conf = safe_prediction(result.get("prediction", {}))
    rag = result.get("rag", {})

    override = result.get("reasoning", {}).get("override", False)

    if override:
        insert_guardrail_log(user_id=data.user_id, query=data.symptoms, severity=sev, message=rag.get("answer"))
        prediction_id = None
    else:
        try:
            prediction_id = insert_prediction(
                user_id=data.user_id,
                severity=sev,
                confidence=conf,
                reasoning=rag.get("answer", "")
            )
            if not prediction_id:
                raise Exception("Insert returned None")
        except Exception as e:
            logger.error(f"[API] prediction store failed: {e}")
            raise HTTPException(503, "Database unavailable")

        try:
            add_prediction_context(
                user_id=data.user_id,
                query=data.symptoms,
                severity=sev,
                confidence=conf
            )
        except Exception:
            pass

    latency = round((time.time() - start) * 1000, 2)

    return {
        "prediction": {
            "severity": round(sev, 4),
            "confidence": round(conf, 4),
            "prediction_id": prediction_id
        },
        "rag": rag,
        "request_id": request_key,
        "latency_ms": latency
    }


# ─────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────
@app.post("/feedback")
async def feedback(data: FeedbackRequest):

    predicted = max(0.0, min(1.0, data.predicted))
    actual = max(0.0, min(1.0, data.actual))
    rating = max(1, min(10, data.rating))

    trust_score = compute_trust_single({
        "predicted": predicted,
        "actual": actual,
        "rating": rating
    })

    try:
        ok = insert_feedback(
            user_id=data.user_id,
            prediction_id=data.prediction_id,
            predicted=predicted,
            actual=actual,
            rating=rating,
            trust_score=trust_score
        )
        if not ok: 
            raise Exception("Insert failed")
    except Exception as e:
        logger.error(f"[API] feedback insert failed: {e}")
        raise HTTPException(503, "Database unavailable")

    asyncio.create_task(
        asyncio.to_thread(update_personalization_from_feedback, data.user_id)
    )

    return {"status": "ok", "trust_score": round(trust_score, 4)}