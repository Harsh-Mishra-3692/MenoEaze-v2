# pipeline.py — FINAL ELITE v2 (FULL CLINICAL ORCHESTRATOR)

import time
import logging
from typing import Dict, Any

import numpy as np
import torch

from ml_engine.model_loader import load_model
from ml_engine.hybrid_retriever import hybrid_retrieve
from ml_engine.reranker import rerank
from ml_engine.rag_engine import generate_answer
from ml_engine.rag_evaluator import evaluate_rag
from ml_engine.llm_client import LLMClient

from ml_engine.personalization_adapter import PersonalizationAdapter
from ml_engine.trust_filter import compute_trust_score
from ml_engine.memory import extract_user_signal
from ml_engine.db_client import get_user_weights, fetch_feedback_history
from ml_engine.gating import select_strategy
from ml_engine.doctor_recommender import recommend_doctor
from ml_engine.clinical_guardrail import evaluate_guardrail

logger = logging.getLogger("menoeaze.pipeline")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 5
FEATURES = 11
MAX_QUERY_LENGTH = 500
ANOMALY_THRESHOLD = 0.5


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
try:
    _model, _meta = load_model()
    _model.to(DEVICE)
    _model.eval()
    _adapter = PersonalizationAdapter(_model)
except Exception:
    logger.exception("[Pipeline] Model init failed")
    _model, _adapter = None, None

try:
    _llm = LLMClient()
except Exception:
    logger.exception("[Pipeline] LLM init failed")
    _llm = None


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────
def _sanitize_query(q):
    return (q or "").strip()[:MAX_QUERY_LENGTH]


def _clamp(x):
    try:
        return float(max(0.0, min(1.0, x)))
    except:
        return 0.5


def _default_pred():
    return {"severity": 0.5, "confidence": 0.2, "personalized": False}


def _validate_sequence(seq):
    if isinstance(seq, np.ndarray) and seq.shape == (SEQ_LEN, FEATURES):
        if np.isfinite(seq).all():
            return seq.astype(np.float32)
    return None


def _detect_anomaly(current, historical_avg):
    return abs(current - historical_avg) > ANOMALY_THRESHOLD


def _fallback_answer(level):
    if level == "low":
        return "Limited evidence. Monitor symptoms and maintain healthy habits."
    if level == "medium":
        return "Evidence is uncertain. Consider lifestyle changes and monitoring."
    return "Medical guidance insufficient. Please consult a healthcare professional."


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(sequence, user_id):

    seq = _validate_sequence(sequence)

    if seq is None or _adapter is None:
        return _default_pred()

    try:
        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        weights = get_user_weights(user_id) or {}
        feedback = fetch_feedback_history(user_id) or []

        strategy = select_strategy({"feedback_logs": feedback})

        if strategy == "none":
            return _default_pred()

        out = _adapter.predict(x, weights)

        return {
            "severity": _clamp(out.get("severity")),
            "confidence": _clamp(out.get("confidence")),
            "personalized": bool(out.get("personalized"))
        }

    except:
        logger.exception("[Pipeline] Prediction failed")
        return _default_pred()


# ─────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────
def rag(query, severity, user_id):

    docs = hybrid_retrieve(query, user_id=user_id)
    if not docs:
        return None, {"decision": "unsafe"}

    docs = rerank(query, docs)

    raw = generate_answer(
        query=query,
        severity=severity,
        docs=docs,
        symptoms=None,
        llm_client=_llm
    )

    context_docs = [d["content"] for d in docs]

    eval_res = evaluate_rag(
        query=query,
        answer=raw.get("answer", ""),
        retrieved_docs=context_docs,
        context_docs=context_docs,
        base_confidence=raw.get("confidence", 0.5)
    )

    return raw, eval_res


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def full_pipeline(user_id, query, sequence, user_history):

    t0 = time.time()
    timings = {}

    try:
        query = _sanitize_query(query)

        # ───────── SIGNAL
        t = time.time()
        signal = extract_user_signal(user_id) or {}
        timings["signal"] = time.time() - t

        # ───────── PREDICTION
        t = time.time()
        pred = predict(sequence, user_id)
        timings["prediction"] = time.time() - t

        severity = pred["severity"]
        confidence = pred["confidence"]

        trend = signal.get("trend", "unknown")
        avg_sev = signal.get("avg_severity", severity)

        # ───────── ANOMALY CHECK
        anomaly = _detect_anomaly(severity, avg_sev)

        # ───────── GUARDRAIL
        t = time.time()
        guard = evaluate_guardrail(query, severity)
        timings["guardrail"] = time.time() - t

        if guard.get("override"):
            return {
                "prediction": pred,
                "rag": {"answer": guard.get("message"), "sources": []},
                "reasoning": {"override": True, "anomaly": anomaly},
                "doctor": {"recommend": True, "urgency": "immediate"},
                "meta": signal,
                "latency_ms": int((time.time() - t0) * 1000)
            }

        # ───────── RAG
        t = time.time()
        rag_out, eval_res = rag(query, severity, user_id)
        timings["rag"] = time.time() - t

        level = "low" if severity < 0.3 else "medium" if severity < 0.6 else "high"

        if not rag_out or eval_res["decision"] in ["unsafe", "weak"]:
            rag_out = {
                "answer": _fallback_answer(level),
                "sources": []
            }

        # ───────── TRUST
        t = time.time()
        feedback = fetch_feedback_history(user_id) or []
        trust = _clamp(compute_trust_score(feedback, severity))
        timings["trust"] = time.time() - t

        rag_conf = eval_res.get("confidence_adjusted", 0.3)

        # ───────── ADAPTIVE CONFIDENCE
        confidence = _clamp(
            0.4 * confidence +
            0.3 * trust +
            0.3 * rag_conf
        )

        # ───────── DOCTOR
        doctor = recommend_doctor(
            severity=severity,
            trend=trend,
            history=user_history or [],
            query=query,
            confidence=confidence
        )

        # ───────── BAND
        uncertainty = 1 - confidence
        band = {
            "lower": _clamp(severity - 0.2 * uncertainty),
            "upper": _clamp(severity + 0.2 * uncertainty)
        }

        return {
            "prediction": {
                "severity": round(severity, 3),
                "confidence": round(confidence, 3),
                "personalized": pred["personalized"]
            },
            "rag": rag_out,
            "reasoning": {
                "trend": trend,
                "anomaly": anomaly,
                "confidence_band": band,
                "factors": {
                    "trust": trust,
                    "rag_quality": eval_res.get("composite_score", 0)
                }
            },
            "doctor": doctor,
            "meta": signal,
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "latency_ms": int((time.time() - t0) * 1000)
        }

    except Exception:
        logger.exception("[Pipeline] CRITICAL")

        return {
            "prediction": _default_pred(),
            "rag": {"answer": "System error occurred.", "sources": []},
            "reasoning": {},
            "doctor": {"recommend": False},
            "meta": {},
            "latency_ms": 0
        }