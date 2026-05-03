# pipeline.py — FINAL ELITE v4 (DETERMINISTIC + SAFE + AUDITABLE)

import time
import logging
from typing import Dict, Any, Optional, List

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

try:
    from ml_engine.clinical_guardrail import evaluate_guardrail
except Exception:
    def evaluate_guardrail(query: str, severity: float) -> Dict[str, Any]:
        return {"safe": True, "override": False, "flags": [], "message": None}

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
    if _model:
        _model.to(DEVICE)
        _model.eval()
        _adapter = PersonalizationAdapter(_model)
    else:
        _adapter = None
except Exception:
    logger.exception("[PIPELINE][INIT][FAIL] Model init failed")
    _model, _adapter = None, None

try:
    _llm = LLMClient()
except Exception:
    logger.exception("[PIPELINE][INIT][FAIL] LLM init failed")
    _llm = None


# ─────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────
def _sanitize_query(q: Optional[str]) -> str:
    if not isinstance(q, str):
        return ""
    q = q.strip()
    return q[:MAX_QUERY_LENGTH]


def _clamp(x: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(x))))
    except Exception:
        return 0.5


def _default_pred() -> Dict[str, Any]:
    return {"severity": 0.5, "confidence": 0.2, "personalized": False}


def _validate_sequence(seq: Any) -> Optional[np.ndarray]:
    if isinstance(seq, np.ndarray) and seq.shape == (SEQ_LEN, FEATURES):
        if np.isfinite(seq).all():
            return seq.astype(np.float32)
    return None


def _detect_anomaly(current: float, historical_avg: float) -> bool:
    return abs(current - historical_avg) > ANOMALY_THRESHOLD


def _fallback_answer(level: str) -> str:
    if level == "low":
        return "Limited evidence. Monitor symptoms and maintain healthy habits."
    if level == "medium":
        return "Evidence is uncertain. Consider lifestyle changes and monitoring."
    return "Medical guidance insufficient. Please consult a healthcare professional."


# ─────────────────────────────────────────────
# DEGRADATION TRACKING (FIXED)
# ─────────────────────────────────────────────
class DegradationTracker:
    def __init__(self):
        self.reasons: List[str] = []

    def add(self, reason: str):
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def export(self) -> Dict[str, Any]:
        if not self.reasons:
            return {}
        return {
            "status": "degraded",
            "reasons": self.reasons
        }


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(sequence: Any, user_id: str, tracker: DegradationTracker) -> Dict[str, Any]:

    seq = _validate_sequence(sequence)
    if seq is None or _adapter is None:
        tracker.add("invalid_sequence_or_model")
        return _default_pred()

    try:
        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        weights = get_user_weights(user_id) or {}
        feedback = fetch_feedback_history(user_id) or []

        strategy = select_strategy({"feedback_logs": feedback})
        if strategy == "none":
            weights = {}

        out = _adapter.predict(x, weights)

        if not isinstance(out, dict):
            tracker.add("invalid_prediction_output")
            return _default_pred()

        return {
            "severity": _clamp(out.get("severity")),
            "confidence": _clamp(out.get("confidence")),
            "personalized": bool(out.get("personalized"))
        }

    except Exception:
        logger.exception("[PIPELINE][GRU][FAIL]")
        tracker.add("prediction_failure")
        return _default_pred()


# ─────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────
def rag(query: str, severity: float, user_id: str, tracker: DegradationTracker):

    try:
        docs = hybrid_retrieve(query, user_id=user_id)

        if not docs:
            tracker.add("no_retrieval_docs")
            return None, {}

        docs = rerank(query, docs)

        raw = generate_answer(
            query=query,
            severity=severity,
            docs=docs,
            symptoms=None,
            llm_client=_llm
        )

        if not isinstance(raw, dict):
            tracker.add("invalid_llm_output")
            return None, {}

        context_docs = [d.get("content", "") for d in docs if d.get("content")]

        eval_res = evaluate_rag(
            query=query,
            answer=raw.get("answer", ""),
            retrieved_docs=context_docs,
            context_docs=context_docs,
            base_confidence=raw.get("confidence", 0.5)
        )

        return raw, eval_res or {}

    except Exception:
        logger.exception("[PIPELINE][RAG][FAIL]")
        tracker.add("rag_failure")
        return None, {}


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def full_pipeline(user_id, query, sequence, user_history):

    t0 = time.time()
    timings = {}
    tracker = DegradationTracker()

    try:
        # VALIDATION
        query = _sanitize_query(query)
        if not user_id or not query:
            raise ValueError("Invalid input")

        # SIGNAL
        t = time.time()
        signal = extract_user_signal(user_id) or {}
        timings["signal"] = time.time() - t

        # PREDICTION
        t = time.time()
        pred = predict(sequence, user_id, tracker)
        timings["prediction"] = time.time() - t

        severity = pred["severity"]
        confidence = pred["confidence"]

        trend = signal.get("trend", "unknown")
        avg_sev = signal.get("avg_severity", severity)
        anomaly = _detect_anomaly(severity, avg_sev)

        # TRUST
        t = time.time()
        try:
            feedback = fetch_feedback_history(user_id) or []
            trust = _clamp(compute_trust_score(feedback, severity))
        except Exception:
            tracker.add("trust_failure")
            trust = 0.5
        timings["trust"] = time.time() - t

        # GUARDRAIL
        t = time.time()
        try:
            guard = evaluate_guardrail(query, severity)
        except Exception:
            tracker.add("guardrail_failure")
            guard = {"override": False}
        timings["guardrail"] = time.time() - t

        if guard.get("override"):
            response = {
                "prediction": pred,
                "rag": {"answer": guard.get("message") or "Seek medical attention.", "sources": []},
                "reasoning": {"override": True, "anomaly": anomaly},
                "doctor": {"recommend": True, "urgency": "immediate"},
                "meta": signal
            }
            response.update(tracker.export())
            response["latency_ms"] = int((time.time() - t0) * 1000)
            return response

        # RAG
        t = time.time()
        rag_out, eval_res = rag(query, severity, user_id, tracker)
        timings["rag"] = time.time() - t

        level = "low" if severity < 0.3 else "medium" if severity < 0.6 else "high"

        if not rag_out:
            rag_out = {"answer": _fallback_answer(level), "sources": []}

        rag_conf = _clamp(eval_res.get("confidence_adjusted", 0.3))

        # CONFIDENCE FUSION
        confidence = _clamp(
            0.4 * confidence +
            0.3 * trust +
            0.3 * rag_conf
        )

        # DOCTOR
        t = time.time()
        try:
            doctor = recommend_doctor(
                severity=severity,
                trend=trend,
                history=user_history or [],
                query=query,
                confidence=confidence
            )
        except Exception:
            tracker.add("doctor_failure")
            doctor = {"recommend": False, "urgency": "none"}
        timings["doctor"] = time.time() - t

        # RESPONSE
        uncertainty = 1 - confidence
        band = {
            "lower": _clamp(severity - 0.2 * uncertainty),
            "upper": _clamp(severity + 0.2 * uncertainty)
        }

        response = {
            "prediction": {
                "severity": round(severity, 3),
                "confidence": round(confidence, 3),
                "personalized": pred.get("personalized", False)
            },
            "rag": rag_out,
            "reasoning": {
                "trend": trend,
                "anomaly": anomaly,
                "confidence_band": band,
                "factors": {
                    "trust": round(trust, 3),
                    "rag_quality": round(eval_res.get("composite_score", 0), 3)
                }
            },
            "doctor": doctor,
            "meta": signal,
            "timings": {k: round(v, 4) for k, v in timings.items()},
            "latency_ms": int((time.time() - t0) * 1000)
        }

        response.update(tracker.export())
        return response

    except Exception:
        logger.exception("[PIPELINE][CRITICAL][FAIL]")
        return {
            "prediction": _default_pred(),
            "rag": {"answer": "System error occurred. Please consult a healthcare professional.", "sources": []},
            "reasoning": {},
            "doctor": {"recommend": False},
            "meta": {},
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "failed"
        }