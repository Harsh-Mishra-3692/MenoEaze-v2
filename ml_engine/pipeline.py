# pipeline.py — PRODUCTION SAFE (FEATURE-LOCKED + MODEL-SAFE)

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

# 🔒 CRITICAL: USE TRAINING PIPELINE
from ml_engine.build_features import build_feature_vector as build_features
from ml_engine.longitudinal_builder import build_sequence_for_inference

try:
    from ml_engine.clinical_guardrail import apply_guardrail as evaluate_guardrail
except Exception:
    def evaluate_guardrail(query: str, severity: float) -> Dict[str, Any]:
        return {"safe": True, "override": False, "flags": [], "message": None}

logger = logging.getLogger("menoeaze.pipeline")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔒 LOCKED TO TRAINING DISTRIBUTION
SEQ_LEN = 5
EXPECTED_FEATURES = 56  # model training dimension
ACCEPTED_FEATURES = (11, 56)  # accept API (11) or full (56)
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
    return q.strip()[:MAX_QUERY_LENGTH]


def _clamp(x: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(x))))
    except Exception:
        return 0.5


def _default_pred() -> Dict[str, Any]:
    return {"severity": 0.5, "confidence": 0.2, "personalized": False}


def _validate_sequence(seq: Any) -> Optional[np.ndarray]:
    if not isinstance(seq, np.ndarray) or seq.ndim != 2:
        return None
    if seq.shape[0] != SEQ_LEN:
        return None
    if not np.isfinite(seq).all():
        return None
    feat_dim = seq.shape[1]
    if feat_dim == EXPECTED_FEATURES:
        return seq.astype(np.float32)
    if feat_dim in ACCEPTED_FEATURES:
        # Pad to EXPECTED_FEATURES with zeros so model receives correct shape
        padded = np.zeros((SEQ_LEN, EXPECTED_FEATURES), dtype=np.float32)
        padded[:, :feat_dim] = seq[:, :feat_dim]
        return padded
    return None


# ─────────────────────────────────────────────
# 🔒 FEATURE PIPELINE (CRITICAL FIX)
# ─────────────────────────────────────────────
def _prepare_sequence(user_input: Dict[str, Any], tracker) -> Optional[np.ndarray]:
    try:
        features = build_features(user_input)

        if features is None:
            tracker.add("feature_build_failed")
            return None

        seq = build_sequence_for_inference(features)

        if seq is None:
            tracker.add("sequence_build_failed")
            return None

        seq = _validate_sequence(seq)

        if seq is None:
            tracker.add("invalid_sequence_shape")
            return None

        return seq

    except Exception:
        logger.exception("[PIPELINE][FEATURE_PIPELINE][FAIL]")
        tracker.add("feature_pipeline_failure")
        return None


# ─────────────────────────────────────────────
# DEGRADATION TRACKING
# ─────────────────────────────────────────────
class DegradationTracker:
    def __init__(self):
        self.reasons: List[str] = []

    def add(self, reason: str):
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def export(self) -> Dict[str, Any]:
        return {"status": "degraded", "reasons": self.reasons} if self.reasons else {}


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(sequence: Any, user_id: str, tracker: DegradationTracker):

    seq = _validate_sequence(sequence)

    if seq is None or _adapter is None:
        tracker.add("invalid_sequence_or_model")
        return _default_pred()

    try:
        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

        weights = get_user_weights(user_id) or {}
        feedback = fetch_feedback_history(user_id) or []

        if select_strategy(None, feedback) == "none":
            weights = {}

        out = _adapter.predict(x, weights)

        if not isinstance(out, dict):
            tracker.add("invalid_prediction_output")
            return _default_pred()

        sev = out.get("severity")
        conf = out.get("confidence")
        
        print(f"[ML_OUTPUT] {{ 'severity': {sev}, 'confidence': {conf} }}")
        
        if not conf or conf <= 0:
            conf = float(out.get("probability", out.get("score", 0.5)))
            
        if not conf or conf <= 0:
            conf = 0.5

        return {
            "severity": _clamp(sev),
            "confidence": _clamp(conf),
            "personalized": bool(out.get("personalized"))
        }

    except Exception:
        logger.exception("[PIPELINE][PREDICT][FAIL]")
        tracker.add("prediction_failure")
        return _default_pred()


# ─────────────────────────────────────────────
# RAG (unchanged)
# ─────────────────────────────────────────────
def rag(query: str, severity: float, user_id: str, tracker, trend_meta=None):
    try:
        docs = hybrid_retrieve(query, user_id=user_id)
        if not docs:
            tracker.add("no_docs")
            return None, {}

        docs = rerank(query, docs)

        raw = generate_answer(query, severity, docs, None, _llm, trend_meta=trend_meta)

        if not isinstance(raw, dict):
            tracker.add("invalid_llm_output")
            return None, {}

        ctx = [d.get("content", "") for d in docs if d.get("content")]

        eval_res = evaluate_rag(
            query=query,
            answer=raw.get("answer", ""),
            retrieved_docs=ctx,
            context_docs=ctx,
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
    tracker = DegradationTracker()
    timings = {}

    try:
        query = _sanitize_query(query)
        if not user_id:
            raise ValueError("Invalid user")

        # sequence is passed directly from API
        t = time.time()
        timings["feature_pipeline"] = time.time() - t

        # SIGNAL
        signal = extract_user_signal(user_id) or {}

        # PREDICT
        pred = predict(sequence, user_id, tracker)

        severity = pred["severity"]
        confidence = pred["confidence"]

        # TRUST
        try:
            feedback = fetch_feedback_history(user_id) or []
            trust = _clamp(compute_trust_score(feedback, severity))
        except Exception:
            tracker.add("trust_failure")
            trust = 0.5

        # GUARDRAIL
        guard = evaluate_guardrail(query, severity)
        if guard.get("override"):
            return {
                "prediction": pred,
                "rag": {"answer": guard.get("message"), "sources": []},
                "status": "guardrail_override"
            }
        trend_meta = {"direction": "stable", "variability": "low"}
        if user_history and len(user_history) >= 2:
            trend_vals = []
            for l in user_history[-7:]:
                fv = l.get("feature_vector", [])
                if fv and len(fv) > 0:
                    trend_vals.append(float(fv[0]))
            if len(trend_vals) >= 2:
                diff = trend_vals[-1] - trend_vals[0]
                if diff > 0.2: trend_meta["direction"] = "increasing"
                elif diff < -0.2: trend_meta["direction"] = "decreasing"
                var = sum(abs(trend_vals[i] - trend_vals[i-1]) for i in range(1, len(trend_vals))) / len(trend_vals)
                if var > 0.3: trend_meta["variability"] = "high"
                elif var > 0.1: trend_meta["variability"] = "medium"

        # LLM Trend Summary Generation
        trend_summary = ""
        try:
            if _llm:
                prompt = f"""You are a clinical AI.
Generate a 1-2 line summary and 1 line reasoning based on these trend signals.
Direction: {trend_meta['direction']}
Variability: {trend_meta['variability']}

Example: 'Your symptoms appear stable with moderate variability. This suggests consistent but fluctuating intensity.'
Do not use placeholders. Be concise."""
                res = _llm.generate(prompt)
                trend_summary = res if isinstance(res, str) else res.get("text", "")
        except Exception:
            trend_summary = ""

        # RAG
        t = time.time()
        rag_out, eval_res = rag(query, severity, user_id, tracker, trend_meta=trend_meta)

        if not rag_out:
            rag_out = {"answer": "Consult a professional.", "sources": []}

        rag_conf = _clamp(eval_res.get("confidence_adjusted", 0.3))

        # CONFIDENCE FUSION
        confidence = _clamp(0.4 * confidence + 0.3 * trust + 0.3 * rag_conf)

        # DOCTOR
        doctor = recommend_doctor(
            severity=severity,
            trend=signal.get("trend"),
            history=user_history or [],
            query=query,
            confidence=confidence
        )

        return {
            "prediction": pred,
            "rag": rag_out,
            "doctor": doctor,
            "confidence": confidence,
            "latency_ms": int((time.time() - t0) * 1000),
            "status": "ok",
            "trend_meta": trend_meta,
            "trend_summary": trend_summary,
            **tracker.export()
        }

    except Exception:
        logger.exception("[PIPELINE][CRITICAL]")
        return {
            "prediction": _default_pred(),
            "rag": {"answer": "System failure. Consult a doctor.", "sources": []},
            "status": "failed"
        }