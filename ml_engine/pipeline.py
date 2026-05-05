# pipeline.py — PRODUCTION v2 (FULLY FAULT-TOLERANT)

import logging
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

logger = logging.getLogger("menoeaze.pipeline")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 5
EXPECTED_FEATURES = 56
ACCEPTED_FEATURES = (11, 56)


# ───────── INIT ─────────
_model, _meta = load_model()

if not _model:
    raise RuntimeError("MODEL FAILED TO LOAD")

_model.to(DEVICE)
_model.eval()

_adapter = PersonalizationAdapter(_model)
_llm = LLMClient()


# ───────── SAFE UTILS ─────────
def _clamp(x):
    try:
        return float(max(0.0, min(1.0, float(x))))
    except:
        return 0.5


def _validate_sequence(seq):
    if not isinstance(seq, np.ndarray) or seq.ndim != 2:
        raise ValueError("Invalid sequence")

    if seq.shape[0] != SEQ_LEN:
        raise ValueError("Invalid sequence length")

    if not np.isfinite(seq).all():
        raise ValueError("Non-finite values")

    if seq.shape[1] == EXPECTED_FEATURES:
        return seq.astype(np.float32)

    if seq.shape[1] in ACCEPTED_FEATURES:
        padded = np.zeros((SEQ_LEN, EXPECTED_FEATURES), dtype=np.float32)
        padded[:, :seq.shape[1]] = seq
        return padded

    raise ValueError("Invalid feature dimension")


# ───────── MODEL ─────────
def predict(sequence, user_id):
    seq = _validate_sequence(sequence)

    x = torch.tensor(seq).unsqueeze(0).to(DEVICE)

    weights = get_user_weights(user_id) or {}
    feedback = fetch_feedback_history(user_id) or []

    if select_strategy(None, feedback) == "none":
        weights = {}

    out = _adapter.predict(x, weights)

    return {
        "severity": _clamp(out.get("severity")),
        "confidence": _clamp(out.get("confidence", 0.5)),
        "personalized": bool(out.get("personalized"))
    }


# ───────── SAFE LLM FALLBACK ─────────
def _llm_fallback(query):
    try:
        res = _llm.generate(query)
        return res.get("text", "Unable to generate response.")
    except Exception as e:
        logger.error(f"LLM fallback failed: {e}")
        return "I'm unable to generate a detailed response right now. Please consult a healthcare professional."


# ───────── SAFE RAG ─────────
def rag(query, severity, user_id, trend_meta):

    try:
        docs = hybrid_retrieve(query, user_id=user_id)
    except Exception as e:
        logger.error(f"Retriever failed: {e}")
        docs = []

    # 🔥 FIX: NO HARD FAIL
    if not docs:
        logger.warning("No documents retrieved → fallback LLM")

        return {
            "answer": _llm_fallback(query),
            "sources": [],
            "confidence": 0.4
        }, {
            "confidence_adjusted": 0.3
        }

    try:
        docs = rerank(query, docs)
    except Exception as e:
        logger.error(f"Rerank failed: {e}")

    try:
        raw = generate_answer(query, severity, docs, None, _llm, trend_meta=trend_meta)
    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        return {
            "answer": _llm_fallback(query),
            "sources": [],
            "confidence": 0.4
        }, {
            "confidence_adjusted": 0.3
        }

    ctx = [d.get("content", "") for d in docs]

    try:
        eval_res = evaluate_rag(
            query=query,
            answer=raw.get("answer", ""),
            retrieved_docs=ctx,
            context_docs=ctx,
            base_confidence=raw.get("confidence", 0.5)
        )
    except Exception:
        eval_res = {}

    return raw, eval_res


# ───────── MAIN ─────────
def full_pipeline(user_id, query, sequence, user_history):

    pred = predict(sequence, user_id)

    severity = pred["severity"]

    feedback = fetch_feedback_history(user_id) or []
    trust = _clamp(compute_trust_score(feedback, severity))

    trend_meta = {"direction": "stable", "variability": "low"}

    rag_out, eval_res = rag(query, severity, user_id, trend_meta)

    rag_conf = _clamp(eval_res.get("confidence_adjusted", 0.3))

    confidence = _clamp(
        0.4 * pred["confidence"] +
        0.3 * trust +
        0.3 * rag_conf
    )

    doctor = recommend_doctor(
        severity=severity,
        trend=None,
        history=user_history or [],
        query=query,
        confidence=confidence
    )

    return {
        "prediction": pred,
        "rag": rag_out,
        "doctor": doctor,
        "confidence": confidence,
        "status": "ok",
        "trend_meta": trend_meta
    }
