# pipeline.py — ELITE v3 (PRODUCTION + RESEARCH + ABLATION READY)

import time
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch

from ml_engine.model_loader import load_model
from ml_engine.bias_control import compute_stable_bias, apply_bias
from ml_engine.hybrid_retriever import hybrid_retrieve, adaptive_retrieve
from ml_engine.reranker import rerank
from ml_engine.rag_engine import generate_answer
from ml_engine.llm_client import LLMClient
from ml_engine.gating import select_strategy
from ml_engine.adaptation import apply_adaptation, prepare_adaptation_data
from ml_engine.adaptation_worker import enqueue_adaptation
from ml_engine.maml_inference import maml_predict

logger = logging.getLogger("menoeaze.pipeline")

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# GLOBAL INIT
# ─────────────────────────────────────────────
try:
    _model, _model_meta = load_model()
    _model.to(DEVICE)
    _model.eval()
except Exception as e:
    logger.error(f"[Pipeline] Failed to load base model: {e}")
    _model, _model_meta = None, {}

try:
    _llm = LLMClient()
except Exception as e:
    logger.error(f"[Pipeline] Failed to init LLM client: {e}")
    _llm = None

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ADAPT_COOLDOWN = 5
MIN_DOCS_REQUIRED = 1

_last_adapt_time: Dict[str, float] = {}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _prepare_input(sequence: np.ndarray) -> torch.Tensor:
    if sequence.shape != (5, 11):
        raise ValueError("Invalid input shape")
    return torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def _safe_history(user_history: Optional[Dict]) -> Dict:
    return user_history or {"predictions": [], "actuals": []}


def _compute_confidence(pred: float, bias_conf: float) -> float:
    return float(np.clip(0.5 + 0.5 * bias_conf, 0.0, 1.0))


# ─────────────────────────────────────────────
# PREDICTION PIPELINE
# ─────────────────────────────────────────────
def predict_pipeline(
    user_id: str,
    sequence: np.ndarray,
    user_history: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    start = time.time()

    try:
        user_history = _safe_history(user_history)
        x = _prepare_input(sequence)

        # CONCURRENCY SAFETY: Capture a local reference to the model.
        # If _periodic_model_refresh hot-swaps pipeline._model during
        # this forward pass, the local ref keeps the old model alive
        # until this request completes (Python GC won't collect it).
        model_ref = _model

        if model_ref is None:
            logger.warning("[Pipeline] Model is missing. Returning degraded response.")
            return {"status": "degraded", "reason": "Model missing", "data": {}}

        with torch.no_grad():
            raw_output = model_ref(x).item()

        # NORMALIZATION DRIFT DETECTION: If the raw model output is far
        # outside [0,1], the input scalers are stale or the model has drifted.
        # Clipping to [0,1] masks this — MAML/bias cannot adapt at the boundary.
        if raw_output < -0.5 or raw_output > 1.5:
            logger.warning(
                f"[Pipeline] NORMALIZATION DRIFT: raw GRU output={raw_output:.4f} "
                f"is far outside [0,1]. Input scalers may be stale. "
                f"Consider rerunning preprocess.py and retraining."
            )

        base_pred = float(np.clip(raw_output, 0, 1))

        strategy = select_strategy(user_history)

        final_pred = base_pred
        bias_data = {"bias": 0.0, "confidence": 0.0, "variance": 0.0}

        if strategy == "bias":
            bias_data = compute_stable_bias(
                user_history["predictions"],
                user_history["actuals"]
            )

            final_pred = apply_bias(
                base_pred,
                bias_data["bias"],
                bias_data["confidence"]
            )

        elif strategy == "maml":
            # Phase 5: MAML fast adaptation on an ISOLATED model clone.
            # Requires user_history to contain feedback entries with
            # 'sequence' and 'actual_severity' keys.
            feedback_logs = user_history.get("feedback_logs", [])

            if feedback_logs:
                maml_pred = maml_predict(model_ref, x, feedback_logs)

                if maml_pred is not None:
                    final_pred = maml_pred
                    logger.info(f"[Pipeline] MAML adaptation used | pred={maml_pred:.4f}")
                else:
                    # MAML failed — fall through to base prediction (backward compat)
                    logger.warning("[Pipeline] MAML returned None, using base prediction.")
            else:
                logger.info("[Pipeline] MAML selected but no feedback_logs available, using base.")

        elif strategy == "adapt":
            adapted_pred = apply_adaptation(model_ref, x, user_id)

            if adapted_pred is not None:
                final_pred = adapted_pred

            now = time.time()
            last_time = _last_adapt_time.get(user_id, 0)

            if now - last_time > ADAPT_COOLDOWN:
                data = prepare_adaptation_data(x, user_history)

                if data:
                    enqueue_adaptation({
                        "user_id": user_id,
                        "sequence": data["sequence"],
                        "targets": data["targets"]
                    })
                    _last_adapt_time[user_id] = now

        confidence = _compute_confidence(final_pred, bias_data["confidence"])
        latency = (time.time() - start) * 1000

        data = {
            "severity": round(final_pred, 4),
            "base_severity": round(base_pred, 4),
            "confidence": round(confidence, 3),
            "strategy": strategy,
            "bias": bias_data,
            "latency_ms": round(latency, 2),
        }
        return {"status": "success", "reason": "OK", "data": data}

    except Exception as e:
        logger.error(f"[Pipeline] Prediction failed: {e}")
        return {"status": "error", "reason": str(e), "data": {}}


# ─────────────────────────────────────────────
# RAG PIPELINE (WITH ABLATION SUPPORT)
# ─────────────────────────────────────────────
def rag_pipeline(
    query: str,
    severity: float,
    symptoms: Optional[Dict[str, float]] = None,
    use_reranker: bool = True,  # 🔥 NEW
    return_debug: bool = False  # 🔥 NEW
) -> Dict[str, Any]:

    start = time.time()

    try:
        # ── Retrieval (PHASE 4: Severity-Conditioned Adaptive RAG)
        try:
            raw_docs = adaptive_retrieve(query, severity=severity)
        except Exception as e:
            logger.warning(f"[Pipeline] Adaptive retrieval failed, falling back to static: {e}")
            try:
                raw_docs = hybrid_retrieve(query)
            except Exception as e2:
                logger.warning(f"[Pipeline] Static retrieval also failed: {e2}")
                raw_docs = []

        docs = raw_docs

        # ── Reranker (ABLATION CONTROL)
        if use_reranker and docs:
            docs = rerank(query, docs)

        # ── Quality gate
        if len(docs) < MIN_DOCS_REQUIRED:
            logger.warning("[Pipeline] Low retrieval quality")
            docs = []

        result = generate_answer(
            query=query,
            severity=severity,
            docs=docs,
            symptoms=symptoms,
            llm_client=_llm
        )

        latency = (time.time() - start) * 1000

        result["latency_ms"] = round(latency, 2)
        result["docs_used"] = len(docs)

        # 🔥 DEBUG DATA FOR RESEARCH
        if return_debug:
            result["debug"] = {
                "raw_docs": raw_docs,
                "reranked_docs": docs,
                "reranker_used": use_reranker
            }

        return {"status": "success", "reason": "OK", "data": result}

    except Exception as e:
        logger.error(f"[Pipeline] RAG failed: {e}")
        return {
            "status": "error",
            "reason": str(e),
            "data": {
                "answer": "Unable to generate response at the moment.",
                "confidence": 0.0,
                "fallback": True,
            }
        }


# ─────────────────────────────────────────────
# FULL PIPELINE (RESEARCH ENABLED)
# ─────────────────────────────────────────────
def full_pipeline(
    user_id: str,
    query: str,
    sequence: Optional[np.ndarray] = None,
    user_history: Optional[Dict[str, Any]] = None,
    use_reranker: bool = True,     # 🔥 NEW
    return_debug: bool = False     # 🔥 NEW
) -> Dict[str, Any]:

    start = time.time()

    try:
        severity = 0.5
        pred_out = None

        pred_status = "success"
        pred_reason = "OK"

        if sequence is not None:
            pred_res = predict_pipeline(user_id, sequence, user_history)
            pred_out = pred_res.get("data", {})
            pred_status = pred_res.get("status", "error")
            pred_reason = pred_res.get("reason", "")
            severity = pred_out.get("severity", 0.5)

        symptoms = {}
        if user_history:
            recent_preds = [p for p in user_history.get("predictions", []) if p is not None]
            if recent_preds:
                symptoms["historical_severity"] = sum(recent_preds) / len(recent_preds)
                symptoms["trend"] = "increasing" if recent_preds[-1] > symptoms["historical_severity"] else "stable"
                
        if sequence is not None and len(sequence) > 0:
            last_log = sequence[-1]
            if len(last_log) >= 6:
                symptoms["hot_flashes"] = last_log[0]
                symptoms["night_sweats"] = last_log[1]
                symptoms["poor_sleep"] = 1.0 - last_log[2] # Inverted: higher is worse
                symptoms["low_mood"] = 1.0 - last_log[3]
                symptoms["fatigue"] = last_log[4]
                symptoms["anxiety"] = last_log[5]

        rag_res = rag_pipeline(
            query=query,
            severity=severity,
            symptoms=symptoms,
            use_reranker=use_reranker,
            return_debug=return_debug
        )
        rag_out = rag_res.get("data", {})
        rag_status = rag_res.get("status", "error")
        rag_reason = rag_res.get("reason", "")

        total_latency = (time.time() - start) * 1000

        data = {
            "prediction": pred_out,
            "rag": rag_out,
            "total_latency_ms": round(total_latency, 2),
        }
        
        status = "success"
        reason = "OK"
        if pred_status != "success" or rag_status != "success":
            status = "degraded"
            reason = f"Prediction: {pred_reason} | RAG: {rag_reason}"
            
        return {"status": status, "reason": reason, "data": data}

    except Exception as e:
        logger.error(f"[Pipeline] Full pipeline failed: {e}")
        return {"status": "error", "reason": str(e), "data": {}}