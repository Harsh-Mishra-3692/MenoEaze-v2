# evaluation.py — ELITE (SYSTEM EVALUATION MODULE)

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger("menoeaze.eval")

# ─────────────────────────────────────────────
# REGRESSION METRICS (ML)
# ─────────────────────────────────────────────
def evaluate_regression(
    predictions: List[float],
    actuals: List[float]
) -> Dict[str, float]:

    if not predictions or not actuals or len(predictions) != len(actuals):
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "mse": 0.0,
        }

    try:
        preds = np.array(predictions)
        acts = np.array(actuals)

        mae = mean_absolute_error(acts, preds)
        mse = mean_squared_error(acts, preds)
        rmse = np.sqrt(mse)

        return {
            "mae": round(float(mae), 5),
            "mse": round(float(mse), 5),
            "rmse": round(float(rmse), 5),
        }

    except Exception as e:
        logger.error(f"[Eval] Regression failed: {e}")
        return {"mae": 0.0, "mse": 0.0, "rmse": 0.0}


# ─────────────────────────────────────────────
# PERSONALIZATION GAIN
# ─────────────────────────────────────────────
def evaluate_personalization(
    base_preds: List[float],
    final_preds: List[float],
    actuals: List[float]
) -> Dict[str, float]:

    try:
        base_mae = mean_absolute_error(actuals, base_preds)
        final_mae = mean_absolute_error(actuals, final_preds)

        gain = base_mae - final_mae

        return {
            "base_mae": round(float(base_mae), 5),
            "final_mae": round(float(final_mae), 5),
            "improvement": round(float(gain), 5),
        }

    except Exception as e:
        logger.error(f"[Eval] Personalization failed: {e}")
        return {
            "base_mae": 0.0,
            "final_mae": 0.0,
            "improvement": 0.0,
        }


# ─────────────────────────────────────────────
# RETRIEVAL METRICS (RAG)
# ─────────────────────────────────────────────
def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]

    if not retrieved_k:
        return 0.0

    hits = sum(1 for doc in retrieved_k if doc in relevant)
    return hits / k


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]
    hits = sum(1 for doc in retrieved_k if doc in relevant)

    return hits / len(relevant)


def evaluate_retrieval(
    relevant_docs: List[str],
    retrieved_docs: List[str]
) -> Dict[str, float]:

    try:
        return {
            "precision@3": round(precision_at_k(relevant_docs, retrieved_docs, 3), 4),
            "precision@5": round(precision_at_k(relevant_docs, retrieved_docs, 5), 4),
            "recall@3": round(recall_at_k(relevant_docs, retrieved_docs, 3), 4),
            "recall@5": round(recall_at_k(relevant_docs, retrieved_docs, 5), 4),
        }

    except Exception as e:
        logger.error(f"[Eval] Retrieval failed: {e}")
        return {
            "precision@3": 0.0,
            "precision@5": 0.0,
            "recall@3": 0.0,
            "recall@5": 0.0,
        }


# ─────────────────────────────────────────────
# RAG RESPONSE QUALITY (PROXY METRICS)
# ─────────────────────────────────────────────
def evaluate_rag_response(
    answer: str,
    context_docs: List[str]
) -> Dict[str, float]:

    try:
        if not answer or not context_docs:
            return {"groundedness": 0.0}

        answer_tokens = set(answer.lower().split())
        context_tokens = set(" ".join(context_docs).lower().split())

        overlap = answer_tokens.intersection(context_tokens)

        groundedness = len(overlap) / len(answer_tokens) if answer_tokens else 0.0

        return {
            "groundedness": round(float(groundedness), 4)
        }

    except Exception as e:
        logger.error(f"[Eval] RAG response failed: {e}")
        return {"groundedness": 0.0}


# ─────────────────────────────────────────────
# FULL SYSTEM EVALUATION
# ─────────────────────────────────────────────
def evaluate_system(
    predictions: List[float],
    final_predictions: List[float],
    actuals: List[float],
    relevant_docs: Optional[List[str]] = None,
    retrieved_docs: Optional[List[str]] = None,
    rag_answer: Optional[str] = None,
    context_docs: Optional[List[str]] = None
) -> Dict[str, Any]:

    results = {}

    # ML performance
    results["regression"] = evaluate_regression(predictions, actuals)

    # personalization
    results["personalization"] = evaluate_personalization(
        predictions,
        final_predictions,
        actuals
    )

    # retrieval
    if relevant_docs and retrieved_docs:
        results["retrieval"] = evaluate_retrieval(
            relevant_docs,
            retrieved_docs
        )

    # RAG quality
    if rag_answer and context_docs:
        results["rag"] = evaluate_rag_response(
            rag_answer,
            context_docs
        )

    return results
