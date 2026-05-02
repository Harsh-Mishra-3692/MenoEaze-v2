# user_store.py — ELITE (CLEAN + CONSISTENT + SAFE DATA LAYER)

import logging
from datetime import datetime, timedelta, timezone # DEPRECATED: memory logic is now fully DB-based via Supabase.
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from ml_engine.db_client import fetch_table, insert_feedback

logger = logging.getLogger("menoeaze.user_store")

SEQ_SHAPE = (5, 11)


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def _validate_sequence(seq) -> Optional[np.ndarray]:
    try:
        arr = np.array(seq, dtype=np.float32)
        if arr.shape == SEQ_SHAPE:
            return arr
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────
# USER HISTORY
# ─────────────────────────────────────────────
def get_user_history(
    user_id: str,
    days: int = 30,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[float], List[float]]:

    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        # ── Fetch predictions ───────────────
        preds = fetch_table(
            table="prediction_history",
            limit=1000,
            filters={"user_id": user_id}
        )

        # ── Fetch feedback ──────────────────
        feedback = fetch_table(
            table="user_feedback",
            limit=1000,
            filters={"user_id": user_id}
        )

        if not preds:
            return None, None, [], []

        sequences = []
        pred_values = []

        for row in preds:
            if "created_at" in row and row["created_at"] < cutoff:
                continue

            seq = _validate_sequence(row.get("input_sequence"))
            if seq is not None:
                sequences.append(seq)
                pred_values.append(float(row.get("predicted_severity", 0.5)))

        if not sequences:
            return None, None, pred_values, []

        # ── Map feedback ────────────────────
        feedback_map = {
            f["prediction_id"]: f["actual_severity"]
            for f in feedback if "prediction_id" in f
        }

        actual_values = []
        aligned_actuals = []

        for row in preds:
            pid = row.get("id")
            if pid in feedback_map:
                actual_values.append(feedback_map[pid])

        # Align targets
        if actual_values and len(actual_values) >= len(sequences):
            aligned_actuals = actual_values[: len(sequences)]
        else:
            aligned_actuals = pred_values  # fallback

        history_x = np.array(sequences, dtype=np.float32)
        history_y = np.array(aligned_actuals, dtype=np.float32)

        return history_x, history_y, pred_values, actual_values

    except Exception as e:
        logger.error(f"[UserStore] history fetch failed: {e}")
        return None, None, [], []


# ─────────────────────────────────────────────
# STORE PREDICTION
# ─────────────────────────────────────────────
def store_prediction(
    user_id: str,
    input_sequence: list,
    predicted_severity: float,
    adapted: bool = False,
) -> Optional[str]:

    try:
        seq = _validate_sequence(input_sequence)
        if seq is None:
            logger.error("[UserStore] invalid sequence")
            return None

        record = {
            "user_id": user_id,
            "input_sequence": seq.tolist(),
            "predicted_severity": round(predicted_severity, 4),
            "adapted": adapted,
        }

        # Use db_client bulk insert pattern
        success = insert_feedback(record)  # reuse safe layer

        return "ok" if success else None

    except Exception as e:
        logger.error(f"[UserStore] store prediction failed: {e}")
        return None


# ─────────────────────────────────────────────
# STORE FEEDBACK
# ─────────────────────────────────────────────
def store_feedback(
    user_id: str,
    prediction_id: str,
    actual_severity: float,
) -> bool:

    try:
        record = {
            "user_id": user_id,
            "prediction_id": prediction_id,
            "actual": round(actual_severity, 4),
        }

        return insert_feedback(record)

    except Exception as e:
        logger.error(f"[UserStore] feedback failed: {e}")
        return False


# ─────────────────────────────────────────────
# AGGREGATED FEEDBACK (TRAINING)
# ─────────────────────────────────────────────
def get_aggregated_feedback(days: int = 90) -> List[Dict[str, Any]]:

    try:
        feedback = fetch_table("user_feedback", limit=5000)
        preds = fetch_table("prediction_history", limit=5000)

        pred_map = {p["id"]: p.get("input_sequence") for p in preds}

        results = []

        for fb in feedback:
            pid = fb.get("prediction_id")

            if pid in pred_map:
                seq = _validate_sequence(pred_map[pid])

                if seq is not None:
                    results.append({
                        "input_sequence": seq.tolist(),
                        "actual_severity": fb.get("actual_severity"),
                    })

        return results

    except Exception as e:
        logger.error(f"[UserStore] aggregation failed: {e}")
        return []