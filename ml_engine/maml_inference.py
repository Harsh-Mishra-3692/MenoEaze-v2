# maml_inference.py — PHASE 5 (MAML FAST ADAPTATION AT INFERENCE TIME)
#
# HARD CONSTRAINTS:
#   1. NEVER mutate the global _model. All adaptation is on an isolated deep copy.
#   2. Max 2 inner-loop gradient steps to bound latency.
#   3. Explicit memory cleanup (del clone, torch.cuda.empty_cache) after every call.
#   4. Graceful fallback: if adaptation fails for any reason, return None so the
#      caller can fall through to the standard EMA bias path.

import copy
import gc
import logging
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from ml_engine.model_def import GRUModel

logger = logging.getLogger("menoeaze.maml_inference")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAML_INNER_LR = 0.01       # Fast adaptation learning rate
MAML_INNER_STEPS = 2       # Strictly bounded: 1-3 gradient steps max
MIN_SUPPORT_SAMPLES = 3    # Minimum user feedback logs to attempt MAML
MAX_SUPPORT_SAMPLES = 10   # Cap to bound memory and latency

SEQ_LEN = 5
NUM_FEATURES = 11


# ─────────────────────────────────────────────
# SUPPORT SET BUILDER
# ─────────────────────────────────────────────
def _build_support_set(
    user_feedback: List[Dict[str, Any]],
) -> Optional[tuple]:
    """
    Converts raw user feedback history dicts into a mini support-set
    of (X, y) tensors for fast adaptation.

    Each feedback entry is expected to contain:
      - 'sequence': List[List[float]] of shape (5, 11)
      - 'actual_severity': float

    Returns:
        (X, y) tensors or None if insufficient/invalid data.
    """
    sequences = []
    targets = []

    for entry in user_feedback[-MAX_SUPPORT_SAMPLES:]:
        seq = entry.get("sequence")
        severity = entry.get("actual_severity")

        if seq is None or severity is None:
            continue

        try:
            arr = np.array(seq, dtype=np.float32)
            sev = float(severity)
        except (ValueError, TypeError):
            continue

        # Defensive shape check
        if arr.shape != (SEQ_LEN, NUM_FEATURES):
            continue

        # Defensive NaN/Inf check
        if np.isnan(arr).any() or np.isinf(arr).any():
            continue
        if np.isnan(sev) or np.isinf(sev):
            continue

        sequences.append(arr)
        targets.append(sev)

    if len(sequences) < MIN_SUPPORT_SAMPLES:
        return None

    X = torch.tensor(np.array(sequences), dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    # Final shape assertion
    assert X.shape == (len(sequences), SEQ_LEN, NUM_FEATURES), f"X shape violation: {X.shape}"

    return X, y


# ─────────────────────────────────────────────
# FAST ADAPTATION (CORE MAML INNER LOOP)
# ─────────────────────────────────────────────
def fast_adapt_user_model(
    base_model: GRUModel,
    user_feedback: List[Dict[str, Any]],
    steps: int = MAML_INNER_STEPS,
    lr: float = MAML_INNER_LR,
) -> Optional[GRUModel]:
    """
    Performs MAML-style fast adaptation on an ISOLATED clone of the base model.

    The global base_model is NEVER mutated. A deep copy is created,
    fine-tuned for 1-3 steps on the user's support set, and returned.

    The caller is responsible for running inference on the clone and then
    deleting it immediately.

    Args:
        base_model: The global shared GRU model. READ ONLY.
        user_feedback: List of user feedback dicts with 'sequence' and 'actual_severity'.
        steps: Number of inner-loop gradient steps (strictly 1-3).
        lr: Inner-loop learning rate.

    Returns:
        Adapted GRUModel clone, or None if adaptation is not possible.
    """
    # Enforce step bounds to guarantee latency
    steps = max(1, min(steps, 3))

    # Build the support set from user history
    support = _build_support_set(user_feedback)
    if support is None:
        return None

    X_support, y_support = support
    device = next(base_model.parameters()).device

    X_support = X_support.to(device)
    y_support = y_support.to(device)

    clone = None
    try:
        # 1. Create a fully isolated deep copy — NEVER touch base_model
        clone = copy.deepcopy(base_model)
        clone.train()

        # 2. Use SGD for the inner loop (standard MAML uses SGD, not Adam)
        inner_optimizer = torch.optim.SGD(clone.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # 3. Fast inner-loop: bounded gradient steps
        for step in range(steps):
            inner_optimizer.zero_grad()

            preds = clone(X_support)
            loss = criterion(preds, y_support)

            loss.backward()

            # Gradient clipping to prevent exploding gradients on small support sets
            torch.nn.utils.clip_grad_norm_(clone.parameters(), max_norm=1.0)

            inner_optimizer.step()

        clone.eval()

        logger.info(
            f"[MAML] Fast adaptation complete | "
            f"support_size={X_support.size(0)} | steps={steps} | "
            f"final_loss={loss.item():.4f}"
        )

        return clone

    except Exception as e:
        logger.error(f"[MAML] Fast adaptation failed: {e}")
        # Clean up the failed clone
        if clone is not None:
            del clone
        return None

    finally:
        # Always clean up support tensors
        del X_support, y_support
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────
# SAFE INFERENCE WITH CLEANUP
# ─────────────────────────────────────────────
def maml_predict(
    base_model: GRUModel,
    query_input: torch.Tensor,
    user_feedback: List[Dict[str, Any]],
) -> Optional[float]:
    """
    Full MAML inference pipeline with guaranteed memory cleanup.

    1. Adapts a clone of base_model to the user's feedback history.
    2. Runs inference on the adapted clone.
    3. Deletes the clone immediately and forces garbage collection.

    Returns:
        Adapted severity prediction (float), or None if MAML not applicable.
    """
    adapted_model = None
    try:
        adapted_model = fast_adapt_user_model(base_model, user_feedback)

        if adapted_model is None:
            return None

        with torch.no_grad():
            pred = adapted_model(query_input).item()

        pred = float(np.clip(pred, 0.0, 1.0))

        logger.info(f"[MAML] Inference complete | maml_severity={pred:.4f}")
        return pred

    except Exception as e:
        logger.error(f"[MAML] Inference failed: {e}")
        return None

    finally:
        # CRITICAL: Explicit cleanup to prevent OOM after thousands of requests
        if adapted_model is not None:
            del adapted_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
