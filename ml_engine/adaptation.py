# adaptation.py — ELITE (NON-BLOCKING PERSONALIZATION ENGINE)

import copy
import logging
from typing import Optional, Dict

import torch
import numpy as np

from ml_engine.model_def import GRUModel

logger = logging.getLogger("menoeaze.adaptation")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_HISTORY = 5
ADAPT_THRESHOLD = 15

# ─────────────────────────────────────────────
# LIGHTWEIGHT CACHE (TEMP — REPLACE WITH REDIS LATER)
# ─────────────────────────────────────────────
_adapt_cache: Dict[str, Dict] = {}


# ─────────────────────────────────────────────
# LOAD ADAPTED HEAD (NO TRAINING HERE)
# ─────────────────────────────────────────────
def load_adapted_model(
    base: GRUModel,
    user_id: Optional[str] = None,
) -> GRUModel:
    """
    Loads adapted FC layer if available.
    DOES NOT train.
    """

    if not user_id:
        return base

    try:
        state = _adapt_cache.get(user_id)

        if not state:
            return base

        model = copy.deepcopy(base)
        model.fc.load_state_dict(state)

        return model

    except Exception as e:
        logger.error(f"[Adapt] Load failed: {e}")
        return base


# ─────────────────────────────────────────────
# SAVE ADAPTATION (CALLED BY WORKER ONLY)
# ─────────────────────────────────────────────
def save_adaptation(
    user_id: str,
    fc_state: Dict
):
    """
    Save adapted head.
    Should be called by async worker ONLY.
    """

    try:
        _adapt_cache[user_id] = fc_state
        logger.info(f"[Adapt] Saved adaptation for user={user_id}")

    except Exception as e:
        logger.error(f"[Adapt] Save failed: {e}")


# ─────────────────────────────────────────────
# DECIDE WHETHER TO TRIGGER ADAPTATION
# ─────────────────────────────────────────────
def should_adapt(user_history: Optional[Dict]) -> bool:
    if not user_history:
        return False

    history_len = len(user_history.get("actuals", []))

    return history_len >= ADAPT_THRESHOLD


# ─────────────────────────────────────────────
# PREPARE DATA FOR WORKER
# ─────────────────────────────────────────────
def prepare_adaptation_data(
    sequence: torch.Tensor,
    user_history: Dict
) -> Optional[Dict]:
    """
    Prepare training batch for async worker.
    """

    try:
        if sequence is None or user_history is None:
            return None

        actuals = user_history.get("actuals")

        if not actuals or len(actuals) < MIN_HISTORY:
            return None

        return {
            "sequence": sequence.detach().cpu(),
            "targets": torch.tensor(actuals, dtype=torch.float32),
        }

    except Exception as e:
        logger.error(f"[Adapt] Data prep failed: {e}")
        return None


# ─────────────────────────────────────────────
# PERSONALIZATION APPLY (SAFE)
# ─────────────────────────────────────────────
def apply_adaptation(
    base_model: GRUModel,
    x: torch.Tensor,
    user_id: Optional[str] = None
) -> float:
    """
    Apply adapted model if available.
    No training.
    """

    try:
        model = load_adapted_model(base_model, user_id)

        model.eval()

        with torch.no_grad():
            pred = model(x).item()

        return float(np.clip(pred, 0, 1))

    except Exception as e:
        logger.error(f"[Adapt] Apply failed: {e}")
        return None
