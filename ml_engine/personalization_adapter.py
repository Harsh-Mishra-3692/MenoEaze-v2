# personalization_adapter.py — FINAL HARDENED (CLINICAL + SAFE + STABLE)

import logging
import numpy as np
import torch
from typing import Optional, Dict, Any

logger = logging.getLogger("menoeaze.personalization")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_GLOBAL_ADJUST = 0.15
MAX_FEATURE_ADJUST = 0.10
MAX_TEMPORAL_ADJUST = 0.08

MIN_CONFIDENCE = 0.2
MAX_CONFIDENCE = 1.0

FEATURES = 11


# ─────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────
def _clamp(v, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(v)))
    except:
        return lo


def _safe_array(arr, size):
    try:
        arr = np.array(arr, dtype=np.float32)
        if arr.shape[0] == size and np.isfinite(arr).all():
            return arr
    except:
        pass
    return np.zeros(size, dtype=np.float32)


def _safe_tensor(x):
    try:
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            return x
    except:
        pass
    return None


def _normalize(v):
    norm = np.linalg.norm(v) + 1e-6
    return v / norm


# ─────────────────────────────────────────────
# PERSONALIZATION ADAPTER
# ─────────────────────────────────────────────
class PersonalizationAdapter:

    def __init__(self, base_model):
        self.base = base_model

    # ─────────────────────────────────────────
    # SAFE BASE MODEL CALL
    # ─────────────────────────────────────────
    def _safe_base(self, x):

        try:
            out = self.base(x)

            if isinstance(out, dict):
                return {
                    "severity": _clamp(out.get("severity", 0.5)),
                    "confidence": _clamp(out.get("confidence", 0.5))
                }

            return {
                "severity": _clamp(out),
                "confidence": 0.5
            }

        except Exception as e:
            logger.error(f"[Personalization] base model failed: {e}")
            return {"severity": 0.5, "confidence": 0.5}

    # ─────────────────────────────────────────
    # MAIN PREDICT
    # ─────────────────────────────────────────
    def predict(
        self,
        x,
        user_weights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:

        x = _safe_tensor(x)

        if x is None:
            return {"severity": 0.5, "confidence": 0.3, "personalized": False}

        base = self._safe_base(x)

        severity = base["severity"]
        confidence = base["confidence"]

        # ─────────────────────────
        # NO PERSONALIZATION
        # ─────────────────────────
        if not user_weights:
            return {
                "severity": severity,
                "confidence": confidence,
                "personalized": False
            }

        user_conf = _clamp(user_weights.get("confidence", 0.0), 0.0, MAX_CONFIDENCE)

        if user_conf < MIN_CONFIDENCE:
            return {
                "severity": severity,
                "confidence": confidence,
                "personalized": False
            }

        try:
            x_np = x.detach().cpu().numpy()

            last_step = x_np[0, -1] if x_np.shape[1] > 0 else np.zeros(FEATURES)

            if not np.isfinite(last_step).all():
                last_step = np.zeros(FEATURES)

            # ─────────────────────────
            # GLOBAL BIAS
            # ─────────────────────────
            bias = np.clip(
                _clamp(user_weights.get("bias", 0.0), -1.0, 1.0),
                -MAX_GLOBAL_ADJUST,
                MAX_GLOBAL_ADJUST
            )

            # ─────────────────────────
            # FEATURE ADJUST
            # ─────────────────────────
            fw = _safe_array(user_weights.get("feature_weights", []), FEATURES)
            fw = np.clip(fw, -MAX_FEATURE_ADJUST, MAX_FEATURE_ADJUST)

            feature_adjust = float(np.dot(_normalize(last_step), fw)) / FEATURES

            # ─────────────────────────
            # LoRA SAFE
            # ─────────────────────────
            A = _safe_array(user_weights.get("lora_A", []), FEATURES)
            B = _safe_array(user_weights.get("lora_B", []), FEATURES)

            A = _normalize(A)
            B = _normalize(B)

            lora_effect = float(np.dot(A, last_step) * np.dot(B, last_step))
            lora_effect /= (FEATURES + 1e-6)
            lora_effect = np.clip(lora_effect, -MAX_FEATURE_ADJUST, MAX_FEATURE_ADJUST)

            # ─────────────────────────
            # TEMPORAL (ROBUST)
            # ─────────────────────────
            seq_std = float(np.std(x_np))
            temporal_weight = _clamp(user_weights.get("temporal_weight", 0.0), -1.0, 1.0)

            temporal_adjust = np.clip(
                seq_std * temporal_weight,
                -MAX_TEMPORAL_ADJUST,
                MAX_TEMPORAL_ADJUST
            )

            # ─────────────────────────
            # COMBINE (STRICTLY BOUNDED)
            # ─────────────────────────
            total_adjust = (
                bias +
                feature_adjust +
                lora_effect +
                temporal_adjust
            )

            total_adjust *= user_conf
            total_adjust = np.clip(total_adjust, -MAX_GLOBAL_ADJUST, MAX_GLOBAL_ADJUST)

            final_severity = _clamp(severity + total_adjust)

            # safer confidence update
            confidence_delta = (abs(total_adjust) * 0.3) * user_conf
            final_confidence = _clamp(confidence + confidence_delta)

            return {
                "severity": final_severity,
                "confidence": final_confidence,
                "personalized": True
            }

        except Exception as e:
            logger.error(f"[Personalization] failed: {e}")

            return {
                "severity": severity,
                "confidence": confidence,
                "personalized": False
            }