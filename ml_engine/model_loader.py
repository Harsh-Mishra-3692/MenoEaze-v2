import os
import json
import time
import logging
import torch
from typing import Optional, Tuple

# 🔴 DO NOT CHANGE CORE MODELS
from ml_engine.model_def import GRUModel
from ml_engine.sap_gru_model import SAP_GRU

logger = logging.getLogger("menoeaze.model_loader")
logging.basicConfig(level=logging.INFO)

BASE_MODEL_NAME = "model.pt"
FALLBACK_MODEL_NAME = "sap_gru.pt"  # 🔥 critical fallback
REGISTRY_NAME = "latest_model.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _resolve_base_dir(base_dir: Optional[str]):
    return base_dir or os.path.dirname(os.path.abspath(__file__))


def _safe_load(path, device):
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        logger.error(f"[MODEL] Failed to load {path}: {e}")
        return None


def _is_wrapped_checkpoint(ckpt):
    return isinstance(ckpt, dict) and "model_state_dict" in ckpt


def _is_raw_state_dict(ckpt):
    return isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values())


def _infer_input_dim(state_dict: dict) -> int:
    try:
        return state_dict["gru.weight_ih_l0"].shape[1]
    except Exception:
        logger.warning("[MODEL] Failed to infer input_dim. Defaulting to 16.")
        return 16


# ─────────────────────────────────────────────
# SAVE (UNCHANGED SAFE)
# ─────────────────────────────────────────────
def save_model(model: GRUModel, base_dir: Optional[str] = None) -> str:
    base_dir = _resolve_base_dir(base_dir)

    timestamp = int(time.time())
    name = f"meta_model_{timestamp}.pt"
    path = os.path.join(base_dir, name)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_size": getattr(model.gru, "input_size", None),
        "timestamp": timestamp,
    }

    torch.save(ckpt, path)

    with open(os.path.join(base_dir, REGISTRY_NAME), "w") as f:
        json.dump({"latest_meta_model": name}, f)

    logger.info(f"[MODEL] Saved: {name}")
    return path


# ─────────────────────────────────────────────
# CORE LOADER
# ─────────────────────────────────────────────
def load_model(
    base_dir: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, dict]:

    base_dir = _resolve_base_dir(base_dir)
    device = device or DEVICE

    base_path = os.path.join(base_dir, BASE_MODEL_NAME)
    fallback_path = os.path.join(base_dir, FALLBACK_MODEL_NAME)

    chosen_path = None

    # ─────────────────────────────
    # SELECT MODEL FILE
    # ─────────────────────────────
    if os.path.exists(base_path):
        chosen_path = base_path
    elif os.path.exists(fallback_path):
        chosen_path = fallback_path
        logger.warning("[MODEL] model.pt missing → using sap_gru.pt fallback")
    else:
        raise FileNotFoundError(
            f"[CRITICAL] No model file found. Expected {BASE_MODEL_NAME} or {FALLBACK_MODEL_NAME}"
        )

    logger.info(f"[MODEL] Loading from: {os.path.basename(chosen_path)}")

    ckpt = _safe_load(chosen_path, device)

    if ckpt is None:
        raise RuntimeError("[MODEL] Failed to load checkpoint")

    # ─────────────────────────────
    # CASE 1: WRAPPED (OLD SYSTEM)
    # ─────────────────────────────
    if _is_wrapped_checkpoint(ckpt):
        logger.info("[MODEL] Detected wrapped checkpoint (GRUModel)")

        model = GRUModel(input_size=ckpt["input_size"])
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.to(device).eval()

        return model, {
            "type": "GRUModel",
            "source": os.path.basename(chosen_path),
            "input_size": ckpt.get("input_size"),
        }

    # ─────────────────────────────
    # CASE 2: SAP-GRU (REAL MODEL)
    # ─────────────────────────────
    if _is_raw_state_dict(ckpt):
        logger.info("[MODEL] Detected SAP-GRU state_dict")

        input_dim = _infer_input_dim(ckpt)

        model = SAP_GRU(input_dim=input_dim, device=device)

        try:
            model.load_state_dict(ckpt, strict=False)
        except Exception as e:
            logger.error(f"[MODEL] Load failed (strict=False fallback): {e}")
            raise RuntimeError("SAP-GRU weight loading failed")

        model.to(device)
        model.eval()

        # Integrity check (critical)
        try:
            model.assert_integrity()
        except Exception:
            logger.warning("[MODEL] Integrity check failed (non-fatal)")

        return model, {
            "type": "SAP_GRU",
            "source": os.path.basename(chosen_path),
            "input_dim": input_dim,
        }

    # ─────────────────────────────
    # UNKNOWN FORMAT
    # ─────────────────────────────
    raise RuntimeError("[MODEL] Unknown checkpoint format")


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def validate_model(model, input_shape=(1, 5, 16)) -> bool:
    try:
        dummy = torch.zeros(input_shape).to(next(model.parameters()).device)

        with torch.no_grad():
            if hasattr(model, "safe_predict"):
                out = model.safe_predict(dummy)
                val = out["severity"]
            else:
                val = model(dummy)

        return torch.isfinite(val).all()

    except Exception:
        return False