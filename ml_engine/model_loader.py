import os
import json
import time
import logging
import torch
from typing import Optional, Tuple

from ml_engine.model_def import GRUModel

logger = logging.getLogger("menoeaze.model_loader")
logging.basicConfig(level=logging.INFO)

BASE_MODEL_NAME = "model.pt"
REGISTRY_NAME = "latest_model.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model: GRUModel, base_dir: Optional[str] = None) -> str:
    """
    Saves the model with a timestamped version and updates the registry.
    Ensures the original model.pt is NEVER overwritten.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    timestamp = int(time.time())
    meta_model_name = f"meta_model_{timestamp}.pt"
    save_path = os.path.join(base_dir, meta_model_name)
    
    # Save the state dict and architecture params safely
    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_size": model.gru.input_size,
        "timestamp": timestamp,
    }
    torch.save(ckpt, save_path)
    logger.info(f"[MODEL] Saved new incremental version: {meta_model_name}")
    
    # Update metadata registry safely
    registry_path = os.path.join(base_dir, REGISTRY_NAME)
    registry_data = {"latest_meta_model": meta_model_name, "timestamp": timestamp}
    
    with open(registry_path, "w") as f:
        json.dump(registry_data, f)
        
    logger.info(f"[MODEL] Registry updated to point to {meta_model_name}")
    return save_path


def load_model(base_dir: Optional[str] = None, device: Optional[torch.device] = None) -> Tuple[GRUModel, dict]:
    """
    Loads the latest model by checking the metadata registry.
    Safely falls back to the immutable base model.pt if the registry is missing or corrupt.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
    device = device or DEVICE
    base_path = os.path.join(base_dir, BASE_MODEL_NAME)
    registry_path = os.path.join(base_dir, REGISTRY_NAME)
    
    chosen_path = base_path
    source_type = "base"
    
    # Try to read the registry for the latest meta model
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                registry_data = json.load(f)
                latest_meta_model = registry_data.get("latest_meta_model")
                
            if latest_meta_model:
                meta_path = os.path.join(base_dir, latest_meta_model)
                if os.path.exists(meta_path):
                    chosen_path = meta_path
                    source_type = "meta"
                else:
                    logger.warning(f"[MODEL] Registry pointed to {latest_meta_model} but file missing. Falling back.")
        except Exception as e:
            logger.error(f"[MODEL] Failed to read registry: {e}. Falling back.")
            
    if not os.path.exists(chosen_path):
        raise FileNotFoundError(f"CRITICAL: Immutable base model missing at {base_path}")
        
    logger.info(f"[MODEL] Loading {source_type} model from: {os.path.basename(chosen_path)}")
    
    try:
        ckpt = torch.load(chosen_path, map_location=device, weights_only=False)
        
        # Defensive validation of checkpoint
        if "model_state_dict" not in ckpt or "input_size" not in ckpt:
            raise ValueError("Invalid checkpoint format")
            
        model = GRUModel(input_size=ckpt["input_size"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        
        metadata = {
            "source": os.path.basename(chosen_path),
            "input_size": ckpt.get("input_size"),
            "timestamp": ckpt.get("timestamp"),
            "type": source_type
        }
        
        return model, metadata
        
    except Exception as e:
        logger.exception("[MODEL] Failed to load checkpoint. Attempting ultimate fallback if needed.")
        if source_type == "meta":
            # If meta failed, attempt ultimate fallback to base
            logger.warning("[MODEL] Falling back to base model.pt due to meta model corruption.")
            ckpt = torch.load(base_path, map_location=device, weights_only=False)
            model = GRUModel(input_size=ckpt["input_size"])
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
            model.eval()
            return model, {"source": BASE_MODEL_NAME, "type": "base"}
        else:
            raise RuntimeError(f"Base model loading failed: {e}")

def validate_model(model: GRUModel, input_shape=(1, 5, 11)) -> bool:
    """Run a quick forward pass to ensure model is valid."""
    try:
        dummy = torch.zeros(input_shape).to(next(model.parameters()).device)
        with torch.no_grad():
            output = model(dummy)
        if torch.isnan(output).any():
            return False
        return True
    except Exception:
        return False
