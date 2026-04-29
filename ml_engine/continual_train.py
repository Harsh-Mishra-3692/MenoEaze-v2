import copy
import logging
import torch
import torch.nn as nn
from typing import Tuple

from ml_engine.model_loader import load_model, save_model

logger = logging.getLogger("menoeaze.continual_train")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
REPTILE_EPSILON = 0.1    # Soft interpolation rate for Reptile meta-update
INCREMENTAL_LR = 1e-4    # Low LR to prevent catastrophic forgetting
INCREMENTAL_EPOCHS = 3   # Bounded epoch count
GRAD_CLIP_NORM = 1.0     # Gradient clipping max norm


def train_incremental(batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
    """
    Performs safe incremental learning with Reptile-style meta-weight update.

    Hard Constraints enforced:
    - Extremely low learning rate (1e-4) to prevent catastrophic forgetting
    - Exactly 3 epochs for bounded updates
    - Gradient clipping to prevent exploding gradients
    - Reptile soft interpolation: θ_meta += ε * (θ_task - θ_meta)
      This optimizes the initialization for MAML fast adaptation.
    """
    X, y = batch

    # 1. Safely unzip and convert to float32
    X = X.to(torch.float32)
    y = y.to(torch.float32)

    # Soft shape validation (asserts crash the daemon thread with cryptic errors)
    if X.dim() != 3:
        logger.error(f"[TRAIN] Shape violation: Expected 3D tensor for X (batch, seq, features), got {X.dim()}D. Skipping batch.")
        return
    if y.dim() != 2:
        logger.error(f"[TRAIN] Shape violation: Expected 2D tensor for y (batch, 1), got {y.dim()}D. Skipping batch.")
        return

    # 2. Load the current active model safely
    try:
        model, metadata = load_model()
    except Exception as e:
        logger.error(f"[TRAIN] Failed to load model for incremental training: {e}")
        return

    # 3. Snapshot the meta-weights BEFORE task-specific training (Reptile)
    meta_state = copy.deepcopy(model.state_dict())

    model.train()

    # 4. Setup Optimizer and Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=INCREMENTAL_LR)
    criterion = nn.MSELoss()

    # 5. Run exactly 3 epochs (task-specific inner loop)
    for epoch in range(INCREMENTAL_EPOCHS):
        optimizer.zero_grad()

        predictions = model(X)
        predictions = predictions.view_as(y)
        loss = criterion(predictions, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

        optimizer.step()

    logger.info(f"[TRAIN] Inner loop finished. Final batch loss: {loss.item():.4f}")

    # 6. Reptile meta-update: θ_meta = θ_meta + ε * (θ_task - θ_meta)
    #    This biases the global model toward initializations that are
    #    easy to fine-tune for individual users (exactly what MAML needs).
    task_state = model.state_dict()
    reptile_state = {}

    for key in meta_state:
        reptile_state[key] = (
            meta_state[key] + REPTILE_EPSILON * (task_state[key] - meta_state[key])
        )

    model.load_state_dict(reptile_state)
    logger.info(f"[TRAIN] Reptile meta-update applied (ε={REPTILE_EPSILON})")

    # 7. Save the updated model
    try:
        model.eval()
        save_model(model)
        logger.info("[TRAIN] Meta-model successfully saved and registered.")
    except Exception as e:
        logger.error(f"[TRAIN] Failed to save meta-model: {e}")

