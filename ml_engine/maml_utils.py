"""
maml_utils.py
=============
MAML (Model-Agnostic Meta-Learning) utilities.

Each patient = one meta-learning task.
Support set → inner-loop adaptation.
Query set   → outer-loop meta-gradient.

Constraints (non-negotiable):
- Inner loop: max 3 steps, gradient clipping ≤ 1.0
- Outer loop: standard Adam update on query loss
- Never called inside the API — offline only
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict

from ml_engine.model_def import GRUModel


# ── Inner Loop ───────────────────────────────────────────────
def inner_loop_update(
    model: GRUModel,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    inner_lr: float = 1e-3,
    inner_steps: int = 3,
    grad_clip: float = 1.0,
) -> GRUModel:
    """
    Perform K inner-loop gradient steps on the support set.

    Returns a cloned model with adapted parameters.
    The original model is NEVER modified.

    Args:
        model: base model (will be deepcopied)
        support_x: (N, seq_len, features)
        support_y: (N,)
        inner_lr: learning rate for inner updates
        inner_steps: max gradient steps (capped at 3)
        grad_clip: max gradient norm (capped at 1.0)
    """
    # Safety caps
    inner_steps = min(inner_steps, 3)
    grad_clip = min(grad_clip, 1.0)

    adapted = copy.deepcopy(model)
    adapted.train()

    loss_fn = nn.MSELoss()

    for _ in range(inner_steps):
        pred = adapted(support_x)
        loss = loss_fn(pred, support_y)

        # Check for NaN
        if torch.isnan(loss):
            return copy.deepcopy(model)  # fallback to base

        adapted.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), grad_clip)

        # Manual SGD step (MAML uses plain SGD for inner loop)
        with torch.no_grad():
            for param in adapted.parameters():
                if param.grad is not None:
                    param -= inner_lr * param.grad

    return adapted


# ── Query Loss ───────────────────────────────────────────────
def compute_query_loss(
    adapted_model: GRUModel,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
) -> torch.Tensor:
    """
    Compute MSE loss on the query set using the adapted model.
    This loss is backpropagated through the inner-loop to update
    the meta-parameters.
    """
    adapted_model.train()
    pred = adapted_model(query_x)
    return nn.MSELoss()(pred, query_y)


# ── Task Construction ────────────────────────────────────────
def build_patient_tasks(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    support_ratio: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> List[Dict[str, torch.Tensor]]:
    """
    Construct meta-learning tasks from preprocessed data.
    Each patient becomes one task with support/query split.

    Returns:
        List of dicts with keys:
            'support_x', 'support_y', 'query_x', 'query_y'
    """
    tasks = []
    unique_pids = np.unique(patient_ids)

    for pid in unique_pids:
        mask = patient_ids == pid
        px = X[mask]
        py = y[mask]

        if len(px) < 4:  # need at least 2+2 for support/query
            continue

        split = max(2, int(len(px) * support_ratio))

        tasks.append({
            "support_x": torch.tensor(px[:split], dtype=torch.float32).to(device),
            "support_y": torch.tensor(py[:split], dtype=torch.float32).to(device),
            "query_x": torch.tensor(px[split:], dtype=torch.float32).to(device),
            "query_y": torch.tensor(py[split:], dtype=torch.float32).to(device),
        })

    return tasks


# ── Task Sampling ────────────────────────────────────────────
def sample_task_batch(
    tasks: List[Dict[str, torch.Tensor]],
    batch_size: int = 8,
) -> List[Dict[str, torch.Tensor]]:
    """
    Randomly sample a batch of tasks for one meta-update step.
    """
    indices = np.random.choice(len(tasks), size=min(batch_size, len(tasks)), replace=False)
    return [tasks[i] for i in indices]
