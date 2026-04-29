# meta_train.py — ELITE (RESEARCH-GRADE MAML TRAINING)

import os
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn

from ml_engine.model_def import GRUModel
from ml_engine.maml_utils import (
    inner_loop_update,
    compute_query_loss,
    build_patient_tasks,
    sample_task_batch,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
META_EPOCHS = 200
META_LR = 5e-4
INNER_LR = 1e-3
INNER_STEPS = 3
GRAD_CLIP = 1.0
TASK_BATCH_SIZE = 8
PATIENCE = 25
VAL_SPLIT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger("menoeaze.meta")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    set_seed(SEED)

    logger.info("━" * 60)
    logger.info("  MAML Meta-Training Pipeline (Elite)")
    logger.info("━" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Load Data ────────────────────────────
    data = torch.load(os.path.join(base_dir, "preprocessed_data.pt"), weights_only=False)

    X_train = data["X_train"].numpy()
    y_train = data["y_train"].numpy()
    train_pids = data.get("train_patient_ids")

    # ── Build Tasks ──────────────────────────
    tasks = build_patient_tasks(
        X_train, y_train, train_pids,
        support_ratio=0.5,
        device=DEVICE,
    )

    if len(tasks) < 5:
        raise RuntimeError("Not enough tasks for meta-learning")

    # ── Split train / val tasks ──────────────
    random.shuffle(tasks)
    split_idx = int(len(tasks) * (1 - VAL_SPLIT))
    train_tasks = tasks[:split_idx]
    val_tasks = tasks[split_idx:]

    logger.info(f"Train tasks: {len(train_tasks)} | Val tasks: {len(val_tasks)}")

    # ── Load Base Model ──────────────────────
    ckpt = torch.load(os.path.join(base_dir, "model.pt"), weights_only=False)

    model = GRUModel(input_size=ckpt["input_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=META_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        meta_optimizer, mode="min", patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    start_time = time.time()

    # ── TRAIN LOOP ───────────────────────────
    for epoch in range(1, META_EPOCHS + 1):

        model.train()
        task_batch = sample_task_batch(train_tasks, TASK_BATCH_SIZE)

        meta_loss = torch.tensor(0.0, device=DEVICE)

        for task in task_batch:
            adapted = inner_loop_update(
                model,
                task["support_x"],
                task["support_y"],
                inner_lr=INNER_LR,
                inner_steps=INNER_STEPS,
                grad_clip=GRAD_CLIP,
            )

            q_loss = compute_query_loss(
                adapted,
                task["query_x"],
                task["query_y"],
            )

            meta_loss += q_loss

        meta_loss /= len(task_batch)

        if torch.isnan(meta_loss):
            logger.warning(f"NaN loss at epoch {epoch}")
            continue

        meta_optimizer.zero_grad()
        meta_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        meta_optimizer.step()

        # ── VALIDATION ───────────────────────
        model.eval()
        val_loss = torch.tensor(0.0, device=DEVICE)

        with torch.no_grad():
            val_batch = sample_task_batch(val_tasks, min(len(val_tasks), TASK_BATCH_SIZE))

            for task in val_batch:
                adapted = inner_loop_update(
                    model,
                    task["support_x"],
                    task["support_y"],
                    inner_lr=INNER_LR,
                    inner_steps=INNER_STEPS,
                    grad_clip=GRAD_CLIP,
                )

                q_loss = compute_query_loss(
                    adapted,
                    task["query_x"],
                    task["query_y"],
                )

                val_loss += q_loss

        val_loss /= len(val_batch)

        train_loss_val = meta_loss.item()
        val_loss_val = val_loss.item()

        scheduler.step(val_loss_val)

        history.append((epoch, train_loss_val, val_loss_val))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:>4} | Train: {train_loss_val:.6f} | Val: {val_loss_val:.6f}"
            )

        # ── CHECKPOINT ───────────────────────
        if val_loss_val < best_val_loss:
            best_val_loss = val_loss_val
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": ckpt["input_size"],
                    "best_epoch": best_epoch,
                    "val_loss": best_val_loss,
                },
                os.path.join(base_dir, "meta_model.pt"),
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    duration = time.time() - start_time

    logger.info("━" * 60)
    logger.info(f"Completed in {duration:.1f}s")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best val loss: {best_val_loss:.6f}")
    logger.info("Saved: meta_model.pt")
    logger.info("━" * 60)


if __name__ == "__main__":
    main()