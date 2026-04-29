import time
import logging
import threading
import torch

# Import exactly the modules from Phase 1 and Phase 2
from ml_engine.online_learning_buffer import OnlineLearningBuffer
from ml_engine.continual_train import train_incremental

logger = logging.getLogger("menoeaze.adaptation_worker")
logging.basicConfig(level=logging.INFO)

# Global shared instance of the buffer (Phase 1)
learning_buffer = OnlineLearningBuffer(max_capacity=10000)

def worker_loop(check_interval_seconds: int = 60, min_batch_size: int = 50):
    """
    Immortal background worker loop for asynchronous continual learning.
    Constantly checks the shared buffer for new batches and triggers training.
    """
    logger.info("[WORKER] Adaptation worker started. Polling buffer...")
    
    while True:
        try:
            # Pull a batch from Phase 1 buffer
            x_batch, y_batch = learning_buffer.get_batch(min_size=min_batch_size)
            
            # If batch exists and is valid
            if x_batch.numel() > 0 and y_batch.numel() > 0:
                logger.info(f"[WORKER] Found valid batch of size {x_batch.size(0)}. Triggering incremental training...")
                
                # Trigger Phase 2 training
                train_incremental((x_batch, y_batch))
                
                logger.info(f"[WORKER] Batch processed. Sleeping for {check_interval_seconds}s to prevent CPU thrashing...")
                
            # Sleep to yield CPU and prevent thrashing
            time.sleep(check_interval_seconds)
            
        except Exception as e:
            # Immortal worker constraint: Catch all, log, and continue
            logger.error(f"[WORKER] CRITICAL ERROR in adaptation loop: {e}. Worker will not die. Sleeping before retry.")
            time.sleep(check_interval_seconds)

def start_worker_thread(check_interval_seconds: int = 60, min_batch_size: int = 50) -> threading.Thread:
    """
    Spawns and starts the worker loop in a daemon thread.
    """
    t = threading.Thread(
        target=worker_loop, 
        args=(check_interval_seconds, min_batch_size), 
        daemon=True,
        name="AdaptationWorkerThread"
    )
    t.start()
    return t


# ─────────────────────────────────────────────
# LEGACY COMPAT: enqueue_adaptation
# ─────────────────────────────────────────────
# pipeline.py imports this for MAML-style per-user adaptation.
# We route it through the same shared learning_buffer so the
# background worker can pick it up.
def enqueue_adaptation(task: dict) -> None:
    """
    Accepts a per-user adaptation task dict and pipes the data
    into the shared learning buffer for asynchronous training.

    Expected keys: 'sequence' (Tensor/list), 'targets' (Tensor/list).
    """
    try:
        seq = task.get("sequence")
        targets = task.get("targets")

        if seq is None or targets is None:
            return

        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)

        # Flatten batch into individual samples for the buffer
        if seq.dim() == 3:
            for i in range(seq.size(0)):
                learning_buffer.add_sample(seq[i], targets[i].unsqueeze(0) if targets[i].dim() == 0 else targets[i])
        elif seq.dim() == 2:
            t = targets.unsqueeze(0) if targets.dim() == 0 else targets
            learning_buffer.add_sample(seq, t)

        logger.info(f"[WORKER] Enqueued adaptation task for user={task.get('user_id', 'unknown')}")

    except Exception as e:
        logger.error(f"[WORKER] Failed to enqueue adaptation: {e}")
