import torch
import torch.nn as nn
import numpy as np
import random
import os
import tempfile
import json
from datetime import datetime, UTC

from sap_gru_model import SAP_GRU
from sequence_loader import load_sequences, to_tensor
from research.metrics_report import MetricsReporter


# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "data/processed/longitudinal_sequences.csv"
MODEL_PATH = "sap_gru.pt"
CHECKPOINT_META = "sap_gru_meta.json"

EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_GRAD_NORM = 0.5
RANK_SUBSAMPLE = 32


# =========================================================
# 🔥 REPRODUCIBILITY (FIXED)
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
def safe_save(obj, path):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    torch.save(obj, tmp.name)
    tmp.close()
    os.replace(tmp.name, path)


# =========================================================
def save_metadata(best_val):
    meta = {
        "timestamp": datetime.now(UTC).isoformat(),  # ✅ FIXED
        "best_val_loss": float(best_val),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "device": str(DEVICE)
    }
    with open(CHECKPOINT_META, "w") as f:
        json.dump(meta, f, indent=2)


# =========================================================
def split_data(seq, tgt, length, weight, val_ratio=0.1):
    idx = np.random.permutation(len(seq))
    split = int(len(seq) * (1 - val_ratio))

    def subset(indices):
        return (
            [seq[i] for i in indices],
            [tgt[i] for i in indices],
            [length[i] for i in indices],
            [weight[i] for i in indices],
        )

    return subset(idx[:split]), subset(idx[split:])


# =========================================================
def weighted_mse(pred, target, weights):
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)

    pred, target, weights = pred[mask], target[mask], weights[mask]
    return (weights * (pred - target) ** 2).mean()


# =========================================================
def ranking_loss(y_pred, y_true):

    # 🔥 guard against low variance (important)
    if torch.std(y_true) < 1e-4:
        return torch.tensor(0.0, device=y_pred.device)

    if y_pred.size(0) > RANK_SUBSAMPLE:
        idx = torch.randperm(y_pred.size(0))[:RANK_SUBSAMPLE]
        y_pred = y_pred[idx]
        y_true = y_true[idx]

    diff_pred = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    diff_true = y_true.unsqueeze(1) - y_true.unsqueeze(0)

    return torch.mean((diff_pred - diff_true) ** 2)


# =========================================================
def get_rank_weight(epoch):
    if epoch < 5:
        return 0.0
    elif epoch < 10:
        return 0.05
    else:
        return 0.1


# =========================================================
class EMA:
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * p.data
                )

    def apply_to(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[name])


# =========================================================
def validate_tensor(t):
    return torch.isfinite(t).all()


# =========================================================
def train():
    set_seed(SEED)

    reporter = MetricsReporter()

    sequences, targets, lengths, weights = load_sequences(DATA_PATH)

    (seq_tr, tgt_tr, len_tr, w_tr), \
    (seq_val, tgt_val, len_val, w_val) = split_data(
        sequences, targets, lengths, weights
    )

    train_data = to_tensor(seq_tr, tgt_tr, len_tr, w_tr, DEVICE)
    val_data = to_tensor(seq_val, tgt_val, len_val, w_val, DEVICE)

    X_train, y_train, l_train, w_train = (
        train_data["X"], train_data["y"],
        train_data["lengths"], train_data["weights"]
    )

    X_val, y_val, l_val, w_val = (
        val_data["X"], val_data["y"],
        val_data["lengths"], val_data["weights"]
    )

    assert validate_tensor(X_train)
    assert validate_tensor(y_train)

    model = SAP_GRU(input_dim=X_train.shape[-1], device=DEVICE)
    model.assert_integrity()

    params = list(model.adapter.parameters()) + list(model.local_head.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    ema = EMA(model)

    best_val = float("inf")
    patience = 0

    print(f"\nTraining on {len(X_train)} | Validation on {len(X_val)}\n")

    # =========================================================
    for epoch in range(EPOCHS):
        model.train()

        perm = torch.randperm(X_train.size(0))

        total_loss = 0
        total_mse = 0
        total_rank = 0
        steps = 0
        skipped_batches = 0

        rank_weight = get_rank_weight(epoch)

        for start in range(0, len(perm), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]

            Xb = X_train[idx]
            yb = y_train[idx]
            lb = l_train[idx]
            wb = w_train[idx]

            if Xb.numel() == 0:
                skipped_batches += 1
                continue

            try:
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    out = model(Xb, lb)["severity"]

                    if not validate_tensor(out):
                        skipped_batches += 1
                        continue

                    mse = weighted_mse(out, yb, wb)

                    if rank_weight > 0:
                        rank = ranking_loss(out, yb)
                        loss = mse + rank_weight * rank
                    else:
                        rank = torch.tensor(0.0, device=DEVICE)
                        loss = mse

                if not torch.isfinite(loss):
                    skipped_batches += 1
                    continue

                scaler.scale(loss).backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)

                if not torch.isfinite(grad_norm):
                    skipped_batches += 1
                    continue

                scaler.step(optimizer)
                scaler.update()

                ema.update(model)

                total_loss += loss.item()
                total_mse += mse.item()
                total_rank += rank.item()
                steps += 1

            except RuntimeError as e:
                print(f"[WARNING] batch skipped: {str(e)[:100]}")
                skipped_batches += 1
                continue

        scheduler.step()

        train_loss = total_loss / max(steps, 1)

        # =========================================================
        # VALIDATION (NO RE-CREATION)
        # =========================================================
        model.eval()
        ema.apply_to(model)

        with torch.no_grad():
            val_out = model(X_val, l_val)["severity"]
            val_loss = weighted_mse(val_out, y_val, w_val).item()

        reporter.log_epoch(epoch, train_loss, val_loss, y_val, val_out)

        print(
            f"Epoch {epoch:02d} | "
            f"Train: {train_loss:.4f} | "
            f"MSE: {total_mse/max(steps,1):.4f} | "
            f"Rank: {total_rank/max(steps,1):.4f} | "
            f"RankW: {rank_weight:.2f} | "
            f"Skipped: {skipped_batches}"
        )

        reporter.print_latest()

        # =========================================================
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            safe_save(model.state_dict(), MODEL_PATH)
            save_metadata(best_val)
        else:
            patience += 1

        if patience >= PATIENCE:
            print("Early stopping.")
            break

    reporter.finalize()

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    train()