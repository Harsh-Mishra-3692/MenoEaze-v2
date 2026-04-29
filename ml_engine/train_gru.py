"""
train_gru.py
============
Final research-grade GRU training pipeline.

Features:
- reproducibility (CPU + CUDA)
- stable training loop
- early stopping (best checkpoint)
- LR scheduler
- gradient clipping
- uncertainty estimation
- CSV logging
- test evaluation
"""

import os
import time
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split

# ── Config ─────────────────────────────────────────────
SEED = 42
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3
VAL_SPLIT = 0.1
PATIENCE = 12
GRAD_CLIP = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Reproducibility ─────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ── Model ───────────────────────────────────────────────
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden=64):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)


# ── Metrics ─────────────────────────────────────────────
def compute_metrics(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    return mae, rmse


# ── MC Dropout ──────────────────────────────────────────
def mc_predict(model, x, n_samples=20):
    model.train()
    preds = []

    for _ in range(n_samples):
        preds.append(model(x).unsqueeze(0))

    preds = torch.cat(preds, dim=0)
    return preds.mean(dim=0), preds.std(dim=0)


# ── Main ───────────────────────────────────────────────
def main():
    print("━" * 60)
    print("Final GRU Training Pipeline")
    print("━" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "preprocessed_data.pt")

    # FIX for PyTorch 2.6+
    data = torch.load(data_path, weights_only=False)

    X = data["X_train"]
    y = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    input_size = X.shape[2]

    print(f"Device        : {DEVICE}")
    print(f"Input size    : {input_size}")
    print(f"Train samples : {len(X)}")

    # ── Split ───────────────────────────────────────────
    dataset = TensorDataset(X, y)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # ── Model ───────────────────────────────────────────
    model = GRUModel(input_size).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    start_time = time.time()

    # ── Training Loop ───────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()

            preds = model(xb)
            loss = loss_fn(preds, yb)

            if torch.isnan(loss):
                raise ValueError("NaN detected in loss!")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= train_size

        # ── Validation ───────────────────────────────────
        model.eval()
        val_loss = 0.0
        preds_list, targets_list = [], []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)

                preds = model(xb)
                loss = loss_fn(preds, yb)

                val_loss += loss.item() * xb.size(0)

                preds_list.append(preds)
                targets_list.append(yb)

        val_loss /= val_size

        preds_cat = torch.cat(preds_list)
        targets_cat = torch.cat(targets_list)

        mae, rmse = compute_metrics(preds_cat, targets_cat)

        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:>3}/{EPOCHS} | "
            f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
            f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | LR: {lr_current:.6f}"
        )

        history.append([epoch, train_loss, val_loss, mae, rmse, lr_current])

        scheduler.step(val_loss)

        # ── Early stopping ───────────────────────────────
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": input_size,
                    "best_epoch": best_epoch,
                    "val_loss": best_loss,
                },
                os.path.join(base_dir, "model.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    duration = time.time() - start_time

    # ── Save history ────────────────────────────────────
    df = pd.DataFrame(
        history,
        columns=["epoch", "train_loss", "val_loss", "mae", "rmse", "lr"],
    )
    df.to_csv(os.path.join(base_dir, "training_history.csv"), index=False)

    print(f"\nTraining completed in {duration:.1f}s")
    print(f"Best epoch: {best_epoch}")

    # ── Test evaluation ─────────────────────────────────
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(DEVICE))

    mae_test, rmse_test = compute_metrics(preds, y_test.to(DEVICE))

    print("\nTest Performance:")
    print(f"MAE  : {mae_test:.4f}")
    print(f"RMSE : {rmse_test:.4f}")

    # ── Uncertainty ─────────────────────────────────────
    mean, std = mc_predict(model, X_test[:100].to(DEVICE))

    print("\nUncertainty sample (std):")
    print(std[:5])

    print("━" * 60)


if __name__ == "__main__":
    main()
