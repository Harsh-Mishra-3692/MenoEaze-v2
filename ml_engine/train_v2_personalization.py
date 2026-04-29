# train_v2_personalization.py

import torch
import torch.nn as nn
import numpy as np

from ml_engine.personalization_model import PersonalizationModel

DEVICE = torch.device("cpu")


# -----------------------------
# Improved Synthetic Data (Aligned with API)
# -----------------------------
def generate_data(n=8000):
    X = []
    y = []

    for _ in range(n):
        base = np.random.uniform(0.2, 0.8)

        mean_error = np.random.uniform(-0.3, 0.3)
        trend = np.random.uniform(-0.2, 0.2)
        variability = np.random.uniform(0.0, 0.3)

        # MATCH API LOGIC (IMPORTANT)
        correction = (
            0.2 * mean_error +
            0.1 * trend +
            0.05 * variability
        )

        # simulate EMA effect (approx)
        bias = np.random.uniform(-0.2, 0.2)

        total_adjustment = correction + bias

        # clamp like API
        total_adjustment = np.clip(total_adjustment, -0.3, 0.3)

        X.append([base, mean_error, trend, variability])
        y.append(total_adjustment)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# -----------------------------
# Training
# -----------------------------
def train():
    X, y = generate_data()

    X = torch.tensor(X).to(DEVICE)
    y = torch.tensor(y).unsqueeze(1).to(DEVICE)

    model = PersonalizationModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 60

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "personalization_model.pt")
    print("\nSaved model → personalization_model.pt")


if __name__ == "__main__":
    train()
