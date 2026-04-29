# train_v2_real.py

import torch
import torch.nn as nn
import numpy as np
import csv

from ml_engine.personalization_model import PersonalizationModel

DEVICE = torch.device("cpu")


def load_data():
    X, y = [], []

    with open("real_dataset.csv") as f:
        reader = csv.DictReader(f)

        for row in reader:
            X.append([
                float(row["base"]),
                float(row["mean_error"]),
                float(row["trend"]),
                float(row["variability"])
            ])
            y.append(float(row["target"]))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train():
    X, y = load_data()

    X = torch.tensor(X).to(DEVICE)
    y = torch.tensor(y).unsqueeze(1).to(DEVICE)

    model = PersonalizationModel().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(60):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), "personalization_model.pt")
    print("Saved updated model")


if __name__ == "__main__":
    train()
