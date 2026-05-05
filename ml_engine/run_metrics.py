import torch
import numpy as np

from sap_gru_model import SAP_GRU
from sequence_loader import load_sequences, to_tensor
from research.metrics_report import MetricsReporter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "data/processed/longitudinal_sequences.csv"
MODEL_PATH = "sap_gru.pt"


def main():
    print("Loading sequences...")

    sequences, targets, lengths, weights = load_sequences(DATA_PATH)

    data = to_tensor(sequences, targets, lengths, weights, DEVICE)

    X = data["X"]
    y = data["y"]
    l = data["lengths"]

    print(f"Loaded: {len(X)} samples")

    print("Loading model...")

    model = SAP_GRU(input_dim=X.shape[-1], device=DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print("Running inference...")

    preds = []

    with torch.no_grad():
        for i in range(0, len(X), 128):
            xb = X[i:i+128]
            lb = l[i:i+128]

            out = model(xb, lb)["severity"]
            preds.append(out.cpu().numpy())

    y_pred = np.concatenate(preds)
    y_true = y.cpu().numpy()

    print("Generating reports...")

    reporter = MetricsReporter()

    # Fake one epoch just to trigger system
    reporter.log_epoch(
        epoch=0,
        train_loss=0.0,
        val_loss=0.0,
        y_true=y_true,
        y_pred=y_pred
    )

    reporter.finalize()

    print("\n✅ DONE — Check: ml_engine/research/outputs/")


if __name__ == "__main__":
    main()