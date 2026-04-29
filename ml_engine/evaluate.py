"""
evaluate.py
===========
Research-grade evaluation for GRU model.

Outputs:
- results.txt (detailed metrics)
- prediction_plot.png
- pred_vs_actual.png
- residuals.png
"""

import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from ml_engine.model_def import GRUModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

logger = logging.getLogger("menoeaze.evaluate")
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot + 1e-8)

    # Bias
    bias = np.mean(y_pred - y_true)

    # Variance of errors
    var = np.var(y_pred - y_true)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "bias": float(bias),
        "variance": float(var),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    set_seed()

    print("━" * 60)
    print("  GRU Model Evaluation (Research-Grade)")
    print("━" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Load data ─────────────────────────
    data = torch.load(os.path.join(base_dir, "preprocessed_data.pt"), weights_only=False)
    X_test = data["X_test"].to(DEVICE)
    y_test = data["y_test"].to(DEVICE)

    # ── Load model ────────────────────────
    ckpt = torch.load(os.path.join(base_dir, "model.pt"), weights_only=False)

    model = GRUModel(input_size=ckpt["input_size"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── Predict ───────────────────────────
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    y_actual = y_test.cpu().numpy()

    # ── Metrics ───────────────────────────
    metrics = compute_metrics(y_actual, y_pred)

    print("\nTest samples :", len(y_actual))
    for k, v in metrics.items():
        print(f"{k.upper():<10}: {v:.6f}")

    # ── Residuals ─────────────────────────
    residuals = y_pred - y_actual

    # ── Save results.txt ──────────────────
    results_path = os.path.join(base_dir, "results.txt")

    with open(results_path, "w") as f:
        f.write("GRU MODEL EVALUATION\n")
        f.write("=" * 50 + "\n\n")

        for k, v in metrics.items():
            f.write(f"{k.upper():<10}: {v:.6f}\n")

        f.write("\nError Analysis\n")
        f.write("-" * 50 + "\n")
        f.write(f"Residual Mean : {np.mean(residuals):.6f}\n")
        f.write(f"Residual Std  : {np.std(residuals):.6f}\n")
        f.write(f"Max Error     : {np.max(np.abs(residuals)):.6f}\n")

    logger.info(f"Saved results → {results_path}")

    # ── Plots ─────────────────────────────

    # 1. Scatter
    plt.figure()
    plt.scatter(y_actual, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Actual")
    plt.savefig(os.path.join(base_dir, "pred_vs_actual.png"))
    plt.close()

    # 2. Residuals
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.title("Residual Distribution")
    plt.savefig(os.path.join(base_dir, "residuals.png"))
    plt.close()

    # 3. Line plot
    n = min(50, len(y_actual))
    plt.figure()
    plt.plot(y_actual[:n], label="Actual")
    plt.plot(y_pred[:n], label="Predicted")
    plt.legend()
    plt.title("Pred vs Actual (first 50)")
    plt.savefig(os.path.join(base_dir, "prediction_plot.png"))
    plt.close()

    logger.info("Saved plots:")
    logger.info(" - pred_vs_actual.png")
    logger.info(" - residuals.png")
    logger.info(" - prediction_plot.png")

    print("━" * 60)


if __name__ == "__main__":
    main()