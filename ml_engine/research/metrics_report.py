import os
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime

from sklearn.metrics import (
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    explained_variance_score,
)
from scipy.stats import pearsonr, spearmanr, kendalltau, skew, kurtosis


class MetricsReporter:
    def __init__(self, save_dir="research/outputs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.history = []
        self.all_preds = []
        self.all_targets = []

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # =============================
    def _to_numpy(self, x):
        try:
            return x.detach().cpu().numpy()
        except:
            return np.array(x)

    def _clip(self, arr):
        return np.clip(arr, -5, 5)

    def _to_class(self, x):
        bins = np.linspace(0, 1, 4)
        return np.digitize(x, bins) - 1

    # =============================
    def compute_all(self, y_true, y_pred):

        y_true = self._clip(self._to_numpy(y_true))
        y_pred = self._clip(self._to_numpy(y_pred))

        y_true = np.nan_to_num(y_true)
        y_pred = np.nan_to_num(y_pred)

        self.all_targets.extend(y_true.tolist())
        self.all_preds.extend(y_pred.tolist())

        # REGRESSION
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # CORRELATION
        pearson = pearsonr(y_true, y_pred)[0]
        spearman = spearmanr(y_true, y_pred)[0]
        kendall = kendalltau(y_true, y_pred)[0]

        # CLASSIFICATION
        y_true_cls = self._to_class(y_true)
        y_pred_cls = self._to_class(y_pred)

        acc = accuracy_score(y_true_cls, y_pred_cls)
        precision = precision_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "pearson": float(pearson),
            "spearman": float(spearman),
            "kendall": float(kendall),
            "accuracy": float(acc),
            "f1": float(f1),
        }

    # =============================
    def log_epoch(self, epoch, train_loss, val_loss, y_true, y_pred):
        metrics = self.compute_all(y_true, y_pred)

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **metrics,
        }

        self.history.append(entry)

    # =============================
    def finalize(self):
        if not self.all_preds:
            print("No predictions found")
            return

        y_true = np.array(self.all_targets)
        y_pred = np.array(self.all_preds)

        self._save_metrics_table(y_true, y_pred)
        self._plot_prediction_vs_actual(y_true, y_pred)
        self._plot_residuals(y_true, y_pred)
        self._plot_error_histogram(y_true, y_pred)
        self._plot_calibration(y_true, y_pred)
        self._plot_ranking(y_true, y_pred)
        self._plot_training_curve()

        print("✅ All report artifacts generated")

    # =============================
    # TABLE
    # =============================
    def _save_metrics_table(self, y_true, y_pred):
        m = self.compute_all(y_true, y_pred)
        path = os.path.join(self.save_dir, "metrics_table.csv")

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            for k, v in m.items():
                writer.writerow([k, v])

    # =============================
    # PLOTS
    # =============================
    def _plot_prediction_vs_actual(self, y_true, y_pred):
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([0, 1], [0, 1])
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediction vs Actual")
        plt.savefig(os.path.join(self.save_dir, "prediction_vs_actual.png"))
        plt.close()

    def _plot_residuals(self, y_true, y_pred):
        residuals = y_pred - y_true

        plt.figure()
        plt.scatter(y_pred, residuals, alpha=0.3)
        plt.axhline(0)
        plt.title("Residual Plot")
        plt.savefig(os.path.join(self.save_dir, "residual_plot.png"))
        plt.close()

    def _plot_error_histogram(self, y_true, y_pred):
        errors = np.abs(y_true - y_pred)

        plt.figure()
        plt.hist(errors, bins=50)
        plt.title("Error Distribution")
        plt.savefig(os.path.join(self.save_dir, "error_histogram.png"))
        plt.close()

    def _plot_calibration(self, y_true, y_pred):
        bins = np.linspace(0, 1, 10)
        digitized = np.digitize(y_pred, bins)

        pred_means = []
        true_means = []

        for i in range(1, len(bins)):
            mask = digitized == i
            if np.sum(mask) > 0:
                pred_means.append(np.mean(y_pred[mask]))
                true_means.append(np.mean(y_true[mask]))

        plt.figure()
        plt.plot(pred_means, true_means, marker="o")
        plt.title("Calibration Curve")
        plt.savefig(os.path.join(self.save_dir, "calibration_curve.png"))
        plt.close()

    def _plot_ranking(self, y_true, y_pred):
        plt.figure()
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.title("Ranking Scatter")
        plt.savefig(os.path.join(self.save_dir, "ranking_scatter.png"))
        plt.close()

    def _plot_training_curve(self):
        if not self.history:
            return

        epochs = [x["epoch"] for x in self.history]

        plt.figure()
        plt.plot(epochs, [x["train_loss"] for x in self.history], label="Train")
        plt.plot(epochs, [x["val_loss"] for x in self.history], label="Val")
        plt.legend()
        plt.title("Training Curve")
        plt.savefig(os.path.join(self.save_dir, "training_curve.png"))
        plt.close()