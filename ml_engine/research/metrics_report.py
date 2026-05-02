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

    # =========================================================
    def _to_numpy(self, x):
        try:
            return x.detach().cpu().numpy()
        except Exception:
            return np.array(x)

    def _clip(self, arr):
        return np.clip(arr, -5, 5)

    # =========================================================
    # 🔥 FIXED CLASSIFICATION (stable bins)
    # =========================================================
    def _to_class(self, x):
        bins = np.linspace(0, 1, 4)  # fixed bins
        return np.digitize(x, bins) - 1

    # =========================================================
    def compute_all(self, y_true, y_pred):

        y_true = self._clip(self._to_numpy(y_true))
        y_pred = self._clip(self._to_numpy(y_pred))

        y_true = np.nan_to_num(y_true)
        y_pred = np.nan_to_num(y_pred)

        self.all_targets.extend(y_true.tolist())
        self.all_preds.extend(y_pred.tolist())

        std = np.std(y_true)
        unique = len(np.unique(y_true))

        # =========================
        # Regression
        # =========================
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)

        medae = np.median(np.abs(y_true - y_pred))

        r2 = 0.0 if std < 1e-6 else r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)

        # Normalized RMSE
        nrmse = rmse / (np.max(y_true) - np.min(y_true) + 1e-8)

        # Safe MAPE
        denom = np.maximum(np.abs(y_true), 1e-3)
        mape = np.mean(np.abs((y_true - y_pred) / denom))

        # =========================
        # Correlation
        # =========================
        try:
            pearson = pearsonr(y_true, y_pred)[0]
        except:
            pearson = 0.0

        try:
            spearman = spearmanr(y_true, y_pred)[0]
        except:
            spearman = 0.0

        try:
            kendall = kendalltau(y_true, y_pred)[0]
        except:
            kendall = 0.0

        # =========================
        # Distribution Analysis
        # =========================
        errors = y_true - y_pred
        err_skew = skew(errors)
        err_kurt = kurtosis(errors)

        # =========================
        # Classification
        # =========================
        y_true_cls = self._to_class(y_true)
        y_pred_cls = self._to_class(y_pred)

        acc = accuracy_score(y_true_cls, y_pred_cls)
        precision = precision_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
        recall = recall_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)
        f1 = f1_score(y_true_cls, y_pred_cls, average="weighted", zero_division=0)

        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "medae": float(medae),
            "r2": float(r2),
            "explained_var": float(evs),
            "nrmse": float(nrmse),
            "mape": float(mape),
            "pearson": float(pearson),
            "spearman": float(spearman),
            "kendall_tau": float(kendall),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "error_skew": float(err_skew),
            "error_kurtosis": float(err_kurt),
            "target_std": float(std),
            "target_unique": int(unique),
        }

    # =========================================================
    def log_epoch(self, epoch, train_loss, val_loss, y_true, y_pred):
        metrics = self.compute_all(y_true, y_pred)

        entry = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            **metrics,
        }

        self.history.append(entry)

    # =========================================================
    def print_latest(self):
        m = self.history[-1]

        print(
            f"Epoch {m['epoch']:02d} | "
            f"Train: {m['train_loss']:.4f} | "
            f"Val: {m['val_loss']:.4f} | "
            f"MAE: {m['mae']:.4f} | RMSE: {m['rmse']:.4f} | "
            f"R2: {m['r2']:.4f} | Corr: {m['pearson']:.4f} | "
            f"Kendall: {m['kendall_tau']:.4f} | "
            f"Acc: {m['accuracy']:.4f} | F1: {m['f1']:.4f}"
        )

    # =========================================================
    def finalize(self):
        try:
            self._save_json()
            self._save_csv()
            self._plot_losses()
            self._plot_metrics()
            self._scatter_plot()
            self._residual_plot()
            self._save_confidence_interval()
        except Exception as e:
            print(f"[MetricsReporter] finalize error: {e}")

    # =========================================================
    def _save_json(self):
        path = os.path.join(self.save_dir, f"metrics_{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=4)

    def _save_csv(self):
        path = os.path.join(self.save_dir, f"metrics_{self.run_id}.csv")

        keys = self.history[0].keys()

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def _plot_losses(self):
        epochs = [x["epoch"] for x in self.history]

        plt.figure()
        plt.plot(epochs, [x["train_loss"] for x in self.history])
        plt.plot(epochs, [x["val_loss"] for x in self.history])
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()

    def _plot_metrics(self):
        epochs = [x["epoch"] for x in self.history]

        plt.figure()
        plt.plot(epochs, [x["mae"] for x in self.history])
        plt.plot(epochs, [x["rmse"] for x in self.history])
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "error_metrics.png"))
        plt.close()

    def _scatter_plot(self):
        if not self.all_preds:
            return

        plt.figure()
        plt.scatter(self.all_targets, self.all_preds, alpha=0.4)
        plt.plot([0, 1], [0, 1])
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "scatter.png"))
        plt.close()

    def _residual_plot(self):
        if not self.all_preds:
            return

        residuals = np.array(self.all_targets) - np.array(self.all_preds)

        plt.figure()
        plt.scatter(self.all_preds, residuals, alpha=0.4)
        plt.axhline(0)
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, "residuals.png"))
        plt.close()

    def _save_confidence_interval(self):
        if not self.all_preds:
            return

        errors = np.abs(np.array(self.all_targets) - np.array(self.all_preds))
        mean_error = np.mean(errors)
        std_error = np.std(errors)

        ci = {
            "mae_mean": float(mean_error),
            "mae_ci_95": float(1.96 * std_error),
        }

        path = os.path.join(self.save_dir, f"ci_{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(ci, f, indent=4)

        print(f"[CI] MAE: {mean_error:.4f} ± {1.96 * std_error:.4f}")