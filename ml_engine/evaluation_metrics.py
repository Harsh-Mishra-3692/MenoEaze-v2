# evaluation_metrics.py

import numpy as np


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def calibration_error(y_true, y_pred, bins=10):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bin_edges = np.linspace(0, 1, bins + 1)
    error = 0

    for i in range(bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
        if np.sum(mask) > 0:
            avg_pred = np.mean(y_pred[mask])
            avg_true = np.mean(y_true[mask])
            error += abs(avg_pred - avg_true)

    return error / bins


def personalization_gain(base_preds, adapted_preds, actuals):
    base_error = mae(actuals, base_preds)
    adapted_error = mae(actuals, adapted_preds)
    return base_error - adapted_error
