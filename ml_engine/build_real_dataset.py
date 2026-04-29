# build_real_dataset.py (PRODUCTION DATA PIPELINE)

import numpy as np
import csv
import logging
from typing import List, Dict
from collections import defaultdict

from ml_engine.db_client import fetch_feedback

logger = logging.getLogger("menoeaze.dataset")
logging.basicConfig(level=logging.INFO)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_HISTORY = 5
MAX_HISTORY = 30
OUTLIER_Z = 3.0
FETCH_LIMIT = 50000


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def _valid(row: Dict) -> bool:
    required = ["user_id", "predicted", "actual", "error", "created_at"]

    for k in required:
        if k not in row or row[k] is None:
            return False

    try:
        float(row["predicted"])
        float(row["actual"])
        float(row["error"])
    except:
        return False

    return True


# ─────────────────────────────────────────────
# OUTLIER REMOVAL
# ─────────────────────────────────────────────
def _remove_outliers(arr: np.ndarray) -> np.ndarray:
    if len(arr) < 5:
        return arr

    mean = np.mean(arr)
    std = np.std(arr)

    if std == 0:
        return arr

    z = np.abs((arr - mean) / std)
    return arr[z < OUTLIER_Z]


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────
def compute_features(history: List[Dict]):

    errors = np.array([h["error"] for h in history], dtype=np.float32)

    errors = _remove_outliers(errors)

    if len(errors) < 2:
        return None

    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    trend = float(errors[-1] - errors[0])
    recent = float(np.mean(errors[-3:])) if len(errors) >= 3 else float(errors[-1])
    momentum = float(errors[-1] - errors[-2]) if len(errors) >= 2 else 0.0

    return [mean_error, std_error, trend, recent, momentum]


# ─────────────────────────────────────────────
# BUILD DATASET
# ─────────────────────────────────────────────
def build_dataset():

    logger.info("Fetching feedback from Supabase...")

    raw = fetch_feedback(limit=FETCH_LIMIT)

    if not raw:
        logger.warning("No feedback data found")
        return []

    users = defaultdict(list)

    # ── CLEAN + GROUP ─────────────────────────
    for row in raw:
        if not _valid(row):
            continue

        users[row["user_id"]].append(row)

    dataset = []
    total_users = 0

    # ── BUILD PER USER ────────────────────────
    for user_id, history in users.items():

        if len(history) < MIN_HISTORY:
            continue

        total_users += 1

        # sort by created_at (correct field)
        history = sorted(history, key=lambda x: x["created_at"])

        for i in range(MIN_HISTORY, len(history)):

            start = max(0, i - MAX_HISTORY)
            sub_hist = history[start:i]

            features = compute_features(sub_hist)
            if features is None:
                continue

            base = float(history[i]["predicted"])
            target = float(history[i]["error"])

            # safety
            if np.isnan(base) or np.isnan(target):
                continue

            if any(np.isnan(f) for f in features):
                continue

            dataset.append([
                base,
                *features,
                target
            ])

    logger.info(f"Users used: {total_users}")
    logger.info(f"Dataset size: {len(dataset)} samples")

    return dataset


# ─────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────
def save_csv(dataset, path="real_dataset.csv"):

    if not dataset:
        logger.warning("Empty dataset, skipping save")
        return

    header = [
        "base",
        "mean_error",
        "std_error",
        "trend",
        "recent",
        "momentum",
        "target"
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(dataset)

    logger.info(f"Saved dataset → {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    dataset = build_dataset()
    save_csv(dataset)

    print(f"\n✅ DONE: {len(dataset)} samples generated")
    
