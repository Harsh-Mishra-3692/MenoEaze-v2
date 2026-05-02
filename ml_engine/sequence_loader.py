import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Iterator
import logging

logger = logging.getLogger(__name__)


# =========================================================
# CONFIG
# =========================================================
REQUIRED_COLUMNS = {"swanid", "visit", "target", "sequence_id"}
EXCLUDE_COLUMNS = REQUIRED_COLUMNS.union({"is_synthetic"})

MAX_SEQ_LEN = 10
MIN_SEQ_LEN = 3
EPS = 1e-8


# =========================================================
# VALIDATION
# =========================================================
def _validate_df(df: pd.DataFrame):
    if df is None or df.empty:
        raise ValueError("Empty dataframe")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


# =========================================================
# SANITIZE
# =========================================================
def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(["", " ", "nan", "NaN"], np.nan)

    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df["target"] = df["target"].fillna(0.0)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


# =========================================================
# LOAD SEQUENCES (CORRECTED)
# =========================================================
def load_sequences(
    path: str,
) -> Tuple[List[np.ndarray], List[float], List[int], List[float]]:

    df = pd.read_csv(path)

    _validate_df(df)
    df = _sanitize(df)

    df = df.sort_values(["sequence_id", "visit"]).reset_index(drop=True)

    feature_cols = sorted([c for c in df.columns if c not in EXCLUDE_COLUMNS])

    if not feature_cols:
        raise ValueError("No valid feature columns found")

    sequences, targets, lengths, weights = [], [], [], []

    grouped = df.groupby("sequence_id", sort=False)

    dropped_short = 0
    dropped_invalid = 0

    for seq_id, group in grouped:
        try:
            group = group.sort_values("visit")

            if len(group) < MIN_SEQ_LEN:
                dropped_short += 1
                continue

            features = group[feature_cols].to_numpy(dtype=np.float32)
            target_arr = group["target"].to_numpy(dtype=np.float32)

            # =========================================================
            # ✅ CORRECT: USE FULL SEQUENCE
            # =========================================================
            seq = features
            target_val = float(target_arr[-1])

            # =========================================================
            # VALIDATION
            # =========================================================
            if len(seq) < MIN_SEQ_LEN:
                dropped_short += 1
                continue

            if not np.isfinite(seq).all() or not np.isfinite(target_val):
                dropped_invalid += 1
                continue

            # =========================================================
            # WEIGHT
            # =========================================================
            is_synth = int(group.get("is_synthetic", pd.Series([0])).max())
            sample_weight = 0.5 if is_synth else 1.0

            # =========================================================
            # TRUNCATION
            # =========================================================
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[-MAX_SEQ_LEN:]

            sequences.append(seq)
            targets.append(target_val)
            lengths.append(len(seq))
            weights.append(sample_weight)

        except Exception as e:
            dropped_invalid += 1
            logger.warning(f"[sequence_loader] seq_id={seq_id} failed: {e}")
            continue

    if not sequences:
        raise RuntimeError("No valid sequences generated")

    # =========================================================
    # TARGET DIAGNOSTICS
    # =========================================================
    targets_arr = np.array(targets)

    logger.info(
        f"[sequence_loader] loaded={len(sequences)} | "
        f"dropped_short={dropped_short} | dropped_invalid={dropped_invalid}"
    )

    logger.info(
        f"[TARGET STATS] mean={targets_arr.mean():.4f} | "
        f"std={targets_arr.std():.4f} | "
        f"min={targets_arr.min():.4f} | "
        f"max={targets_arr.max():.4f} | "
        f"unique={len(np.unique(targets_arr))}"
    )

    if targets_arr.std() < 1e-5:
        logger.warning("🚨 Target variance too low → model will fail")

    return sequences, targets, lengths, weights


# =========================================================
# TENSOR CONVERSION
# =========================================================
def to_tensor(
    sequences: List[np.ndarray],
    targets: List[float],
    lengths: List[int],
    weights: List[float],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:

    if not sequences:
        raise ValueError("Empty sequences")

    if not (len(sequences) == len(targets) == len(lengths) == len(weights)):
        raise ValueError("Mismatch in data lengths")

    device = device or torch.device("cpu")

    batch = len(sequences)
    feature_dim = sequences[0].shape[1]

    max_len = min(max(lengths), MAX_SEQ_LEN)

    X = torch.zeros((batch, max_len, feature_dim), dtype=torch.float32)
    mask = torch.zeros((batch, max_len), dtype=torch.float32)

    y = torch.tensor(targets, dtype=torch.float32)
    w = torch.tensor(weights, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    for i, seq in enumerate(sequences):
        l = min(seq.shape[0], max_len)
        if l <= 0:
            continue

        X[i, :l] = torch.from_numpy(seq[:l])
        mask[i, :l] = 1.0

    X = torch.nan_to_num(X)
    y = torch.nan_to_num(y)
    w = torch.clamp(w, 0.1, 1.0)

    return {
        "X": X.to(device),
        "y": y.to(device),
        "mask": mask.to(device),
        "weights": w.to(device),
        "lengths": lengths_tensor.to(device),
    }


# =========================================================
# BATCH GENERATOR
# =========================================================
def batch_generator(
    data_dict: Dict[str, torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

    X = data_dict["X"]
    y = data_dict["y"]
    lengths = data_dict["lengths"]

    n = X.shape[0]

    indices = torch.arange(n)

    if shuffle:
        indices = indices[torch.randperm(n)]

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]

        try:
            Xb = X[batch_idx]
            yb = y[batch_idx]
            lb = lengths[batch_idx]

            if torch.isnan(Xb).any() or torch.isinf(Xb).any():
                continue

            if torch.isnan(yb).any() or torch.isinf(yb).any():
                continue

            yield Xb, yb, lb

        except Exception as e:
            logger.warning(f"[batch_generator] skipped batch: {e}")
            continue