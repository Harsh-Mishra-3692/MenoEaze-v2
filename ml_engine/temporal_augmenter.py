import numpy as np
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# -------------------------
# CONFIG
# -------------------------

DEFAULT_SEQ_LEN = 4
MAX_SEQ_LEN = 6

NOISE_STD = 0.02
TREND_SCALE = 0.05
DRIFT_SCALE = 0.01

MAX_SYNTHETIC_PER_USER = 1
GLOBAL_SEED = 42

EPS = 1e-8

np.random.seed(GLOBAL_SEED)


# -------------------------
# RNG (DETERMINISTIC + SAFE)
# -------------------------
def _get_rng(user_id: int) -> np.random.Generator:
    seed = (int(user_id) * 9973 + GLOBAL_SEED) % (2**32 - 1)
    return np.random.default_rng(seed)


# -------------------------
# SAFETY HELPERS
# -------------------------
def _safe_clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _sanitize_vector(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    return _safe_clip(x)


# -------------------------
# FEATURE-SENSITIVE SCALING
# -------------------------
def _compute_feature_scale(base: np.ndarray) -> np.ndarray:
    """
    Adaptive scaling:
    - low variance → less noise
    - high variance → more flexibility
    """
    std = np.std(base) + EPS
    scale = np.clip(std, 0.05, 0.5)
    return scale


# -------------------------
# TEMPORAL GENERATOR (STABLE)
# -------------------------
def _generate_temporal_dynamics(
    base: np.ndarray,
    seq_len: int,
    rng: np.random.Generator
) -> np.ndarray:

    dim = base.shape[0]
    sequence = np.zeros((seq_len, dim), dtype=np.float32)

    current = base.copy()
    feature_scale = _compute_feature_scale(base)

    # normalized direction
    direction = rng.uniform(-1, 1, size=dim)
    direction /= (np.linalg.norm(direction) + EPS)

    for t in range(seq_len):

        progress = t / max(seq_len - 1, 1)

        drift = rng.normal(0, DRIFT_SCALE * feature_scale, size=dim)
        trend = direction * TREND_SCALE * progress
        noise = rng.normal(0, NOISE_STD * feature_scale, size=dim)

        # bounded update (prevents explosion)
        delta = drift + trend + noise
        delta = np.clip(delta, -0.15, 0.15)

        current = current + delta
        current = _sanitize_vector(current)

        sequence[t] = current

    return sequence


# -------------------------
# PUBLIC API
# -------------------------
def generate_synthetic_sequence(
    row: pd.Series,
    feature_cols: List[str],
    user_id: int,
    seq_len: Optional[int] = None
) -> pd.DataFrame:

    if row is None or len(feature_cols) == 0:
        raise ValueError("Invalid input row or feature columns")

    rng = _get_rng(user_id)

    if seq_len is None:
        seq_len = int(rng.integers(DEFAULT_SEQ_LEN, MAX_SEQ_LEN + 1))

    base = row[feature_cols].values.astype(np.float32)

    if not np.isfinite(base).all():
        base = np.nan_to_num(base, nan=0.0)

    base = _safe_clip(base)

    seq_array = _generate_temporal_dynamics(base, seq_len, rng)

    seq_df = pd.DataFrame(seq_array, columns=feature_cols)

    seq_df["visit"] = np.arange(1, seq_len + 1, dtype=np.int32)

    return seq_df


# -------------------------
# AUGMENT DATAFRAME (SAFE + FAST)
# -------------------------
def augment_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    min_seq_len: int = 2
) -> pd.DataFrame:

    if df is None or df.empty:
        raise ValueError("Input dataframe is empty")

    required_cols = {"swanid", "visit"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Missing required columns: swanid / visit")

    # ensure numeric consistency
    df = df.copy()
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0)

    augmented = []

    grouped = df.groupby("swanid", sort=False)

    for user_id, group in grouped:

        try:
            group = group.sort_values("visit")

            # -------------------------
            # MULTI-VISIT → KEEP
            # -------------------------
            if len(group) >= min_seq_len:
                g = group.copy()
                g["is_synthetic"] = 0
                augmented.append(g)
                continue

            # -------------------------
            # SINGLE VISIT → AUGMENT
            # -------------------------
            base_row = group.iloc[0]

            synth_seq = generate_synthetic_sequence(
                base_row,
                feature_cols,
                user_id=int(user_id)
            )

            synth_seq["swanid"] = user_id
            synth_seq["is_synthetic"] = 1

            augmented.append(synth_seq)

        except Exception as e:
            logger.warning(f"[augment] user={user_id} failed: {e}")
            continue

    if not augmented:
        raise RuntimeError("Augmentation produced no valid data")

    result = pd.concat(augmented, ignore_index=True)

    validate_augmented(result, feature_cols)

    return result


# -------------------------
# VALIDATION
# -------------------------
def validate_augmented(df: pd.DataFrame, feature_cols: List[str]):

    if df.empty:
        raise ValueError("Augmented dataframe is empty")

    if not np.isfinite(df[feature_cols].values).all():
        raise ValueError("Non-finite values detected")

    if df[feature_cols].isna().any().any():
        raise ValueError("NaN values detected")

    if not {"visit", "swanid"}.issubset(df.columns):
        raise ValueError("Missing required columns after augmentation")

    # sanity distribution check
    if df[feature_cols].std().mean() < 1e-5:
        logger.warning("⚠️ Low variance detected in augmented data")