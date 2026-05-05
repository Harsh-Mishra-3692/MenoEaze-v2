import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, UTC

from ml_engine.feature_schema import FeatureSchema


# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = Path("data/processed/features.csv")

OUTPUT_SEQ_PATH = Path("data/processed/longitudinal_sequences.csv")
OUTPUT_FALLBACK_PATH = Path("data/processed/non_sequential_users.csv")
AUDIT_PATH = Path("data/processed/sequence_audit.json")

MIN_SEQUENCE_LENGTH = 3
MAX_SEQUENCE_LENGTH = 10
MAX_SEQUENCES_PER_USER = 6   # ↓ reduce synthetic amplification
MAX_MISSING_RATIO = 0.4
EPS = 1e-8

SYNTHETIC_PROB = 0.3         # 🔥 critical: limit synthetic usage

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

schema = FeatureSchema()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError("Empty dataset")

    logger.info(f"✔ Loaded features: {df.shape}")
    return df


# =========================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["swanid"] = pd.to_numeric(df["swanid"], errors="coerce")
    df["visit"] = pd.to_numeric(df["visit"], errors="coerce")
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce")

    df = df.dropna(subset=["swanid", "visit", "severity"])
    df = df.drop_duplicates(["swanid", "visit"])
    df = df.sort_values(["swanid", "visit"])

    return df


# =========================================================
def is_valid_np(arr: np.ndarray) -> bool:
    if arr.shape[0] < MIN_SEQUENCE_LENGTH:
        return False
    if not np.isfinite(arr).all():
        return False
    if np.isnan(arr).mean() > MAX_MISSING_RATIO:
        return False
    return True


# =========================================================
# 🔥 IMPROVED SYNTHETIC GENERATOR (LESS DOMINANT)
# =========================================================
def generate_synthetic_sequence(row, feature_cols):

    base_x = row[feature_cols].values.astype(np.float32)
    base_sev = float(row["severity"])

    seq_len = rng.integers(4, 7)

    X, y, visits = [], [], []

    # reduced bias
    severity = np.clip(base_sev + rng.normal(0, 0.05), 0, 1)

    for t in range(seq_len):

        noise = rng.normal(0, 0.04, size=len(base_x))

        # smoother trend (less artificial)
        trend = 0.05 * np.sin(t / seq_len * np.pi)

        xt = np.clip(base_x + noise + trend, -3, 3)

        # mild progression only
        severity += 0.03 * np.tanh(t / seq_len)
        severity += rng.normal(0, 0.02)

        severity = np.clip(severity, 0.0, 1.0)

        X.append(xt)
        y.append(severity)
        visits.append(t + 1)

    df = pd.DataFrame(X, columns=feature_cols)
    df["severity"] = y
    df["visit"] = visits

    return df


# =========================================================
def build_sequences(df: pd.DataFrame):

    feature_cols = schema.get_model_input_order()

    sequences = []
    metadata = []
    fallback = []

    grouped = df.groupby("swanid", sort=False)

    for user_id, group in grouped:

        try:
            g = group.sort_values("visit")

            X = g[feature_cols].to_numpy(dtype=np.float32)
            y = g["severity"].to_numpy(dtype=np.float32)
            visits = g["visit"].to_numpy(dtype=np.float32)

            n = len(X)

            # -------------------------------------------------
            # 🔥 CONTROLLED SYNTHETIC USAGE
            # -------------------------------------------------
            if n < MIN_SEQUENCE_LENGTH:

                if rng.random() < SYNTHETIC_PROB:
                    try:
                        synth = generate_synthetic_sequence(g.iloc[0], feature_cols)

                        X = synth[feature_cols].to_numpy(dtype=np.float32)
                        y = synth["severity"].to_numpy(dtype=np.float32)
                        visits = synth["visit"].to_numpy(dtype=np.float32)
                        n = len(X)
                    except Exception:
                        fallback.append(g)
                        continue
                else:
                    fallback.append(g)
                    continue

            count = 0

            for start in range(n - 1):

                if count >= MAX_SEQUENCES_PER_USER:
                    break

                end = min(start + MAX_SEQUENCE_LENGTH, n)

                seq_X = X[start:end - 1]
                seq_y = y[start + 1:end]

                if len(seq_X) < MIN_SEQUENCE_LENGTH:
                    continue

                if not is_valid_np(seq_X):
                    continue

                seq_len = len(seq_X)

                # -------------------------------------------------
                # 🔥 BETTER TEMPORAL FEATURES
                # -------------------------------------------------
                time_idx = np.linspace(0, 1, seq_len)

                gap = np.diff(visits[start:end], prepend=visits[start])
                std = np.std(gap)

                visit_gap = (
                    np.zeros(seq_len)
                    if std < EPS
                    else (gap - gap.mean()) / (std + EPS)
                )

                seq_df = pd.DataFrame(seq_X, columns=feature_cols)

                seq_df["target"] = seq_y
                seq_df["swanid"] = user_id
                seq_df["visit"] = visits[start:end - 1]

                seq_df["time_idx"] = time_idx
                seq_df["visit_gap"] = visit_gap[:-1]

                seq_id = f"{int(user_id)}_{start}"

                seq_df["sequence_id"] = seq_id
                seq_df["sequence_length"] = seq_len

                sequences.append(seq_df)

                metadata.append({
                    "sequence_id": seq_id,
                    "user": int(user_id),
                    "length": seq_len,
                    "synthetic": int(n < MIN_SEQUENCE_LENGTH)
                })

                count += 1

        except Exception as e:
            logger.warning(f"[sequence_builder] user={user_id} failed: {e}")
            continue

    if not sequences:
        raise RuntimeError("No valid sequences generated")

    return (
        pd.concat(sequences, ignore_index=True),
        pd.concat(fallback, ignore_index=True) if fallback else pd.DataFrame(),
        pd.DataFrame(metadata)
    )


# =========================================================
def build_audit(seq_df, fallback_df, metadata):

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "total_sequences": int(len(metadata)),
        "avg_length": float(metadata["length"].mean()),
        "fallback_users": int(fallback_df["swanid"].nunique()) if not fallback_df.empty else 0,
        "synthetic_ratio": float(metadata["synthetic"].mean())
    }


# =========================================================
def main():

    df = load_data(INPUT_PATH)
    df = preprocess(df)

    seq_df, fallback_df, metadata = build_sequences(df)

    audit = build_audit(seq_df, fallback_df, metadata)

    seq_df.to_csv(OUTPUT_SEQ_PATH, index=False)

    if not fallback_df.empty:
        fallback_df.to_csv(OUTPUT_FALLBACK_PATH, index=False)

    with open(AUDIT_PATH, "w") as f:
        json.dump(audit, f, indent=2)

    logger.info("=== LONGITUDINAL REPORT ===")
    logger.info(f"Sequences: {len(metadata)}")
    logger.info(f"Fallback users: {audit['fallback_users']}")
    logger.info(f"Synthetic ratio: {audit['synthetic_ratio']:.2f}")



# =========================================================
# 🔥 PRODUCTION WRAPPER (REQUIRED BY PIPELINE)
# =========================================================
def build_sequence_for_inference(features: list) -> np.ndarray:
    """
    Thin wrapper to convert feature list to (5, 11) sequence.
    LOCKED to production feature count (11).
    """
    if not isinstance(features, (list, np.ndarray)):
        return None
        
    # Standardize to 11 features
    vec = np.array(features, dtype=np.float32).flatten()
    if vec.shape[0] != 11:
        # Pad or truncate to 11
        tmp = np.zeros(11, dtype=np.float32)
        n = min(len(vec), 11)
        tmp[:n] = vec[:n]
        vec = tmp
        
    # Create dummy sequence of length 5
    seq = np.tile(vec, (5, 1))
    return seq


if __name__ == "__main__":
    main()