# build_features.py — FINAL (DUAL MODE: OFFLINE + ONLINE SAFE)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Any
import json
import logging
from datetime import datetime, UTC

from ml_engine.feature_schema import FeatureSchema

logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = Path("data/interim/unified_longitudinal.csv")
OUTPUT_PATH = Path("data/processed/features.csv")
AUDIT_PATH = Path("data/processed/features_audit.json")

EPS = 1e-8
FEATURES = 11

schema = FeatureSchema()

# =========================================================
# ================= ONLINE MODE ============================
# =========================================================

def build_feature_vector(raw: Dict[str, Any]) -> List[float]:
    """
    🔥 CRITICAL: Used by API (/log-symptom)
    Converts raw user input → 11-dim model vector
    """

    if not isinstance(raw, dict):
        raise ValueError("Invalid input: expected dict")

    feature_order = schema.get_model_input_order()

    if len(feature_order) != FEATURES:
        raise ValueError("Schema mismatch: expected 11 features")

    vector = []

    for f in feature_order:
        v = raw.get(f, 0.0)

        try:
            v = float(v)
        except:
            v = 0.0

        # clamp to safe range
        v = max(0.0, min(10.0, v))

        vector.append(v)

    if len(vector) != FEATURES:
        raise ValueError("Feature vector length mismatch")

    return vector

# =========================================================
# ================= OFFLINE MODE ===========================
# =========================================================

def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Dataset not found — skipping offline pipeline")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        logger.warning("Dataset empty")
        return pd.DataFrame()

    logger.info(f"Loaded: {df.shape}")
    return df


def unify_visit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "visit" not in df.columns:
        df["visit"] = df.groupby("swanid").cumcount() + 1

    df["visit"] = pd.to_numeric(df["visit"], errors="coerce")
    df = df.dropna(subset=["swanid", "visit"])

    return df.sort_values(["swanid", "visit"], kind="stable")


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = schema.get_model_input_order()

    if len(features) != FEATURES:
        raise ValueError("Expected 11 features")

    cols = ["swanid", "visit"] + features
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[cols].copy()


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    features = schema.get_model_input_order()

    for col in features:
        if col not in df:
            df[col] = 0.0

        df[col] = pd.to_numeric(df[col], errors="coerce")

        df[col] = df.groupby("swanid")[col].transform(
            lambda x: x.fillna(x.median() if x.notna().any() else 0.0)
        )

    return df.fillna(0.0)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["swanid", "visit"])
    features = schema.get_model_input_order()

    for col in features:
        try:
            delta = df.groupby("swanid")[col].diff().fillna(0)
            std = delta.std()

            if std > EPS:
                delta = delta / std

            df[f"{col}_delta"] = delta.clip(-5, 5)

        except Exception:
            df[f"{col}_delta"] = 0.0

    return df


def compute_severity(df: pd.DataFrame) -> pd.Series:
    try:
        features = schema.get_model_input_order()
        X = df[features].values.astype("float32")

        severity = np.mean(X, axis=1)

        min_v, max_v = severity.min(), severity.max()

        if max_v - min_v > EPS:
            severity = (severity - min_v) / (max_v - min_v)
        else:
            severity = np.zeros_like(severity)

        return pd.Series(np.clip(severity, 0.0, 1.0), index=df.index)

    except Exception:
        return pd.Series(np.zeros(len(df)), index=df.index)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)
    return df


def build_features() -> Tuple[pd.DataFrame, Path]:
    """
    Offline pipeline (training only)
    Safe: never crashes system
    """

    df = load_data(INPUT_PATH)

    if df.empty:
        return pd.DataFrame(), OUTPUT_PATH

    try:
        df = unify_visit(df)
        df = select_features(df)
        df = handle_missing(df)
        df = add_temporal_features(df)

        df["severity"] = compute_severity(df)

        df = finalize(df)

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)

        logger.info(f"Features saved: {OUTPUT_PATH}")

        return df, OUTPUT_PATH

    except Exception as e:
        logger.error(f"Feature pipeline failed: {e}")
        return pd.DataFrame(), OUTPUT_PATH


if __name__ == "__main__":
    build_features()