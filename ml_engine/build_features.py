import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json
import logging
from datetime import datetime, UTC

from feature_schema import FeatureSchema


# =========================================================
# CONFIG
# =========================================================
INPUT_PATH = Path("data/interim/unified_longitudinal.csv")
OUTPUT_PATH = Path("data/processed/features.csv")
AUDIT_PATH = Path("data/processed/features_audit.json")

EPS = 1e-8
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

schema = FeatureSchema()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================================
# LOAD
# =========================================================
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    logger.info(f"✔ Loaded: {df.shape}")
    return df


# =========================================================
# VISIT NORMALIZATION
# =========================================================
def unify_visit(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "visit" not in df.columns:
        df["visit"] = df.groupby("swanid").cumcount() + 1

    df["visit"] = pd.to_numeric(df["visit"], errors="coerce")
    df = df.dropna(subset=["swanid", "visit"])

    df = df.sort_values(["swanid", "visit"], kind="stable")
    return df


# =========================================================
# FEATURE SELECTION
# =========================================================
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["swanid", "visit"] + schema.get_model_input_order()

    missing = [c for c in cols if c not in df.columns]

    if missing:
        logger.warning(f"Missing columns added: {missing}")
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing)],
            axis=1
        )

    return df[cols].copy()


# =========================================================
# TRANSFORMS
# =========================================================
def apply_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = schema.transform(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# =========================================================
# MISSING HANDLING
# =========================================================
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    features = schema.get_model_input_order()

    grouped = df.groupby("swanid")

    for col in features:
        df[f"{col}_missing"] = df[col].isna().astype("int8")

        df[col] = grouped[col].transform(
            lambda x: x.fillna(x.median() if x.notna().any() else 0.0)
        )

    return df


# =========================================================
# 🔥 NEW: GLOBAL FEATURE SIGNALS (IMPORTANT)
# =========================================================
def add_global_features(df: pd.DataFrame) -> pd.DataFrame:

    # global percentile ranking (cross-user signal)
    if "severity" in df.columns:
        df["severity_global_rank"] = df["severity"].rank(pct=True)

    # symptom aggregation (strong signal)
    symptom_cols = [c for c in df.columns if any(k in c for k in ["hotflash", "nisweat", "sleep", "depress", "irritab", "tense"])]

    if symptom_cols:
        df["symptom_load"] = df[symptom_cols].mean(axis=1)

        df["symptom_rank"] = df["symptom_load"].rank(pct=True)

    return df


# =========================================================
# TEMPORAL FEATURES
# =========================================================
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["swanid", "visit"])
    features = schema.get_model_input_order()

    for col in features:

        delta = df.groupby("swanid")[col].diff().fillna(0)

        std = delta.std()
        if std > EPS:
            delta = delta / std

        df[f"{col}_delta"] = delta.clip(-5, 5)

        roll = (
            df.groupby("swanid")[col]
            .rolling(2, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{col}_roll"] = roll.clip(-5, 5)

    return df


# =========================================================
# 🔥 IMPROVED SEVERITY
# =========================================================
def compute_severity(df: pd.DataFrame) -> pd.Series:

    weights = {
        "hotflash": 0.25,
        "nisweat": 0.20,
        "diffislp": 0.15,
        "depress": 0.10,
        "tense": 0.10,
        "irritab": 0.08,
        "forget": 0.05,
        "headach": 0.03,
        "stiff": 0.02,
    }

    available = [c for c in weights if c in df.columns]

    if not available:
        return pd.Series(np.zeros(len(df)), index=df.index)

    X = df[available].values.astype("float32")
    w = np.array([weights[c] for c in available], dtype="float32")

    base = (X @ w) / (w.sum() + EPS)

    # temporal delta signal
    delta_cols = [f"{c}_delta" for c in available if f"{c}_delta" in df.columns]
    if delta_cols:
        base += 0.3 * df[delta_cols].mean(axis=1)

    # trend
    trend = (
        df.groupby("swanid")[available]
        .diff()
        .fillna(0)
        .mean(axis=1)
    )

    base += 0.2 * trend

    # 🔥 global normalization (critical change)
    min_v, max_v = base.min(), base.max()
    if max_v - min_v > EPS:
        severity = (base - min_v) / (max_v - min_v)
    else:
        severity = np.zeros_like(base)

    # smooth non-linearity
    severity = np.tanh(2 * (severity - 0.5)) * 0.5 + 0.5

    severity += rng.normal(0, 1e-3, size=len(severity))

    return pd.Series(np.clip(severity, 0.0, 1.0), index=df.index)


# =========================================================
# VALIDATION
# =========================================================
def validate(df: pd.DataFrame):
    if df["swanid"].isna().any():
        raise ValueError("Null swanid")

    if df["visit"].isna().any():
        raise ValueError("Null visit")

    if not np.isfinite(df.select_dtypes(include=[np.number]).values).all():
        raise ValueError("Non-finite values detected")


# =========================================================
def add_time_index(df: pd.DataFrame) -> pd.DataFrame:
    df["time_idx"] = df.groupby("swanid").cumcount()
    return df


# =========================================================
def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.fillna(0.0)

    ordered_cols = ["swanid", "visit"] + sorted(
        [c for c in df.columns if c not in ["swanid", "visit"]]
    )

    return df[ordered_cols]


# =========================================================
def build_audit(df: pd.DataFrame) -> Dict:

    visits = df.groupby("swanid").size()

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "rows": int(df.shape[0]),
        "users": int(df["swanid"].nunique()),
        "avg_visits": float(visits.mean()),
        "max_visits": int(visits.max()),
        "severity_std": float(df["severity"].std()),
    }


# =========================================================
def build_features() -> Tuple[pd.DataFrame, Path]:

    df = load_data(INPUT_PATH)

    df = unify_visit(df)
    df = select_features(df)

    df = apply_transforms(df)
    df = handle_missing(df)

    df = add_temporal_features(df)

    df["severity"] = compute_severity(df)

    df = add_global_features(df)   # 🔥 new

    validate(df)

    df = add_time_index(df)
    df = finalize(df)

    audit = build_audit(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)

    with open(AUDIT_PATH, "w") as f:
        json.dump(audit, f, indent=2)

    logger.info("=== FEATURE REPORT ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Users: {df['swanid'].nunique()}")
    logger.info(f"Avg visits: {df.groupby('swanid').size().mean():.2f}")
    logger.info(f"Severity std: {df['severity'].std():.4f}")

    return df, OUTPUT_PATH


if __name__ == "__main__":
    build_features()