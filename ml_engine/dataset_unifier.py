import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import json
import logging
import hashlib
from datetime import datetime, UTC
from feature_schema import FeatureSchema


# ---------------------------
# CONFIG
# ---------------------------

INTERIM_DIR = Path("data/interim")

DATASETS = [
    "04368_clean.csv",
    "28762_clean.csv",
    "29221_clean.csv",
    "32961_clean.csv",
]

OUTPUT_PATH = Path("data/interim/unified_longitudinal.csv")
AUDIT_PATH = Path("data/interim/unification_audit.json")

MISSING_THRESHOLD = 0.95
MAX_ABS_VALUE = 1e6
CHUNK_SIZE = 8192

schema = FeatureSchema()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# HASH
# ---------------------------
def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------
# SAFE NUMERIC
# ---------------------------
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").clip(-MAX_ABS_VALUE, MAX_ABS_VALUE)


# ---------------------------
# LOAD DATASET
# ---------------------------
def load_dataset(file_name: str) -> pd.DataFrame:

    path = INTERIM_DIR / file_name

    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path, low_memory=False)

    if df.empty:
        raise ValueError(f"{file_name} is empty")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w]", "", regex=True)
    )

    if "swanid" not in df.columns:
        raise ValueError(f"{file_name}: missing swanid")

    df["swanid"] = safe_numeric(df["swanid"])
    df = df.dropna(subset=["swanid"])

    # ---------------------------
    # IMPROVED VISIT HANDLING
    # ---------------------------
    if "lvisit" in df.columns:
        visit_raw = pd.to_numeric(df["lvisit"], errors="coerce")
        if visit_raw.notna().sum() > 0:
            df["visit"] = visit_raw
        else:
            df["visit"] = df.groupby("swanid").cumcount() + 1
    else:
        df["visit"] = df.groupby("swanid").cumcount() + 1

    # features
    for col in df.columns:
        if col not in {"swanid", "visit"}:
            df[col] = safe_numeric(df[col])

    df["source_dataset"] = file_name
    df["file_hash"] = file_hash(path)

    logger.info(f"✔ Loaded {file_name} → {df.shape}")

    return df


# ---------------------------
# ALIGN
# ---------------------------
def align_all_columns(datasets: List[pd.DataFrame]) -> pd.DataFrame:

    all_cols = sorted(set().union(*(df.columns for df in datasets)))

    aligned = []
    for df in datasets:
        missing = list(set(all_cols) - set(df.columns))
        if missing:
            df = pd.concat(
                [df, pd.DataFrame(np.nan, index=df.index, columns=missing)],
                axis=1
            )
        aligned.append(df[all_cols])

    return pd.concat(aligned, ignore_index=True)


# ---------------------------
# 🔥 GLOBAL TEMPORAL MERGE (CRITICAL FIX)
# ---------------------------
def rebuild_global_visits(df: pd.DataFrame) -> pd.DataFrame:

    # sort by user + dataset + visit
    df = df.sort_values(["swanid", "source_dataset", "visit"], kind="stable")

    # cumulative visit across datasets
    df["visit"] = df.groupby("swanid", sort=False).cumcount() + 1

    return df


# ---------------------------
# SPARSITY
# ---------------------------
def drop_sparse_columns(df: pd.DataFrame):

    missing_ratio = df.isna().mean()

    protected = set(schema.get_model_input_order()) | {"swanid", "visit"}

    drop_cols = [
        col for col in df.columns
        if missing_ratio[col] > MISSING_THRESHOLD and col not in protected
    ]

    logger.info(f"Dropping {len(drop_cols)} sparse columns")

    return df.drop(columns=drop_cols), drop_cols


# ---------------------------
# DEDUPE
# ---------------------------
def clean_dataset(df: pd.DataFrame):

    before = len(df)

    df = df.sort_values(["swanid", "visit"], kind="stable")
    df = df.drop_duplicates(subset=["swanid", "visit"], keep="first")

    removed = before - len(df)

    logger.info(f"Removed duplicates: {removed}")

    return df, removed


# ---------------------------
# SCHEMA PROJECTION
# ---------------------------
def project_to_model_schema(df: pd.DataFrame):

    cols = ["swanid", "visit"] + sorted(schema.get_model_input_order())

    missing = list(set(cols) - set(df.columns))

    if missing:
        df = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing)],
            axis=1
        )

    return df[cols]


# ---------------------------
# SANITIZATION
# ---------------------------
def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    return df


# ---------------------------
# VALIDATION
# ---------------------------
def validate_final(df: pd.DataFrame):

    if df.empty:
        raise ValueError("Final dataset is empty")

    if df["swanid"].isna().any():
        raise ValueError("Invalid swanid")

    if df["visit"].isna().any():
        raise ValueError("Invalid visit")

    numeric = df.select_dtypes(include=[np.number])

    if np.isinf(numeric.values).any():
        raise ValueError("Infinite values detected")


# ---------------------------
# AUDIT
# ---------------------------
def build_audit(df, dropped_cols, removed_rows):

    visits = df.groupby("swanid")["visit"].nunique()

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "users": int(df["swanid"].nunique()),
        "avg_visits": float(visits.mean()),
        "max_visits": int(visits.max()),
        "missing_ratio": float(df.isna().mean().mean()),
        "dropped_columns": dropped_cols,
        "duplicates_removed": int(removed_rows),
    }


# ---------------------------
# MAIN
# ---------------------------
def unify_datasets():

    datasets = []

    for file_name in DATASETS:
        try:
            datasets.append(load_dataset(file_name))
        except Exception as e:
            logger.error(f"Failed loading {file_name}: {e}")

    if not datasets:
        raise RuntimeError("No datasets loaded")

    unified = align_all_columns(datasets)

    unified = rebuild_global_visits(unified)  # 🔥 FIXED

    unified, dropped_cols = drop_sparse_columns(unified)

    unified, removed_rows = clean_dataset(unified)

    unified = project_to_model_schema(unified)

    unified = sanitize_numeric(unified)

    validate_final(unified)

    audit = build_audit(unified, dropped_cols, removed_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(OUTPUT_PATH, index=False)

    with open(AUDIT_PATH, "w") as f:
        json.dump(audit, f, indent=2)

    logger.info("=== FINAL REPORT ===")
    logger.info(f"Shape: {unified.shape}")
    logger.info(f"Users: {unified['swanid'].nunique()}")

    return unified


if __name__ == "__main__":
    unify_datasets()