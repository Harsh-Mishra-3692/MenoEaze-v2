import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import json
import logging
import hashlib


# ---------------------------
# CONFIG
# ---------------------------

MISSING_CODES = {"", " ", "-1", "-9"}

REQUIRED_COLUMNS = {"swanid"}

DATASETS = [
    "04368-0001-Data.tsv",
    "28762-0001-Data.tsv",
    "29221-0001-Data.tsv",
    "32961-0001-Data.tsv",
]

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/interim")
AUDIT_PATH = Path("data/interim/cleaning_audit.json")

NUMERIC_THRESHOLD = 0.95
MAX_ABS_VALUE = 1e6
MAX_ROWS = 2_000_000  # safety cap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# FILE HASH (INTEGRITY)
# ---------------------------
def file_hash(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


# ---------------------------
# SAFE NUMERIC INFERENCE
# ---------------------------
def infer_numeric(series: pd.Series) -> Tuple[pd.Series, bool]:
    converted = pd.to_numeric(series, errors="coerce")
    ratio = converted.notna().mean()

    if ratio >= NUMERIC_THRESHOLD:
        converted = converted.where(converted.abs() < MAX_ABS_VALUE)
        return converted.astype("float32"), True

    return series, False


# ---------------------------
# COLUMN NORMALIZATION
# ---------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


# ---------------------------
# SAFE LOAD
# ---------------------------
def safe_load(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            low_memory=False,
            nrows=MAX_ROWS,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {path.name}: {e}")


# ---------------------------
# CORE CLEAN FUNCTION
# ---------------------------
def clean_dataset(input_path: Path, output_path: Path) -> Tuple[pd.DataFrame, Dict]:

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    df = safe_load(input_path)

    df = normalize_columns(df)

    report = {
        "dataset": input_path.name,
        "file_hash": file_hash(input_path),
        "rows_before": int(df.shape[0]),
        "columns_before": int(df.shape[1]),
    }

    # ---------------------------
    # Replace missing codes
    # ---------------------------
    df = df.replace(list(MISSING_CODES), np.nan)

    report["missing_ratio"] = float(df.isna().mean().mean())

    # ---------------------------
    # Validate required columns
    # ---------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------------------------
    # Numeric inference
    # ---------------------------
    numeric_cols = []
    for col in df.columns:
        cleaned, is_numeric = infer_numeric(df[col])
        if is_numeric:
            df[col] = cleaned
            numeric_cols.append(col)

    report["numeric_columns"] = len(numeric_cols)

    # ---------------------------
    # Enforce ID
    # ---------------------------
    df["swanid"] = pd.to_numeric(df["swanid"], errors="coerce").astype("Int64")

    if "lvisit" in df.columns:
        df["lvisit"] = pd.to_numeric(df["lvisit"], errors="coerce")

    # ---------------------------
    # Drop invalid rows
    # ---------------------------
    before_rows = len(df)
    df = df.dropna(subset=["swanid"])
    report["rows_dropped_invalid_id"] = before_rows - len(df)

    # ---------------------------
    # Remove duplicates
    # ---------------------------
    before_dup = len(df)

    if "lvisit" in df.columns:
        df = df.drop_duplicates(subset=["swanid", "lvisit"])
    else:
        df = df.drop_duplicates(subset=["swanid"])

    report["duplicates_removed"] = before_dup - len(df)

    # ---------------------------
    # Drop fully empty columns
    # ---------------------------
    before_cols = df.shape[1]
    df = df.dropna(axis=1, how="all")
    report["columns_removed_empty"] = before_cols - df.shape[1]

    # ---------------------------
    # Add metadata
    # ---------------------------
    df["source_dataset"] = input_path.stem

    # ---------------------------
    # Sort deterministically
    # ---------------------------
    if "lvisit" in df.columns:
        df = df.sort_values(["swanid", "lvisit"])
    else:
        df = df.sort_values(["swanid"])

    df = df.reset_index(drop=True)

    # ---------------------------
    # FINAL SAFETY PASS
    # ---------------------------
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in numeric_cols:
        df[col] = df[col].fillna(0.0)

    # ---------------------------
    # Save
    # ---------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report["rows_after"] = int(df.shape[0])
    report["columns_after"] = int(df.shape[1])

    logger.info(f"✔ Cleaned {input_path.name} → {df.shape}")

    return df, report


# ---------------------------
# RUN ALL
# ---------------------------
def run_all():

    audits = {}

    for file in DATASETS:
        try:
            _, report = clean_dataset(
                RAW_DIR / file,
                OUTPUT_DIR / f"{file.split('-')[0]}_clean.csv",
            )
            audits[file] = report

        except Exception as e:
            logger.error(f"❌ Failed processing {file}: {e}")
            audits[file] = {"error": str(e)}

    # save audit
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(AUDIT_PATH, "w") as f:
        json.dump(audits, f, indent=2)

    logger.info(f"✔ Audit saved → {AUDIT_PATH}")


# ---------------------------
# ENTRY
# ---------------------------
if __name__ == "__main__":
    run_all()