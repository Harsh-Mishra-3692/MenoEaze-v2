import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import json
import hashlib
import logging
from datetime import datetime
from feature_schema import FeatureSchema


# ---------------------------
# GLOBAL SAFETY
# ---------------------------
pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------
# CONFIG
# ---------------------------
MISSING_CODES = {"", " ", "-1", "-9"}
AUDIT_PATH = Path("data/interim/inspect_audit.json")
MAX_ROWS = 1_000_000  # prevent memory blowups
MAX_ABS_VALUE = 1e6   # sanity clamp

schema = FeatureSchema()


# ---------------------------
# FILE HASH (REPRODUCIBILITY)
# ---------------------------
def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------
# ROBUST LOADER
# ---------------------------
def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    engines = [
        {"engine": "c", "encoding": "utf-8"},
        {"engine": "python", "encoding": "utf-8"},
        {"engine": "python", "encoding": "latin1"},
    ]

    last_error = None

    for config in engines:
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                nrows=MAX_ROWS,
                low_memory=False if config["engine"] == "c" else None,
                engine=config["engine"],
                encoding=config["encoding"],
                on_bad_lines="skip",
            )
            logger.info(f"✔ Loaded {path.name} ({config}) → {df.shape}")
            break
        except Exception as e:
            last_error = e
    else:
        raise RuntimeError(f"Failed loading {path.name}: {last_error}")

    # Normalize columns deterministically
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^\w]", "", regex=True)
    )

    return df


# ---------------------------
# SAFE JSON SERIALIZATION
# ---------------------------
def make_json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    elif isinstance(obj, (pd.Index,)):
        return list(obj)
    return obj


# ---------------------------
# CLEAN
# ---------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(list(MISSING_CODES), np.nan)
    df = df.infer_objects(copy=False)
    return df


# ---------------------------
# NUMERIC EXTRACTION (SAFE)
# ---------------------------
def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = {}

    for col in sorted(df.columns):  # deterministic
        try:
            s = pd.to_numeric(df[col], errors="coerce")

            if s.notna().mean() > 0.8:
                s = s.where(s.abs() < MAX_ABS_VALUE)
                numeric[col] = s

        except Exception:
            continue

    return pd.DataFrame(numeric, index=df.index)


# ---------------------------
# CORE ANALYSIS
# ---------------------------
def analyze(df: pd.DataFrame, dataset_name: str) -> Dict:

    if df.empty:
        return {"error": "Empty dataset"}

    df_clean = clean_dataframe(df)
    numeric_df = safe_numeric(df_clean)

    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # ---------------------------
    # BASIC
    # ---------------------------
    rows, cols = df.shape
    result["shape"] = (int(rows), int(cols))
    result["missing_ratio"] = float(df_clean.isna().mean().mean())

    # ---------------------------
    # SCHEMA COVERAGE
    # ---------------------------
    model_features = schema.get_model_input_order()

    result["schema_coverage"] = {
        col: float(df_clean[col].notna().mean()) if col in df_clean.columns else 0.0
        for col in model_features
    }

    # ---------------------------
    # USERS
    # ---------------------------
    if "swanid" in df_clean.columns:
        result["users"] = int(df_clean["swanid"].nunique())
    else:
        result["users"] = None

    # ---------------------------
    # TEMPORAL
    # ---------------------------
    if {"swanid", "lvisit"}.issubset(df_clean.columns):

        visits = pd.to_numeric(df_clean["lvisit"], errors="coerce")
        grouped = visits.groupby(df_clean["swanid"]).nunique()

        if not grouped.empty:
            result["avg_visits"] = float(grouped.mean())
            result["max_visits"] = int(grouped.max())
        else:
            result["avg_visits"] = 0.0
            result["max_visits"] = 0

    # ---------------------------
    # DISTRIBUTION
    # ---------------------------
    if not numeric_df.empty:
        desc = numeric_df.describe(percentiles=[0.01, 0.5, 0.99]).T

        result["distribution"] = {
            str(idx): {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "p01": float(row["1%"]),
                "p50": float(row["50%"]),
                "p99": float(row["99%"]),
            }
            for idx, row in desc.iterrows()
        }

    # ---------------------------
    # SPARSITY
    # ---------------------------
    sparsity = df_clean.isna().mean().sort_values(ascending=False).head(10)
    result["top_sparse"] = {str(k): float(v) for k, v in sparsity.items()}

    # ---------------------------
    # HEALTH FLAGS
    # ---------------------------
    result["health_flags"] = {
        "high_missing": result["missing_ratio"] > 0.4,
        "extreme_values": any(
            abs(v) > MAX_ABS_VALUE for col in numeric_df.columns for v in numeric_df[col].dropna().head(100)
        ),
        "low_users": result.get("users", 0) is not None and result["users"] < 100,
    }

    return result


# ---------------------------
# DATASET COMPARISON
# ---------------------------
def compare(paths: List[Path]) -> Dict:

    summaries = {}
    all_columns = {}

    for path in paths:
        logger.info(f"\n📊 {path.name}")

        try:
            df = load(path)
            stats = analyze(df, path.name)

        except Exception as e:
            logger.error(f"❌ Failed: {path.name} → {e}")
            summaries[path.name] = {"error": str(e)}
            continue

        summaries[path.name] = stats
        all_columns[path.name] = set(df.columns)

    # ---------------------------
    # COLUMN OVERLAP
    # ---------------------------
    if all_columns:
        base = next(iter(all_columns.values()))

        summaries["column_overlap"] = {
            name: float(len(base & cols) / max(len(base), 1))
            for name, cols in all_columns.items()
        }

    return summaries


# ---------------------------
# SAVE AUDIT
# ---------------------------
def save_audit(audit: Dict):
    safe_audit = make_json_safe(audit)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(AUDIT_PATH, "w") as f:
        json.dump(safe_audit, f, indent=2)

    logger.info(f"✔ Saved audit → {AUDIT_PATH}")


# ---------------------------
# ENTRY
# ---------------------------
if __name__ == "__main__":

    paths = [
        Path("data/raw/04368-0001-Data.tsv"),
        Path("data/raw/28762-0001-Data.tsv"),
        Path("data/raw/29221-0001-Data.tsv"),
        Path("data/raw/32961-0001-Data.tsv"),
    ]

    audit = compare(paths)
    save_audit(audit)