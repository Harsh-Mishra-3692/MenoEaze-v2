from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
import pandas as pd
import numpy as np
import hashlib


FeatureType = Literal["binary", "ordinal", "continuous", "meta"]


# =========================================================
# FEATURE DEFINITION
# =========================================================
@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    ftype: FeatureType
    valid_range: Optional[tuple] = None
    required: bool = False


# =========================================================
# SCHEMA
# =========================================================
@dataclass(frozen=True)
class FeatureSchema:

    version: str = "v10.0"

    features: List[FeatureDefinition] = field(default_factory=lambda: [

        FeatureDefinition("swanid", "meta", required=True),
        FeatureDefinition("visit", "meta", (1, 100), required=True),

        FeatureDefinition("hotflash", "binary"),
        FeatureDefinition("nisweat", "binary"),
        FeatureDefinition("diffislp", "binary"),
        FeatureDefinition("depress", "binary"),
        FeatureDefinition("tense", "binary"),
        FeatureDefinition("irritab", "binary"),
        FeatureDefinition("forget", "binary"),
        FeatureDefinition("headach", "binary"),
        FeatureDefinition("stiff", "binary"),
        FeatureDefinition("dryness", "binary"),

        FeatureDefinition("age", "continuous", (0, 120)),
        FeatureDefinition("bmi", "continuous", (10, 80)),
        FeatureDefinition("health", "ordinal", (1, 5)),
    ])

    missing_codes: List[str] = field(default_factory=lambda: ["", " ", "-1", "-9"])

    # =========================================================
    # ACCESS
    # =========================================================
    def get_features_by_type(self, ftype: FeatureType) -> List[str]:
        return [f.name for f in self.features if f.ftype == ftype]

    def get_model_input_order(self) -> List[str]:
        return [
            "hotflash","nisweat","diffislp",
            "depress","tense","irritab",
            "forget","headach","stiff","dryness",
            "age","bmi","health"
        ]

    def get_training_columns(self) -> List[str]:
        return ["swanid","visit"] + self.get_model_input_order() + ["severity"]

    def get_required_features(self) -> List[str]:
        return [f.name for f in self.features if f.required]

    # =========================================================
    # SCHEMA HASH
    # =========================================================
    def get_schema_hash(self) -> str:
        raw = "|".join([f"{f.name}:{f.ftype}:{f.valid_range}" for f in self.features])
        return hashlib.md5(raw.encode()).hexdigest()

    # =========================================================
    # VALIDATION
    # =========================================================
    def validate_columns(self, df: pd.DataFrame):
        missing = [c for c in self.get_required_features() if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def validate_ranges(self, df: pd.DataFrame):
        for f in self.features:
            if f.valid_range and f.name in df.columns:
                lo, hi = f.valid_range
                df[f.name] = df[f.name].clip(lo, hi)

    # =========================================================
    # ENFORCE SCHEMA
    # =========================================================
    def enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.get_model_input_order():
            if col not in df.columns:
                df[col] = 0.0
        return df

    # =========================================================
    # TRANSFORMS
    # =========================================================
    def transform_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.get_features_by_type("binary"):
            if col not in df.columns:
                continue

            s = pd.to_numeric(df[col], errors="coerce")
            s = s.replace(self.missing_codes, np.nan)

            s = s.map(lambda x: 1.0 if x in [1, 2, True] else 0.0)

            df[col] = s.fillna(0.0).astype("float32")

        return df

    def transform_ordinal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.get_features_by_type("ordinal"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                df[col] = s.fillna(0.0).astype("float32")

        return df

    def transform_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.get_features_by_type("continuous"):
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")

                s = s.replace([np.inf, -np.inf], np.nan)

                median = s.median() if not s.dropna().empty else 0.0
                s = s.fillna(median)

                std = s.std()
                if std > 1e-6:
                    s = (s - s.mean()) / std
                else:
                    s = s * 0.0

                df[col] = s.astype("float32")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.transform_binary(df)
        df = self.transform_ordinal(df)
        df = self.transform_continuous(df)
        df = self.enforce_schema(df)

        df = df.replace([np.inf, -np.inf], 0.0)
        return df

    # =========================================================
    # 🔥 FIXED SEVERITY (TEMPORAL SAFE)
    # =========================================================
    def get_symptom_weights(self) -> Dict[str, float]:
        return {
            "hotflash": 0.25,
            "nisweat": 0.20,
            "diffislp": 0.15,
            "depress": 0.10,
            "tense": 0.10,
            "irritab": 0.08,
            "forget": 0.05,
            "headach": 0.03,
            "stiff": 0.02,
            "dryness": 0.02,
        }

    def compute_severity(self, df: pd.DataFrame) -> pd.Series:
        weights = self.get_symptom_weights()
        cols = [c for c in weights if c in df.columns]

        if not cols:
            return pd.Series(np.zeros(len(df)), index=df.index)

        df = df.copy()

        X = df[cols].fillna(0).values.astype("float32")
        w = np.array([weights[c] for c in cols], dtype="float32")

        raw = X @ w
        raw = raw / (w.sum() + 1e-6)

        # =========================================================
        # 🔥 SEQUENCE-AWARE NORMALIZATION (CRITICAL FIX)
        # =========================================================
        severity = np.zeros_like(raw)

        group_key = "sequence_id" if "sequence_id" in df.columns else "swanid"

        for gid, idx in df.groupby(group_key).groups.items():
            seq_vals = raw[idx]

            if len(seq_vals) < 2:
                severity[idx] = seq_vals
                continue

            min_v = seq_vals.min()
            max_v = seq_vals.max()

            if max_v - min_v > 1e-6:
                severity[idx] = (seq_vals - min_v) / (max_v - min_v)
            else:
                severity[idx] = 0.5  # fallback

        # small noise (break ties safely)
        noise = np.random.default_rng(42).normal(0, 1e-4, size=len(severity))
        severity = severity + noise

        return pd.Series(np.clip(severity, 0.0, 1.0), index=df.index)

    # =========================================================
    # FINALIZE
    # =========================================================
    def finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.get_training_columns()

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing final columns: {missing}")

        df = df[cols].copy()

        for col in self.get_model_input_order():
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0.0).astype("float32")

        df["severity"] = df["severity"].clip(0.0, 1.0)

        return df

    # =========================================================
    # METADATA
    # =========================================================
    def attach_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["schema_version"] = self.version
        df["schema_hash"] = self.get_schema_hash()
        return df