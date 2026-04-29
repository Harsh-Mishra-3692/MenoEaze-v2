"""
preprocess.py
=============
Final production + research-grade preprocessing pipeline.

Features:
- leakage-safe imputation
- separate scaling (temporal vs static)
- scaler persistence (for API)
- strict sequence validation
- MAML-ready splits
- zero-NaN guarantee
- shuffled training data
- metadata for personalization
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# ── Config ─────────────────────────────────────────────
SEED = 42
SEQ_LEN = 5
TRAIN_RATIO = 0.8

np.random.seed(SEED)

TEMPORAL_FEATURES = [
    "hot_flash_score",
    "night_sweats_score",
    "sleep_quality",
    "mood_score",
    "fatigue_score",
    "anxiety_score",
    "physical_activity",
    "stress_level",
    "caffeine_intake",
]

STATIC_FEATURES = ["age", "bmi"]
ALL_FEATURES = TEMPORAL_FEATURES + STATIC_FEATURES
TARGET = "severity"


# ── 1. Imputation ──────────────────────────────────────
def _impute(df):
    df = df.copy()
    df = df.sort_values(["patient_id", "day"]).reset_index(drop=True)

    # ensure numeric
    df[TEMPORAL_FEATURES] = df[TEMPORAL_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    # per-patient interpolation
    df[TEMPORAL_FEATURES] = (
        df.groupby("patient_id")[TEMPORAL_FEATURES]
        .transform(lambda x: x.interpolate().ffill().bfill())
    )

    # fallback fill
    df[TEMPORAL_FEATURES] = df[TEMPORAL_FEATURES].fillna(
        df[TEMPORAL_FEATURES].mean()
    )

    df[STATIC_FEATURES] = df[STATIC_FEATURES].fillna(
        df[STATIC_FEATURES].mean()
    )

    df[TARGET] = df[TARGET].fillna(df[TARGET].mean())

    return df


# ── 2. Diagnostics ─────────────────────────────────────
def _diagnostics(df):
    print("\nDATASET DIAGNOSTICS")
    print("-" * 40)

    print("Mean:\n", df[ALL_FEATURES].mean())
    print("\nStd:\n", df[ALL_FEATURES].std())
    print("\nMin:\n", df[ALL_FEATURES].min())
    print("\nMax:\n", df[ALL_FEATURES].max())

    print("\nRemaining NaNs:", df.isna().sum().sum())
    print("-" * 40)


# ── 3. Sequence Builder ────────────────────────────────
def _build_sequences(df, patient_ids):
    X, y, pids = [], [], []

    for pid in patient_ids:
        patient_df = df[df["patient_id"] == pid].sort_values("day")

        if len(patient_df) < SEQ_LEN + 1:
            continue

        features = patient_df[ALL_FEATURES].values
        targets = patient_df[TARGET].values

        for i in range(len(features) - SEQ_LEN):
            seq_x = features[i:i + SEQ_LEN]
            seq_y = targets[i + SEQ_LEN]

            if np.isnan(seq_x).any() or np.isnan(seq_y):
                continue

            X.append(seq_x)
            y.append(seq_y)
            pids.append(pid)

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(pids),
    )


# ── 4. MAML Split ──────────────────────────────────────
def _maml_split(df, patient_ids):
    support_X, support_y = [], []
    query_X, query_y = [], []

    for pid in patient_ids:
        patient_df = df[df["patient_id"] == pid].sort_values("day")

        if len(patient_df) < 10:
            continue

        features = patient_df[ALL_FEATURES].values
        targets = patient_df[TARGET].values

        split = int(len(features) * 0.5)

        for i in range(split - SEQ_LEN):
            support_X.append(features[i:i + SEQ_LEN])
            support_y.append(targets[i + SEQ_LEN])

        for i in range(split, len(features) - SEQ_LEN):
            query_X.append(features[i:i + SEQ_LEN])
            query_y.append(targets[i + SEQ_LEN])

    return (
        np.array(support_X, dtype=np.float32),
        np.array(support_y, dtype=np.float32),
        np.array(query_X, dtype=np.float32),
        np.array(query_y, dtype=np.float32),
    )


# ── 5. Main ────────────────────────────────────────────
def main():
    print("━" * 60)
    print("Advanced Preprocessing Pipeline (Final Production)")
    print("━" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "synthetic_menopause_data.csv")

    df = pd.read_csv(csv_path)
    print(f"Loaded rows: {len(df)}")

    # ── Impute
    df = _impute(df)

    # ── Split patients
    patient_ids = df["patient_id"].unique()
    np.random.shuffle(patient_ids)

    split = int(len(patient_ids) * TRAIN_RATIO)
    train_ids = patient_ids[:split]
    test_ids = patient_ids[split:]

    # ── Scaling
    scaler_temp = MinMaxScaler()
    scaler_static = MinMaxScaler()

    train_mask = df["patient_id"].isin(train_ids)

    scaler_temp.fit(df.loc[train_mask, TEMPORAL_FEATURES])
    scaler_static.fit(df.loc[train_mask, STATIC_FEATURES])

    df[TEMPORAL_FEATURES] = scaler_temp.transform(df[TEMPORAL_FEATURES])
    df[STATIC_FEATURES] = scaler_static.transform(df[STATIC_FEATURES])

    # ── Final NaN cleanup
    df[ALL_FEATURES] = np.nan_to_num(df[ALL_FEATURES])

    # ── Diagnostics
    _diagnostics(df)

    # ── Build sequences
    X_train, y_train, train_pid = _build_sequences(df, train_ids)
    X_test, y_test, test_pid = _build_sequences(df, test_ids)

    # ── Shuffle training data (IMPORTANT)
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # ── Assertions
    assert X_train.shape[1] == SEQ_LEN
    assert X_train.shape[2] == len(ALL_FEATURES)
    assert not np.isnan(X_train).any()
    assert not np.isnan(y_train).any()

    # ── MAML
    support_X, support_y, query_X, query_y = _maml_split(df, train_ids)

    print(f"\nTrain: {X_train.shape}")
    print(f"Test : {X_test.shape}")
    print(f"Support: {support_X.shape}")
    print(f"Query  : {query_X.shape}")

    # ── Save scalers (CRITICAL)
    torch.save(
        {
            "scaler_temp": scaler_temp,
            "scaler_static": scaler_static,
            "temporal_features": TEMPORAL_FEATURES,
            "static_features": STATIC_FEATURES,
        },
        os.path.join(base_dir, "scalers.pt"),
    )

    # ── Save tensors
    torch.save(
        {
            "X_train": torch.tensor(X_train),
            "y_train": torch.tensor(y_train),
            "X_test": torch.tensor(X_test),
            "y_test": torch.tensor(y_test),
            "support_X": torch.tensor(support_X),
            "support_y": torch.tensor(support_y),
            "query_X": torch.tensor(query_X),
            "query_y": torch.tensor(query_y),
            "train_patient_ids": train_pid,
            "test_patient_ids": test_pid,
            "feature_names": ALL_FEATURES,
            "feature_order": ALL_FEATURES,
            "seq_len": SEQ_LEN,
        },
        os.path.join(base_dir, "preprocessed_data.pt"),
    )

    print("\nSaved:")
    print("- preprocessed_data.pt")
    print("- scalers.pt")
    print("━" * 60)


if __name__ == "__main__":
    main()
