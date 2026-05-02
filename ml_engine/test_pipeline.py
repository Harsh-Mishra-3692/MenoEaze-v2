import sys
import traceback
import pandas as pd
import numpy as np
import torch

from sequence_loader import load_sequences, to_tensor
from temporal_augmenter import generate_synthetic_sequence
from sap_gru_model import SAP_GRU


# ---------------------------
# CONFIG
# ---------------------------
SEQ_PATH = "data/processed/longitudinal_sequences.csv"
FEATURE_PATH = "data/processed/features.csv"

MAX_SAMPLE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# UTIL
# ---------------------------
def fail(msg):
    print(f"\n❌ FAILURE: {msg}")
    sys.exit(1)


def safe_run(step_name, fn):
    print(f"\n=== {step_name} ===")
    try:
        return fn()
    except Exception:
        print(f"\n❌ ERROR in {step_name}")
        traceback.print_exc()
        sys.exit(1)


# ---------------------------
# STEP 1: LOAD SEQUENCES
# ---------------------------
def step_load_sequences():

    seqs, targets, lengths, weights = load_sequences(SEQ_PATH)

    if not seqs or not targets:
        fail("Empty sequences or targets")

    if not (len(seqs) == len(targets) == len(lengths) == len(weights)):
        fail("Mismatch: sequences / targets / lengths / weights")

    if any(l <= 0 for l in lengths):
        fail("Invalid sequence length detected")

    print(f"✔ Sequences: {len(seqs)}")
    print(f"✔ Targets: {len(targets)}")
    print(f"✔ Weights: {len(weights)}")
    print(f"✔ Sample lengths: {lengths[:5]}")

    return seqs, targets, lengths, weights


# ---------------------------
# STEP 2: TENSOR CONVERSION
# ---------------------------
def step_tensor(seqs, targets, lengths, weights):

    data = to_tensor(seqs, targets, lengths, weights, device=DEVICE)

    X = data["X"]
    y = data["y"]
    mask = data["mask"]
    w = data["weights"]

    # shape checks
    if X.ndim != 3:
        fail("X must be 3D tensor")

    if y.ndim != 1:
        fail("y must be 1D tensor")

    if mask.shape[:2] != X.shape[:2]:
        fail("Mask shape mismatch")

    if len(w) != len(y):
        fail("Weights mismatch with targets")

    # numeric safety
    if not torch.isfinite(X).all():
        fail("X contains non-finite values")

    if not torch.isfinite(y).all():
        fail("y contains non-finite values")

    if not torch.isfinite(w).all():
        fail("weights contain non-finite values")

    print(f"✔ X shape: {tuple(X.shape)}")
    print(f"✔ y shape: {tuple(y.shape)}")
    print(f"✔ mask shape: {tuple(mask.shape)}")

    return data


# ---------------------------
# STEP 3: MODEL FORWARD
# ---------------------------
def step_model(data):

    X = data["X"][:MAX_SAMPLE]

    model = SAP_GRU(input_dim=X.shape[-1]).to(DEVICE)
    model.eval()

    with torch.no_grad():
        out = model(X)

    required_keys = {"severity", "global", "personal"}

    if not isinstance(out, dict):
        fail("Model output must be dict")

    if not required_keys.issubset(out.keys()):
        fail(f"Missing keys in model output: {required_keys}")

    if not torch.isfinite(out["severity"]).all():
        fail("Model output contains NaN/Inf")

    if out["severity"].ndim != 1:
        fail("Severity output must be 1D")

    print(f"✔ Output keys: {list(out.keys())}")
    print(f"✔ Sample severity: {out['severity']}")

    return model


# ---------------------------
# STEP 4: TEMPORAL AUGMENTER
# ---------------------------
def step_augmenter():

    df = pd.read_csv(FEATURE_PATH)

    if df.empty:
        fail("features.csv is empty")

    required_cols = {"swanid", "visit"}
    if not required_cols.issubset(df.columns):
        fail("Missing required columns in features")

    feature_cols = [
        c for c in df.columns
        if c not in ["swanid", "visit", "severity"]
    ]

    sample_row = df.iloc[0]
    user_id = int(sample_row["swanid"])

    synthetic = generate_synthetic_sequence(
        sample_row,
        feature_cols,
        user_id=user_id
    )

    if synthetic.empty:
        fail("Synthetic sequence is empty")

    if not np.isfinite(synthetic[feature_cols].values).all():
        fail("Synthetic contains non-finite values")

    if synthetic["visit"].nunique() < 2:
        fail("Synthetic sequence too short")

    print(f"✔ Synthetic shape: {synthetic.shape}")
    print(synthetic.head())


# ---------------------------
# MAIN
# ---------------------------
def main():

    print("\n🚀 STARTING FULL PIPELINE TEST")

    seqs, targets, lengths, weights = safe_run(
        "STEP 1: LOAD SEQUENCES",
        step_load_sequences
    )

    data = safe_run(
        "STEP 2: TENSOR CONVERSION",
        lambda: step_tensor(seqs, targets, lengths, weights)
    )

    safe_run(
        "STEP 3: MODEL FORWARD PASS",
        lambda: step_model(data)
    )

    safe_run(
        "STEP 4: TEMPORAL AUGMENTER",
        step_augmenter
    )

    print("\n✅ ALL TESTS PASSED — SYSTEM STABLE")


if __name__ == "__main__":
    main()