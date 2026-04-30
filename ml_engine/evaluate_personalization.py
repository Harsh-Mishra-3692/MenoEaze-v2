import sys
import os

# Phase 1: sys.path sanitizer
# Temporarily remove the local directory to prevent shadowing standard library modules (like 'logging')
_cwd = os.getcwd()
if _cwd in sys.path:
    sys.path.remove(_cwd)

import numpy as np
import random

# Restore path
sys.path.insert(0, _cwd)

# Phase 2: In-memory evaluation via pipeline.py instead of requests.post
from ml_engine.pipeline import full_pipeline

USER_ID = "eval_user_v2"

def generate_sequence():
    return np.random.uniform(0.2, 0.8, (5, 11)).astype(np.float32)

def main():
    base_errors = []
    adapted_errors = []

    for i in range(30):
        sequence = generate_sequence()
        actual = random.uniform(0.3, 0.9)

        # In-memory inference
        res = full_pipeline(
            user_id=USER_ID,
            query="",
            sequence=sequence,
            user_history=None,
            use_reranker=False
        )

        base = res["ml"]["base_severity"]
        adapted = res["ml"]["severity"]

        base_errors.append(abs(actual - base))
        adapted_errors.append(abs(actual - adapted))

    print("\nMAE BASE:", np.mean(base_errors))
    print("MAE ADAPTED:", np.mean(adapted_errors))

if __name__ == "__main__":
    main()
