# ml_engine/research/run_analysis.py

import numpy as np
import time

from ml_engine.pipeline import full_pipeline
from ml_engine.research.semantic_metrics import semantic_similarity
from ml_engine.research.llm_judge import evaluate_answer


# ─────────────────────────────────────────────
# EVALUATION QUERY SET (EXPANDED)
# ─────────────────────────────────────────────
QUERIES = [
    "hot flashes treatment",
    "anxiety during menopause",
    "sleep issues menopause",
    "hormone therapy risks",
    "menopause depression treatment",
    "bone density menopause",
    "weight gain menopause causes",
    "night sweats solution",
    "menopause fatigue reasons",
    "natural remedies menopause"
]


def run(use_reranker: bool = True):

    print("\n" + "=" * 60)
    print("RUNNING ANALYSIS")
    print("Mode:", "FULL SYSTEM" if use_reranker else "BASELINE")
    print("=" * 60)

    results = []

    for q in QUERIES:
        seq = np.random.rand(5, 11).astype("float32")

        start = time.time()

        out = full_pipeline(
            user_id="test_user",
            query=q,
            sequence=seq,
            user_history=None,
            use_reranker=use_reranker  # 👈 baseline toggle
        )

        latency = (time.time() - start) * 1000

        answer = out["rag"]["answer"]
        docs = out["rag"].get("sources", [])

        context = " ".join(docs)

        sim = semantic_similarity(answer, context)
        judge = evaluate_answer(answer, context)

        results.append({
            "query": q,
            "similarity": sim,
            "judge_score": judge["overall"],
            "confidence": out["rag"]["confidence"],
            "latency": latency
        })

        print(f"\nQuery: {q}")
        print(f"Similarity: {sim:.3f}")
        print(f"Judge: {judge['overall']:.3f}")
        print(f"Latency: {latency:.2f} ms")

    # ─────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────
    sims = [r["similarity"] for r in results]
    judges = [r["judge_score"] for r in results]
    lats = [r["latency"] for r in results]

    print("\n" + "=" * 60)
    print("FINAL METRICS")
    print("=" * 60)

    print(f"Avg Similarity : {np.mean(sims):.3f}")
    print(f"Avg Judge Score: {np.mean(judges):.3f}")
    print(f"Avg Latency    : {np.mean(lats):.2f} ms")

    print(f"Min Similarity : {np.min(sims):.3f}")
    print(f"Max Similarity : {np.max(sims):.3f}")

    return {
        "avg_similarity": float(np.mean(sims)),
        "avg_judge": float(np.mean(judges)),
        "avg_latency": float(np.mean(lats))
    }


# ─────────────────────────────────────────────
# MAIN (BASELINE vs FULL)
# ─────────────────────────────────────────────
if __name__ == "__main__":

    baseline = run(use_reranker=False)
    full = run(use_reranker=True)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    print("\nBaseline:")
    print(baseline)

    print("\nFull System:")
    print(full)