# hybrid_retriever.py — FINAL ELITE v3 (STABLE + CORRECT + HYBRID-SAFE)

import logging
import re
import time
from typing import List, Dict, Any, Optional

from ml_engine.hf_embedder import HFEmbedder
from ml_engine.db_client import call_rpc, fetch_table

# ✅ correct imports (aligned with your new modules)
try:
    from ml_engine.research.bm25_index import bm25_rank
except Exception:
    bm25_rank = None

try:
    from ml_engine.research.fusion import fuse_results
except Exception:
    fuse_results = None

logger = logging.getLogger("menoeaze.hybrid")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_K = 8
MAX_DOC_LENGTH = 1000

PERSONALIZATION_WEIGHT = 0.15
PRIORITY_BOOST = 0.25
MIN_SCORE = 0.05

MAX_QUERY_LENGTH = 500

_embedder = None


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            _embedder = HFEmbedder()
        except Exception:
            logger.exception("[HYBRID][EMBEDDER][FAIL]")
            _embedder = None
    return _embedder


# ─────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────
def _clean(text: str) -> str:
    try:
        return re.sub(r"\s+", " ", text.lower()).strip()[:MAX_QUERY_LENGTH]
    except Exception:
        return ""


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────
def _validate_doc(d: Dict) -> bool:
    if not isinstance(d, dict):
        return False
    if not d.get("content"):
        return False
    return True


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


# ─────────────────────────────────────────────
# VECTOR SEARCH
# ─────────────────────────────────────────────
def _vector_search(query: str, k: int) -> List[Dict]:
    embedder = _get_embedder()
    if embedder is None:
        return []

    try:
        vec = embedder.embed([query])[0]

        if vec is None or not hasattr(vec, "any") or not vec.any():
            return []

        res = call_rpc(
            "match_medical_documents",
            {
                "query_embedding": vec.tolist(),
                "match_count": k,
            }
        )

        if not isinstance(res, list):
            return []

        # normalize schema
        docs = []
        for r in res:
            if not _validate_doc(r):
                continue

            docs.append({
                "content": r.get("content"),
                "source": r.get("document_name"),
                "score": _safe_float(r.get("similarity", 0.0)),
                "priority": _safe_float(r.get("priority", 0.0)),
            })

        return docs

    except Exception as e:
        logger.error(f"[HYBRID][VECTOR][FAIL] {e}")
        return []


# ─────────────────────────────────────────────
# BM25 SEARCH
# ─────────────────────────────────────────────
def _bm25_search(query: str, base_docs: List[Dict], k: int) -> List[Dict]:

    if not bm25_rank or not base_docs:
        return []

    try:
        ranked = bm25_rank(query, base_docs, top_k=k)

        docs = []
        for d in ranked:
            if not _validate_doc(d):
                continue

            docs.append({
                "content": d.get("content"),
                "source": d.get("source") or d.get("document_name"),
                "bm25_score": _safe_float(d.get("bm25_score", 0.0)),
                "priority": _safe_float(d.get("priority", 0.0)),
            })

        return docs

    except Exception as e:
        logger.error(f"[HYBRID][BM25][FAIL] {e}")
        return []


# ─────────────────────────────────────────────
# PRIORITY BOOST
# ─────────────────────────────────────────────
def _apply_priority(docs: List[Dict]) -> List[Dict]:
    for d in docs:
        base = _safe_float(d.get("fusion_score", d.get("score", 0.0)))
        priority = _safe_float(d.get("priority", 0.0))
        d["final_score"] = base + (priority * PRIORITY_BOOST)
    return docs


# ─────────────────────────────────────────────
# PERSONALIZATION
# ─────────────────────────────────────────────
def _apply_personalization(docs: List[Dict], context: str) -> List[Dict]:

    if not context:
        return docs

    ctx_tokens = set(context.split())

    for d in docs:
        text = d.get("content", "")
        tokens = set(text.split())

        if not tokens:
            continue

        overlap = len(tokens & ctx_tokens) / max(len(ctx_tokens), 1)

        d["final_score"] += overlap * PERSONALIZATION_WEIGHT

    return docs


# ─────────────────────────────────────────────
# DEDUPLICATION
# ─────────────────────────────────────────────
def _deduplicate(docs: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for d in docs:
        key = (d.get("source"), d.get("content", "")[:120])

        if key in seen:
            continue

        seen.add(key)
        unique.append(d)

    return unique


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    user_id: Optional[str] = None,
    final_k: int = DEFAULT_K
) -> List[Dict]:

    if not query:
        return []

    start = time.time()

    try:
        query = _clean(query)

        # ───────── CONTEXT
        context = ""
        if user_id:
            rows = fetch_table("user_memory", filters={"user_id": user_id}, limit=10)
            context = " ".join(r.get("content", "") for r in rows if r.get("content"))

        expanded = f"{query} {context}".strip()[:MAX_QUERY_LENGTH]

        # ───────── VECTOR
        vector_docs = _vector_search(expanded, final_k)

        # ───────── BM25 (on same docs for stability)
        bm25_docs = _bm25_search(expanded, vector_docs, final_k)

        # ───────── FUSION
        if fuse_results:
            fused = fuse_results(vector_docs, bm25_docs, top_k=final_k)
        else:
            fused = vector_docs  # safe fallback

        # ───────── PRIORITY + PERSONALIZATION
        fused = _apply_priority(fused)
        fused = _apply_personalization(fused, context)

        # ───────── SORT
        fused.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        # ───────── DEDUP
        fused = _deduplicate(fused)

        # ───────── FINAL FILTER
        results = []
        for d in fused[:final_k]:

            score = _safe_float(d.get("final_score", 0.0))

            if score < MIN_SCORE:
                continue

            results.append({
                "content": d.get("content")[:MAX_DOC_LENGTH],
                "source": d.get("source"),
                "score": round(score, 4),
                "priority": d.get("priority", 0),
            })

        latency = (time.time() - start) * 1000
        logger.info(f"[HYBRID][SUCCESS] {len(results)} docs | {latency:.1f}ms")

        return results

    except Exception as e:
        logger.error(f"[HYBRID][FAIL] {e}")
        return []


# ─────────────────────────────────────────────
# ADAPTIVE
# ─────────────────────────────────────────────
def adaptive_retrieve(query: str, severity: float, user_id=None):

    if severity < 0.4:
        k = 3
    elif severity < 0.7:
        k = 5
    else:
        k = 8

    return hybrid_retrieve(query, user_id, final_k=k)