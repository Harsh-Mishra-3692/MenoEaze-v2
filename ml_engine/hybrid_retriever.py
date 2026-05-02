# hybrid_retriever.py — FINAL ELITE (CLINICAL + ROBUST + PRIORITY-AWARE)

import logging
import re
import time
from typing import List, Dict, Any, Optional

from ml_engine.hf_embedder import HFEmbedder
from ml_engine.db_client import call_rpc, fetch_table

try:
    from ml_engine.research.bm25_index import BM25Index
except:
    BM25Index = None

try:
    from ml_engine.research.fusion import hybrid_fusion
except:
    hybrid_fusion = None

logger = logging.getLogger("menoeaze.hybrid")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_K = 8
MAX_DOC_LENGTH = 1000

PERSONALIZATION_WEIGHT = 0.15
PRIORITY_BOOST = 0.25
MIN_SCORE = 0.05

_embedder = None
_bm25 = None


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            _embedder = HFEmbedder()
        except Exception:
            _embedder = None
    return _embedder


def _get_bm25():
    global _bm25

    if BM25Index is None:
        return None

    if _bm25:
        return _bm25

    try:
        docs = fetch_table("medical_documents", limit=1000)
        bm25 = BM25Index()
        bm25.build(docs)
        _bm25 = bm25
    except:
        _bm25 = None

    return _bm25


# ─────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────
def _clean(text: str):
    try:
        return re.sub(r"\s+", " ", text.lower()).strip()[:300]
    except:
        return ""


# ─────────────────────────────────────────────
# NORMALIZATION
# ─────────────────────────────────────────────
def _normalize_scores(docs, key):
    vals = [d.get(key, 0.0) for d in docs]
    if not vals:
        return docs

    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-6:
        return docs

    for d in docs:
        d[key] = (d.get(key, 0.0) - mn) / (mx - mn)

    return docs


# ─────────────────────────────────────────────
# VECTOR SEARCH
# ─────────────────────────────────────────────
def _vector_search(query, k):
    embedder = _get_embedder()
    if embedder is None:
        return []

    try:
        vec = embedder.embed([query])[0]

        if not vec.any():
            return []

        return call_rpc(
            "match_medical_documents",
            {
                "query_embedding": vec.tolist(),
                "match_count": k,
            }
        ) or []

    except Exception as e:
        logger.error(f"[Hybrid] vector failed: {e}")
        return []


# ─────────────────────────────────────────────
# BM25 SEARCH
# ─────────────────────────────────────────────
def _bm25_search(query, k):
    bm25 = _get_bm25()
    if not bm25:
        return []

    try:
        return bm25.search(query, top_k=k) or []
    except:
        return []


# ─────────────────────────────────────────────
# SAFE FUSION
# ─────────────────────────────────────────────
def _fuse(vector_docs, keyword_docs, k):

    if hybrid_fusion:
        try:
            return hybrid_fusion(vector_docs, keyword_docs, top_k=k)
        except:
            pass

    # fallback: simple merge
    combined = {d["id"]: d for d in vector_docs}
    for d in keyword_docs:
        if d["id"] not in combined:
            combined[d["id"]] = d

    return list(combined.values())[:k]


# ─────────────────────────────────────────────
# PRIORITY BOOST (CLINICAL BOOKS)
# ─────────────────────────────────────────────
def _apply_priority(docs):
    for d in docs:
        base = d.get("similarity", 0.0)
        priority = d.get("priority", 0.0)
        d["score"] = base + (priority * PRIORITY_BOOST)
    return docs


# ─────────────────────────────────────────────
# PERSONALIZATION
# ─────────────────────────────────────────────
def _apply_personalization(docs, context):

    if not context:
        return docs

    ctx_tokens = set(context.split())

    for d in docs:
        text = d.get("content", "")
        tokens = set(text.split())

        overlap = len(tokens & ctx_tokens) / max(len(ctx_tokens), 1)

        d["score"] += overlap * PERSONALIZATION_WEIGHT

    return docs


# ─────────────────────────────────────────────
# DIVERSITY FILTER
# ─────────────────────────────────────────────
def _deduplicate(docs):
    seen = set()
    unique = []

    for d in docs:
        key = (d.get("document_name"), d.get("content")[:100])

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
):

    if not query:
        return []

    start = time.time()

    try:
        query = _clean(query)

        context = ""
        if user_id:
            rows = fetch_table("user_memory", filters={"user_id": user_id}, limit=10)
            context = " ".join(r.get("content", "") for r in rows)

        expanded = f"{query} {context}"[:500]

        vector_docs = _vector_search(expanded, final_k)
        keyword_docs = _bm25_search(expanded, final_k)

        docs = _fuse(vector_docs, keyword_docs, final_k)

        docs = _normalize_scores(docs, "similarity")
        docs = _apply_priority(docs)
        docs = _apply_personalization(docs, context)

        docs.sort(key=lambda x: x.get("score", 0), reverse=True)

        docs = _deduplicate(docs)

        results = []
        for d in docs[:final_k]:

            content = d.get("content", "")
            if not content:
                continue

            results.append({
                "content": content[:MAX_DOC_LENGTH],
                "source": d.get("document_name"),
                "score": round(d.get("score", 0), 4),
                "priority": d.get("priority", 0),
            })

        latency = (time.time() - start) * 1000

        logger.info(f"[Hybrid] {len(results)} docs | {latency:.1f}ms")

        return results

    except Exception as e:
        logger.error(f"[Hybrid] failed: {e}")
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