# hybrid_retriever.py — ELITE v5 (PERSONALIZED + TRUE HYBRID)

import logging
import re
import time
from typing import List, Dict, Any, Optional

from ml_engine.hf_embedder import HFEmbedder
from ml_engine.db_client import call_rpc, fetch_table

from ml_engine.research.bm25_index import BM25Index
from ml_engine.research.fusion import hybrid_fusion

logger = logging.getLogger("menoeaze.hybrid")

# ─────────────────────────────────────────────
# CONFIG (NOW FLEXIBLE)
# ─────────────────────────────────────────────
DEFAULT_VECTOR_K = 8
DEFAULT_BM25_K = 8
DEFAULT_FINAL_K = 8

PERSONALIZATION_WEIGHT = 0.2
MIN_SCORE_THRESHOLD = 0.05

_embedder = None
_bm25 = None


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────
def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HFEmbedder()
    return _embedder


def _get_bm25():
    global _bm25

    if _bm25 is not None:
        return _bm25

    try:
        docs = fetch_table("medical_documents", limit=500)

        if not docs:
            return None

        bm25 = BM25Index()
        bm25.build(docs)
        _bm25 = bm25

        logger.info(f"[Hybrid] BM25 initialized | docs={len(docs)}")

    except Exception as e:
        logger.error(f"[Hybrid] BM25 init failed: {e}")
        _bm25 = None

    return _bm25


# ─────────────────────────────────────────────
# USER CONTEXT (NEW)
# ─────────────────────────────────────────────
def _get_user_context(user_id: str) -> str:
    """
    Pull past user info (chat/symptoms/preferences)
    """
    try:
        rows = fetch_table("user_memory", filters={"user_id": user_id}, limit=20)

        texts = [
            r.get("content", "")
            for r in rows if r.get("content")
        ]

        return " ".join(texts)[:500]

    except Exception:
        return ""


def _expand_query(query: str, user_context: str) -> str:
    if not user_context:
        return query

    return f"{query} {user_context}"


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:300]


def _tokenize(text: str):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


# ─────────────────────────────────────────────
# PERSONALIZATION SCORE (NEW)
# ─────────────────────────────────────────────
def _personalization_score(doc: Dict, user_context: str) -> float:
    if not user_context:
        return 0.0

    doc_tokens = set(_tokenize(doc.get("content", "")))
    user_tokens = set(_tokenize(user_context))

    if not doc_tokens or not user_tokens:
        return 0.0

    overlap = len(doc_tokens & user_tokens)
    return overlap / len(user_tokens)


# ─────────────────────────────────────────────
# VECTOR SEARCH
# ─────────────────────────────────────────────
def _vector_search(query: str, k: int) -> List[Dict[str, Any]]:
    try:
        embedder = _get_embedder()
        vec = embedder.embed([query])[0]

        res = call_rpc(
            "match_medical_documents",
            {
                "match_count": k,
                "query_embedding": vec.tolist(),
            }
        )

        return res or []

    except Exception as e:
        logger.error(f"[Hybrid] Vector search failed: {e}")
        return []


# ─────────────────────────────────────────────
# BM25 SEARCH
# ─────────────────────────────────────────────
def _bm25_search(query: str, k: int) -> List[Dict[str, Any]]:
    bm25 = _get_bm25()

    if bm25 is None:
        return []

    try:
        return bm25.search(query, top_k=k)
    except Exception as e:
        logger.error(f"[Hybrid] BM25 failed: {e}")
        return []


# ─────────────────────────────────────────────
# PERSONALIZATION RE-RANK (NEW)
# ─────────────────────────────────────────────
def _apply_personalization(docs: List[Dict], user_context: str):
    for d in docs:
        p_score = _personalization_score(d, user_context)
        base = d.get("fusion_score", d.get("similarity", 0.0))

        d["personalization_score"] = p_score
        d["final_score"] = (
            (1 - PERSONALIZATION_WEIGHT) * base +
            PERSONALIZATION_WEIGHT * p_score
        )

    docs.sort(key=lambda x: x["final_score"], reverse=True)

    return docs


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    user_id: Optional[str] = None,
    vector_k: int = DEFAULT_VECTOR_K,
    bm25_k: int = DEFAULT_BM25_K,
    final_k: int = DEFAULT_FINAL_K,
    return_debug: bool = False
) -> List[Dict[str, Any]]:

    if not query or not query.strip():
        return []

    start = time.time()

    try:
        query = _clean(query)

        # ── PERSONALIZATION
        user_context = _get_user_context(user_id) if user_id else ""
        expanded_query = _expand_query(query, user_context)

        # ── RETRIEVAL
        vector_docs = _vector_search(expanded_query, vector_k)
        keyword_docs = _bm25_search(expanded_query, bm25_k)

        # ── FUSION
        fused_docs = hybrid_fusion(
            vector_docs,
            keyword_docs,
            method="weighted",
            top_k=final_k
        )

        # ── PERSONALIZATION RE-RANK
        fused_docs = _apply_personalization(fused_docs, user_context)

        # ── FILTER
        fused_docs = [
            d for d in fused_docs
            if d.get("final_score", 0) >= MIN_SCORE_THRESHOLD
        ][:final_k]

        latency = (time.time() - start) * 1000

        logger.info(
            f"[Hybrid] final={len(fused_docs)} | "
            f"user={bool(user_id)} | latency={latency:.2f}ms"
        )

        if return_debug:
            return {
                "results": fused_docs,
                "vector_docs": vector_docs,
                "bm25_docs": keyword_docs,
                "user_context_used": bool(user_context),
                "latency_ms": round(latency, 2),
            }

        return fused_docs

    except Exception as e:
        logger.error(f"[Hybrid] pipeline failed: {e}")
        return []


# ─────────────────────────────────────────────
# STATE-CONDITIONED ADAPTIVE RETRIEVAL (Phase 4)
# ─────────────────────────────────────────────
# Severity-tier config: (top_k, query emphasis keywords)
_SEVERITY_TIERS = {
    "low":    (3, "lifestyle diet preventative wellness sleep hygiene"),
    "medium": (5, "symptom management lifestyle adjustments medical advice"),
    "high":   (7, "clinical intervention medical treatment urgent care severe symptoms"),
}


def adaptive_retrieve(
    query: str,
    severity: float,
    user_id: Optional[str] = None,
    return_debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Severity-conditioned retrieval: dynamically adjusts top-K and
    query emphasis based on the predicted severity.

    LOW  (< 0.4): k=3, prioritize lifestyle/diet/preventative
    MED  (0.4-0.7): k=5, balanced symptom-management + lifestyle
    HIGH (> 0.7): k=7, prioritize clinical/urgent care

    Falls back gracefully to empty list on any failure.
    """
    if not query or not query.strip():
        return []

    # Determine tier
    if severity < 0.4:
        tier = "low"
    elif severity < 0.7:
        tier = "medium"
    else:
        tier = "high"

    k, emphasis = _SEVERITY_TIERS[tier]

    # Expand query with severity-specific emphasis
    adjusted_query = f"{query} {emphasis}"

    try:
        docs = hybrid_retrieve(
            query=adjusted_query,
            user_id=user_id,
            vector_k=k,
            bm25_k=k,
            final_k=k,
            return_debug=return_debug,
        )

        logger.info(f"[Hybrid] adaptive_retrieve | tier={tier} k={k} docs={len(docs) if isinstance(docs, list) else '?'}")
        return docs

    except Exception as e:
        logger.error(f"[Hybrid] adaptive_retrieve failed: {e}. Graceful degradation.")
        return []