# hybrid_retriever.py — PRODUCTION v4 (RAILWAY HARDENED)

import logging
import re
import time
from typing import List, Dict, Optional

from ml_engine.hf_embedder import HFEmbedder
from ml_engine.db_client import call_rpc, fetch_table

logger = logging.getLogger("menoeaze.hybrid")

DEFAULT_K = 6
MAX_QUERY_LENGTH = 500
MAX_DOC_LENGTH = 800

_embedder = None


# ───────── SAFE FALLBACK KNOWLEDGE ─────────
STATIC_FALLBACK_DOCS = [
    {
        "content": "Hot flashes are a common symptom of menopause caused by hormonal changes. They can be managed with lifestyle changes, hydration, and stress reduction.",
        "source": "fallback_medical"
    },
    {
        "content": "Headaches during menopause may be linked to hormonal fluctuations, stress, or sleep issues. Regular sleep and hydration can help.",
        "source": "fallback_medical"
    },
    {
        "content": "Menopause symptoms include hot flashes, night sweats, mood changes, sleep disturbances, and fatigue.",
        "source": "fallback_medical"
    }
]


# ───────── EMBEDDER ─────────
def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            _embedder = HFEmbedder()
        except Exception as e:
            logger.warning(f"Embedder failed: {e}")
            _embedder = None
    return _embedder


# ───────── CLEAN ─────────
def _clean(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip()[:MAX_QUERY_LENGTH]


# ───────── HARD FALLBACK (NEVER EMPTY) ─────────
def _hard_fallback(k: int):
    docs = []

    for d in STATIC_FALLBACK_DOCS[:k]:
        docs.append({
            "content": d["content"],
            "source": d["source"],
            "document_name": d["source"],
            "score": 0.4,
            "priority": 0.0,
        })

    return docs


# ───────── VECTOR SEARCH ─────────
def _vector_search(query: str, k: int):

    embedder = _get_embedder()

    if embedder is None:
        return []

    try:
        vec = embedder.embed([query])[0]

        res = call_rpc(
            "match_rag_documents",
            {
                "query_embedding": vec.tolist(),
                "match_threshold": 0.05,
                "match_count": k,
            }
        )

        if not res:
            return []

        docs = []
        for r in res:
            if not r.get("content"):
                continue

            docs.append({
                "content": r.get("content")[:MAX_DOC_LENGTH],
                "source": r.get("document_name", "db"),
                "document_name": r.get("document_name", "db"),
                "score": float(r.get("similarity", 0.0)),
                "priority": float(r.get("priority", 0.0)),
            })

        return docs

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []


# ───────── TABLE FALLBACK ─────────
def _table_fetch(k: int):
    try:
        rows = fetch_table("medical_documents", limit=k)

        if not rows:
            return []

        docs = []
        for r in rows:
            if r.get("content"):
                docs.append({
                    "content": r["content"][:MAX_DOC_LENGTH],
                    "source": r.get("document_name", "table"),
                    "document_name": r.get("document_name", "table"),
                    "score": 0.3,
                    "priority": float(r.get("priority", 0.0)),
                })

        return docs

    except Exception as e:
        logger.error(f"Table fetch failed: {e}")
        return []


# ───────── MAIN ─────────
def hybrid_retrieve(query: str, user_id: Optional[str] = None, final_k: int = DEFAULT_K):

    start = time.time()

    if not query:
        return _hard_fallback(final_k)

    query = _clean(query)

    # 1. VECTOR
    docs = _vector_search(query, final_k)

    # 2. TABLE FALLBACK
    if not docs:
        docs = _table_fetch(final_k)

    # 3. FINAL HARD FALLBACK
    if not docs:
        logger.warning("Using STATIC fallback docs")
        docs = _hard_fallback(final_k)

    # 4. SORT SAFE
    docs.sort(key=lambda x: x.get("score", 0), reverse=True)

    latency = (time.time() - start) * 1000
    logger.info(f"[HYBRID] Returned {len(docs)} docs | {latency:.1f}ms")

    return docs[:final_k]


# ───────── ADAPTIVE ─────────
def adaptive_retrieve(query: str, severity: float, user_id=None):

    if severity < 0.4:
        k = 3
    elif severity < 0.7:
        k = 5
    else:
        k = 7

    return hybrid_retrieve(query, user_id, k)
