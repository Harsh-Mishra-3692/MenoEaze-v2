# hybrid_retriever.py — PRODUCTION v5 (NO SILENT DEGRADATION)

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
_embedder_failed = False


# ───────── STATIC FALLBACK ─────────
STATIC_FALLBACK_DOCS = [
    {
        "content": "Hot flashes are a common symptom of menopause caused by hormonal changes. They can be managed with hydration, stress reduction, and lifestyle adjustments.",
        "source": "fallback_medical"
    },
    {
        "content": "Headaches during menopause may result from hormonal fluctuations, stress, or sleep disturbances.",
        "source": "fallback_medical"
    },
    {
        "content": "Common menopause symptoms include hot flashes, night sweats, mood changes, and sleep issues.",
        "source": "fallback_medical"
    }
]


# ───────── EMBEDDER ─────────
def _get_embedder():
    global _embedder, _embedder_failed

    if _embedder_failed:
        return None

    if _embedder is None:
        try:
            logger.info("Loading HFEmbedder...")
            _embedder = HFEmbedder()
            logger.info("HFEmbedder loaded successfully")
        except Exception as e:
            logger.error(f"❌ EMBEDDER FAILED: {e}")
            _embedder_failed = True
            _embedder = None

    return _embedder


# ───────── CLEAN ─────────
def _clean(q: str) -> str:
    return re.sub(r"\s+", " ", q.lower()).strip()[:MAX_QUERY_LENGTH]


# ───────── STATIC FALLBACK ─────────
def _hard_fallback(k: int):
    return [
        {
            "content": d["content"],
            "source": d["source"],
            "document_name": d["source"],
            "score": 0.2,
            "priority": 0.0,
        }
        for d in STATIC_FALLBACK_DOCS[:k]
    ]


# ───────── SIMPLE KEYWORD MATCH (NEW) ─────────
def _keyword_search(query: str, rows: List[Dict]):
    scored = []

    for r in rows:
        content = r.get("content", "").lower()
        if not content:
            continue

        score = sum(1 for word in query.split() if word in content)

        if score > 0:
            scored.append({
                "content": content[:MAX_DOC_LENGTH],
                "source": r.get("document_name", "table"),
                "document_name": r.get("document_name", "table"),
                "score": float(score),
                "priority": float(r.get("priority", 0.0)),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ───────── VECTOR SEARCH ─────────
def _vector_search(query: str, k: int):
    embedder = _get_embedder()

    if embedder is None:
        logger.warning("⚠️ Embedder unavailable → skipping vector search")
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
                "content": r["content"][:MAX_DOC_LENGTH],
                "source": r.get("document_name", "db"),
                "document_name": r.get("document_name", "db"),
                "score": float(r.get("similarity", 0.0)),
                "priority": float(r.get("priority", 0.0)),
            })

        return docs

    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        return []


# ───────── TABLE + KEYWORD ─────────
def _table_search(query: str, k: int):
    try:
        rows = fetch_table("medical_documents", limit=50)

        if not rows:
            logger.warning("⚠️ No rows in medical_documents")
            return []

        docs = _keyword_search(query, rows)

        if not docs:
            logger.warning("⚠️ Keyword search returned nothing")

        return docs[:k]

    except Exception as e:
        logger.error(f"❌ Table fetch failed: {e}")
        return []


# ───────── MAIN ─────────
def hybrid_retrieve(query: str, user_id: Optional[str] = None, final_k: int = DEFAULT_K):

    start = time.time()

    if not query:
        return _hard_fallback(final_k)

    query = _clean(query)

    # 1️⃣ VECTOR SEARCH
    docs = _vector_search(query, final_k)

    # 2️⃣ TABLE + KEYWORD SEARCH
    if len(docs) < 2:
        docs += _table_search(query, final_k)

    # 3️⃣ FINAL HARD FALLBACK
    if not docs:
        logger.warning("⚠️ USING STATIC FALLBACK (LAST RESORT)")
        docs = _hard_fallback(final_k)

    # SORT
    docs.sort(key=lambda x: x.get("score", 0), reverse=True)

    latency = (time.time() - start) * 1000
    logger.info(f"[HYBRID] {len(docs)} docs | {latency:.1f}ms")

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
