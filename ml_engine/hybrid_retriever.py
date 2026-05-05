# hybrid_retriever.py — PRODUCTION v6 (L7/L9 HARDENED, DEPLOYMENT SAFE)

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
MAX_TABLE_FETCH = 60

MIN_DOCS_THRESHOLD = 2

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
            logger.info("HFEmbedder loaded")
        except Exception as e:
            logger.error(f"❌ Embedder failed permanently: {e}")
            _embedder_failed = True
            _embedder = None

    return _embedder


# ───────── CLEAN ─────────
def _clean(q: str) -> str:
    if not isinstance(q, str):
        return ""
    return re.sub(r"\s+", " ", q.lower()).strip()[:MAX_QUERY_LENGTH]


# ───────── STATIC FALLBACK ─────────
def _hard_fallback(k: int):
    return [
        {
            "content": d["content"],
            "source": d["source"],
            "document_name": d["source"],
            "score": 0.25,
            "priority": 0.0,
        }
        for d in STATIC_FALLBACK_DOCS[:k]
    ]


# ───────── DEDUP ─────────
def _deduplicate(docs: List[Dict]) -> List[Dict]:
    seen = set()
    unique = []

    for d in docs:
        key = (d.get("content", "")[:120], d.get("source"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)

    return unique


# ───────── SCORE NORMALIZATION ─────────
def _normalize_scores(docs: List[Dict]) -> List[Dict]:
    if not docs:
        return docs

    scores = [d.get("score", 0.0) for d in docs]
    max_score = max(scores) if scores else 1.0

    if max_score == 0:
        return docs

    for d in docs:
        d["score"] = d.get("score", 0.0) / max_score

    return docs


# ───────── KEYWORD SEARCH ─────────
def _keyword_search(query: str, rows: List[Dict]):
    scored = []
    q_words = set(query.split())

    for r in rows:
        content = (r.get("content") or "").lower()
        if not content:
            continue

        matches = sum(1 for word in q_words if word in content)

        if matches > 0:
            scored.append({
                "content": content[:MAX_DOC_LENGTH],
                "source": r.get("document_name", "table"),
                "document_name": r.get("document_name", "table"),
                "score": float(matches),
                "priority": float(r.get("priority", 0.0)),
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# ───────── VECTOR SEARCH ─────────
def _vector_search(query: str, k: int):
    embedder = _get_embedder()

    if embedder is None:
        logger.warning("⚠️ Embedder unavailable → vector search skipped")
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
            content = r.get("content")
            if not content:
                continue

            docs.append({
                "content": content[:MAX_DOC_LENGTH],
                "source": r.get("document_name", "db"),
                "document_name": r.get("document_name", "db"),
                "score": float(r.get("similarity", 0.0)),
                "priority": float(r.get("priority", 0.0)),
            })

        return docs

    except Exception as e:
        logger.error(f"❌ Vector search failed: {e}")
        return []


# ───────── TABLE SEARCH ─────────
def _table_search(query: str, k: int):
    try:
        rows = fetch_table("medical_documents", limit=MAX_TABLE_FETCH)

        if not rows:
            logger.warning("⚠️ No rows returned from medical_documents")
            return []

        docs = _keyword_search(query, rows)

        if not docs:
            logger.warning("⚠️ Keyword search yielded no results")

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
    vector_docs = _vector_search(query, final_k)

    # 2️⃣ TABLE SEARCH
    table_docs = []
    if len(vector_docs) < MIN_DOCS_THRESHOLD:
        table_docs = _table_search(query, final_k)

    docs = vector_docs + table_docs

    # 3️⃣ FALLBACK
    if not docs:
        logger.warning("⚠️ Using static fallback (last resort)")
        docs = _hard_fallback(final_k)

    # 4️⃣ CLEANUP
    docs = _deduplicate(docs)
    docs = _normalize_scores(docs)

    # 5️⃣ SORT
    docs.sort(
        key=lambda x: x.get("score", 0) + 0.2 * x.get("priority", 0),
        reverse=True
    )

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
