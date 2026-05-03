# bm25_index.py — FINAL ELITE v3 (STABLE + CACHED + FUSION-READY)

import math
import re
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from functools import lru_cache

# BM25 PARAMETERS
K1 = 1.5
B = 0.75
EPSILON = 1e-9

# lightweight stopwords (minimal to avoid over-filtering medical terms)
STOPWORDS = {
    "the", "is", "and", "a", "an", "of", "to", "in", "on", "for", "with"
}

MAX_DOC_LENGTH = 2000
MAX_QUERY_TERMS = 50


# ─────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []

    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)

    # remove stopwords + limit pathological input
    tokens = [t for t in tokens if t not in STOPWORDS]

    return tokens[:MAX_DOC_LENGTH]


# ─────────────────────────────────────────────
# SAFE HASH FOR CACHING
# ─────────────────────────────────────────────
def _hash_corpus(corpus: List[str]) -> int:
    try:
        return hash(tuple(corpus))
    except Exception:
        return id(corpus)


# ─────────────────────────────────────────────
# BM25 INDEX
# ─────────────────────────────────────────────
class BM25Index:
    def __init__(self, documents: List[str]):
        self.documents = documents or []
        self.doc_tokens = []
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avg_doc_len = 0.0
        self.N = len(self.documents)

        if self.N == 0:
            return

        self._build()

    def _build(self):
        df = defaultdict(int)

        for doc in self.documents:
            tokens = tokenize(doc)

            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))

            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)

            for token in freqs:
                df[token] += 1

        self.avg_doc_len = sum(self.doc_len) / (self.N + EPSILON)

        for token, freq in df.items():
            # stable IDF
            self.idf[token] = math.log(
                (self.N - freq + 0.5) / (freq + 0.5 + EPSILON) + 1
            )

    def score(self, query: str) -> List[Tuple[int, float]]:
        if not query or self.N == 0:
            return []

        query_tokens = tokenize(query)[:MAX_QUERY_TERMS]
        if not query_tokens:
            return []

        scores = [0.0] * self.N

        # query term frequency weighting
        q_freq = Counter(query_tokens)

        for q, qf in q_freq.items():

            if q not in self.idf:
                continue

            idf = self.idf[q]

            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(q, 0)
                if f == 0:
                    continue

                dl = self.doc_len[i]

                denom = f + K1 * (1 - B + B * dl / (self.avg_doc_len + EPSILON))
                score = idf * ((f * (K1 + 1)) / (denom + EPSILON))

                # incorporate query frequency
                scores[i] += score * (1 + math.log(1 + qf))

        return list(enumerate(scores))


# ─────────────────────────────────────────────
# CACHED INDEX BUILDER
# ─────────────────────────────────────────────
@lru_cache(maxsize=8)
def _get_index(corpus_hash: int, corpus_tuple: tuple) -> BM25Index:
    return BM25Index(list(corpus_tuple))


# ─────────────────────────────────────────────
# NORMALIZATION (FOR FUSION)
# ─────────────────────────────────────────────
def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []

    mn, mx = min(scores), max(scores)

    if abs(mx - mn) < EPSILON:
        return [1.0 for _ in scores]

    return [(s - mn) / (mx - mn + EPSILON) for s in scores]


# ─────────────────────────────────────────────
# HELPER FOR STRUCTURED DOCS
# ─────────────────────────────────────────────
def bm25_rank(
    query: str,
    documents: List[Dict],
    text_key: str = "content",
    top_k: int = 10
) -> List[Dict]:

    if not documents or not query:
        return []

    corpus = [
        (doc.get(text_key) or "")[:MAX_DOC_LENGTH]
        for doc in documents
    ]

    corpus_tuple = tuple(corpus)
    corpus_hash = _hash_corpus(corpus)

    try:
        index = _get_index(corpus_hash, corpus_tuple)
    except Exception:
        # fallback (no cache)
        index = BM25Index(corpus)

    scored = index.score(query)

    if not scored:
        return []

    raw_scores = [s for _, s in scored]
    norm_scores = _normalize(raw_scores)

    results = []
    for (idx, raw), norm in zip(scored, norm_scores):

        if idx >= len(documents):
            continue

        doc = documents[idx].copy()

        doc["bm25_score"] = float(norm)
        doc["_bm25_raw"] = float(raw)  # debug/internal

        results.append(doc)

    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    return results[:top_k]