# ml_engine/research/bm25_index.py — ELITE v3 (ROBUST + RESEARCH-GRADE)

import math
import logging
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any

logger = logging.getLogger("menoeaze.bm25")

EPS = 1e-8


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents: List[Dict[str, Any]] = []
        self.tokenized_docs: List[List[str]] = []
        self.doc_freqs: List[Counter] = []
        self.idf: Dict[str, float] = {}
        self.avgdl: float = 0.0

    # ─────────────────────────────────────────
    # TOKENIZATION (IMPROVED)
    # ─────────────────────────────────────────
    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()

        return tokens

    # ─────────────────────────────────────────
    # BUILD INDEX
    # ─────────────────────────────────────────
    def build(self, documents: List[Dict[str, Any]]):
        """
        documents = [
            {"id": ..., "content": ..., "title": ...}
        ]
        """

        if not documents:
            logger.warning("[BM25] Empty document list")
            return

        self.documents = documents
        self.tokenized_docs = []
        self.doc_freqs = []

        df = defaultdict(int)
        doc_lens = []

        for doc in documents:
            content = doc.get("content", "")
            tokens = self._tokenize(content)

            self.tokenized_docs.append(tokens)

            counts = Counter(tokens)
            self.doc_freqs.append(counts)

            doc_lens.append(len(tokens))

            for word in counts:
                df[word] += 1

        total_docs = len(self.tokenized_docs)

        self.avgdl = sum(doc_lens) / max(total_docs, 1)

        # IDF calculation (BM25 standard)
        self.idf = {
            word: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))
            for word, freq in df.items()
        }

        logger.info(f"[BM25] Built index | docs={total_docs}")

    # ─────────────────────────────────────────
    # SCORE SINGLE DOC
    # ─────────────────────────────────────────
    def _score_doc(self, query_tokens: List[str], index: int) -> float:
        if index >= len(self.tokenized_docs):
            return 0.0

        doc_tokens = self.tokenized_docs[index]
        doc_freq = self.doc_freqs[index]
        doc_len = len(doc_tokens)

        if doc_len == 0:
            return 0.0

        score = 0.0

        for word in query_tokens:
            if word not in doc_freq:
                continue

            freq = doc_freq[word]
            idf = self.idf.get(word, 0.0)

            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl + EPS))

            score += idf * ((freq * (self.k1 + 1)) / (denom + EPS))

        return score

    # ─────────────────────────────────────────
    # NORMALIZATION
    # ─────────────────────────────────────────
    def _normalize(self, scores: List[float]) -> List[float]:
        if not scores:
            return scores

        min_s, max_s = min(scores), max(scores)

        if max_s == min_s:
            return [0.5] * len(scores)

        return [(s - min_s) / (max_s - min_s + EPS) for s in scores]

    # ─────────────────────────────────────────
    # SEARCH
    # ─────────────────────────────────────────
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not self.documents:
            return []

        query_tokens = self._tokenize(query)

        raw_scores = [
            self._score_doc(query_tokens, i)
            for i in range(len(self.tokenized_docs))
        ]

        norm_scores = self._normalize(raw_scores)

        results = []

        for i, score in enumerate(norm_scores):
            if score <= 0:
                continue

            doc = self.documents[i].copy()
            doc["bm25_score"] = float(score)

            results.append(doc)

        results.sort(key=lambda x: x["bm25_score"], reverse=True)

        logger.info(f"[BM25] Retrieved {len(results[:top_k])} docs")

        return results[:top_k]

    # ─────────────────────────────────────────
    # BATCH SEARCH (FOR RESEARCH)
    # ─────────────────────────────────────────
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, Any]]]:
        return [self.search(q, top_k=top_k) for q in queries]