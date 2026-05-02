# hf_embedder.py — FINAL HARDENED (PRODUCTION + FAULT-TOLERANT)

import logging
import numpy as np
import torch
import threading
import time
import hashlib
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("menoeaze.embedder")


class HFEmbedder:
    """
    Production-grade embedding system:
    - Thread-safe lazy loading
    - GPU fallback → CPU
    - Deterministic outputs
    - Input alignment guaranteed
    - Safe failure fallback (no crashes)
    - Lightweight caching
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EXPECTED_DIM = 384
    MAX_BATCH_SIZE = 64
    MAX_TEXT_LEN = 2000
    CACHE_SIZE = 512

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache = {}

    # ─────────────────────────────────────────────
    # SAFE MODEL LOAD (THREAD SAFE + FALLBACK)
    # ─────────────────────────────────────────────
    def _load_model(self):
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                logger.info(f"[Embedder] Loading model on {self.device}")
                self._model = SentenceTransformer(self.MODEL_NAME, device=self.device)

                # warmup
                self._model.encode(
                    ["warmup"],
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

                logger.info("[Embedder] Model ready")

            except Exception as e:
                logger.error(f"[Embedder] GPU load failed, retry CPU: {e}")

                try:
                    self.device = "cpu"
                    self._model = SentenceTransformer(self.MODEL_NAME, device="cpu")

                    self._model.encode(["warmup"], convert_to_numpy=True)
                    logger.info("[Embedder] CPU fallback ready")

                except Exception as e2:
                    logger.critical(f"[Embedder] Model load failed: {e2}")
                    self._model = None

    # ─────────────────────────────────────────────
    # HASH (CACHE KEY)
    # ─────────────────────────────────────────────
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # ─────────────────────────────────────────────
    # SANITIZE (NO DROP, ALIGN SAFE)
    # ─────────────────────────────────────────────
    def _sanitize(self, texts: List[str]) -> List[str]:
        clean = []

        for t in texts:
            if not isinstance(t, str):
                clean.append("")
                continue

            t = t.strip()
            if not t:
                clean.append("")
                continue

            clean.append(t[:self.MAX_TEXT_LEN])

        return clean

    # ─────────────────────────────────────────────
    # SAFE ZERO VECTOR (FALLBACK)
    # ─────────────────────────────────────────────
    def _zero_vector(self):
        return np.zeros(self.EXPECTED_DIM, dtype=np.float32)

    # ─────────────────────────────────────────────
    # EMBED
    # ─────────────────────────────────────────────
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:

        if texts is None:
            raise ValueError("Input is None")

        if isinstance(texts, str):
            texts = [texts]

        if not isinstance(texts, list) or not texts:
            raise ValueError("Invalid input")

        self._load_model()

        if self._model is None:
            logger.error("[Embedder] Model unavailable — fallback to zeros")
            return np.vstack([self._zero_vector() for _ in texts])

        texts = self._sanitize(texts)

        # ── CACHE LOOKUP ───────────────────────
        embeddings = []
        uncached = []
        uncached_idx = []

        for i, t in enumerate(texts):
            key = self._hash(t)

            if key in self._cache:
                embeddings.append(self._cache[key])
            else:
                embeddings.append(None)
                uncached.append(t)
                uncached_idx.append(i)

        # ── COMPUTE UNCACHED ───────────────────
        if uncached:
            try:
                with torch.inference_mode():

                    for i in range(0, len(uncached), self.MAX_BATCH_SIZE):
                        batch = uncached[i:i + self.MAX_BATCH_SIZE]

                        vecs = self._model.encode(
                            batch,
                            batch_size=len(batch),
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )

                        vecs = vecs.astype(np.float32)

                        for j, v in enumerate(vecs):
                            if (
                                v.ndim != 1
                                or v.shape[0] != self.EXPECTED_DIM
                                or not np.isfinite(v).all()
                            ):
                                v = self._zero_vector()

                            idx = uncached_idx[i + j]
                            embeddings[idx] = v

                            # cache insert
                            if len(self._cache) >= self.CACHE_SIZE:
                                self._cache.pop(next(iter(self._cache)))
                            self._cache[self._hash(texts[idx])] = v

            except Exception as e:
                logger.error(f"[Embedder] Embedding failed: {e}")

                # fallback for uncached
                for idx in uncached_idx:
                    embeddings[idx] = self._zero_vector()

        # ── FINAL STACK ───────────────────────
        try:
            result = np.vstack(embeddings)

            if result.ndim != 2 or result.shape[1] != self.EXPECTED_DIM:
                raise ValueError("Invalid final embedding shape")

            return result

        except Exception as e:
            logger.error(f"[Embedder] Final assembly failed: {e}")
            return np.vstack([self._zero_vector() for _ in texts])

    # ─────────────────────────────────────────────
    # FAST SINGLE
    # ─────────────────────────────────────────────
    def embed_one(self, text: str) -> np.ndarray:
        try:
            return self.embed([text])[0]
        except Exception:
            return self._zero_vector()