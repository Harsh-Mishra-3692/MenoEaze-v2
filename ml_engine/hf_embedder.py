# hf_embedder.py — PRODUCTION v7 (L7/L9 HARDENED | RAILWAY SAFE)

import logging
import numpy as np
import torch
import threading
import time
import hashlib
from typing import List, Union

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

logger = logging.getLogger("menoeaze.embedder")


class HFEmbedder:
    """
    L7/L9 Production Embedder:
    - Non-blocking lazy load
    - Strict timeout control
    - Instant fallback mode (Railway safe)
    - Cache + deterministic output
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EXPECTED_DIM = 384
    MAX_BATCH_SIZE = 64
    MAX_TEXT_LEN = 2000
    CACHE_SIZE = 512

    LOAD_TIMEOUT = 6  # 🔴 critical for Railway

    def __init__(self):
        self._model = None
        self._loading = False
        self._lock = threading.Lock()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache = {}

    # ─────────────────────────────────────────────
    # SAFE MODEL LOAD (NON-BLOCKING)
    # ─────────────────────────────────────────────
    def _load_model_async(self):

        def _load():
            try:
                logger.info(f"[Embedder] Loading model on {self.device}")

                model = SentenceTransformer(self.MODEL_NAME, device=self.device)

                model.encode(["warmup"], convert_to_numpy=True)

                self._model = model
                logger.info("[Embedder] Model ready")

            except Exception as e:
                logger.error(f"[Embedder] Load failed: {e}")
                self._model = None

            finally:
                self._loading = False

        if self._loading:
            return

        self._loading = True
        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def _ensure_model(self):
        if self._model is not None:
            return True

        if SentenceTransformer is None:
            logger.warning("[Embedder] SentenceTransformer not available")
            return False

        # start async load
        self._load_model_async()

        # wait limited time only
        start = time.time()
        while self._loading and (time.time() - start < self.LOAD_TIMEOUT):
            time.sleep(0.1)

        return self._model is not None

    # ─────────────────────────────────────────────
    # HASH
    # ─────────────────────────────────────────────
    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # ─────────────────────────────────────────────
    # SANITIZE
    # ─────────────────────────────────────────────
    def _sanitize(self, texts: List[str]) -> List[str]:
        out = []
        for t in texts:
            if not isinstance(t, str):
                out.append("")
                continue
            t = t.strip()
            out.append(t[:self.MAX_TEXT_LEN] if t else "")
        return out

    # ─────────────────────────────────────────────
    # ZERO VECTOR
    # ─────────────────────────────────────────────
    def _zero_vector(self):
        return np.zeros(self.EXPECTED_DIM, dtype=np.float32)

    # ─────────────────────────────────────────────
    # EMBED
    # ─────────────────────────────────────────────
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.zeros((1, self.EXPECTED_DIM), dtype=np.float32)

        texts = self._sanitize(texts)

        # ───────── CACHE FIRST ─────────
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

        # ───────── TRY MODEL ─────────
        model_ready = self._ensure_model()

        if model_ready and uncached:
            try:
                with torch.inference_mode():
                    vecs = self._model.encode(
                        uncached,
                        batch_size=min(len(uncached), self.MAX_BATCH_SIZE),
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    ).astype(np.float32)

                for i, v in enumerate(vecs):
                    idx = uncached_idx[i]

                    if (
                        v.ndim != 1
                        or v.shape[0] != self.EXPECTED_DIM
                        or not np.isfinite(v).all()
                    ):
                        v = self._zero_vector()

                    embeddings[idx] = v

                    if len(self._cache) >= self.CACHE_SIZE:
                        self._cache.pop(next(iter(self._cache)))

                    self._cache[self._hash(texts[idx])] = v

            except Exception as e:
                logger.error(f"[Embedder] Encode failed: {e}")

        # ───────── FALLBACK ─────────
        for i, e in enumerate(embeddings):
            if e is None:
                embeddings[i] = self._zero_vector()

        return np.vstack(embeddings)

    # ─────────────────────────────────────────────
    # SINGLE
    # ─────────────────────────────────────────────
    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
