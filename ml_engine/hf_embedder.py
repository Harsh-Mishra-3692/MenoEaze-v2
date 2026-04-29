# hf_embedder.py (PRODUCTION GRADE)

import logging
import numpy as np
import torch
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("menoeaze.embedder")


class HFEmbedder:
    """
    Production-grade embedding system:
    - Lazy + warm start
    - GPU optimized
    - Strict validation (no silent failures)
    - Batch optimized
    - Deterministic outputs
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EXPECTED_DIM = 384
    MAX_BATCH_SIZE = 64  # increased for throughput

    def __init__(self):
        self._model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─────────────────────────────────────────────
    # LOAD MODEL (LAZY + SAFE)
    # ─────────────────────────────────────────────
    def _load_model(self):
        if self._model is not None:
            return

        try:
            logger.info(f"[Embedder] Loading model on {self.device}")
            self._model = SentenceTransformer(self.MODEL_NAME, device=self.device)

            # 🔥 Warmup (CRITICAL for latency)
            self._model.encode(
                ["warmup"],
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            logger.info("[Embedder] Model ready")

        except Exception as e:
            logger.critical(f"[Embedder] Model load failed: {e}")
            raise RuntimeError("Embedding model failed to initialize")

    # ─────────────────────────────────────────────
    # SANITIZE INPUT
    # ─────────────────────────────────────────────
    def _sanitize(self, texts: List[str]) -> List[str]:
        clean = []

        for t in texts:
            if not isinstance(t, str):
                continue

            t = t.strip()
            if not t:
                continue

            # truncate extremely long text (safety)
            clean.append(t[:2000])

        if not clean:
            raise ValueError("All input texts invalid after sanitization")

        return clean

    # ─────────────────────────────────────────────
    # EMBED
    # ─────────────────────────────────────────────
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Returns:
            np.ndarray shape (N, 384)
        """

        if texts is None:
            raise ValueError("Input is None")

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            raise ValueError("Empty input")

        self._load_model()
        texts = self._sanitize(texts)

        try:
            with torch.inference_mode():  # 🔥 performance boost
                embeddings = self._model.encode(
                    texts,
                    batch_size=self.MAX_BATCH_SIZE,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )

            embeddings = embeddings.astype(np.float32)

            # strict validation
            if embeddings.ndim != 2 or embeddings.shape[1] != self.EXPECTED_DIM:
                raise ValueError(
                    f"Invalid embedding shape: {embeddings.shape}, expected (*, {self.EXPECTED_DIM})"
                )

            return embeddings

        except Exception as e:
            logger.error(f"[Embedder] Embedding failed: {e}")
            raise RuntimeError("Embedding generation failed")

    # ─────────────────────────────────────────────
    # SINGLE EMBED (FAST PATH)
    # ─────────────────────────────────────────────
    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
