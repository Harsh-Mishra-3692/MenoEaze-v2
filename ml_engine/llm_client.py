# llm_client.py — PRODUCTION v4 (RAILWAY SAFE + NO SILENT FALLBACK)

import os
import time
import logging
import uuid
from typing import Optional, Dict, Any

from ml_engine.config import CONFIG

try:
    from groq import Groq
except ImportError:
    raise ImportError("groq package not installed. Run: pip install groq")

logger = logging.getLogger("menoeaze.llm")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LLM_CFG = CONFIG["llm"]

MAX_RETRIES = 3
TIMEOUT_SECONDS = 25   # 🔥 increased for Railway
BACKOFF_BASE = 1.8

CIRCUIT_BREAK_THRESHOLD = 5
CIRCUIT_RESET_TIME = 60

MAX_PROMPT_CHARS = 12000


# ─────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────
class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        # 🔥 FIX: ENV-FIRST strategy (Railway compatible)
        self.api_key = (
            api_key
            or os.getenv("GROQ_API_KEY")
            or getattr(LLM_CFG, "api_key", None)
        )

        self.model = model or LLM_CFG.model
        self.temperature = LLM_CFG.temperature
        self.max_tokens = LLM_CFG.max_tokens

        # 🔥 visibility
        logger.info(f"[LLM] API KEY PRESENT: {bool(self.api_key)}")

        if not self.api_key:
            logger.error("[LLM] GROQ_API_KEY missing — LLM disabled")

        self._client = None

        # circuit breaker
        self._fail_count = 0
        self._last_fail_time = 0

        logger.info(f"[LLM] Initialized | model={self.model}")

    # ─────────────────────────────────────────
    # CLIENT INIT
    # ─────────────────────────────────────────
    def _get_client(self):
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")

        if self._client is None:
            self._client = Groq(api_key=self.api_key)

        return self._client

    # ─────────────────────────────────────────
    # CIRCUIT BREAKER
    # ─────────────────────────────────────────
    def _circuit_open(self) -> bool:
        if self._fail_count < CIRCUIT_BREAK_THRESHOLD:
            return False

        if time.time() - self._last_fail_time > CIRCUIT_RESET_TIME:
            self._fail_count = 0
            return False

        return True

    # ─────────────────────────────────────────
    # PROMPT SAFETY
    # ─────────────────────────────────────────
    def _sanitize_prompt(self, prompt: str) -> str:
        if not prompt:
            return ""
        return prompt[:MAX_PROMPT_CHARS]

    # ─────────────────────────────────────────
    # CORE GENERATION
    # ─────────────────────────────────────────
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        request_id = str(uuid.uuid4())[:8]

        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")

        if self._circuit_open():
            raise RuntimeError("LLM circuit breaker open")

        prompt = self._sanitize_prompt(prompt)

        temp = temperature if temperature is not None else self.temperature
        max_toks = max_tokens if max_tokens is not None else self.max_tokens

        attempt = 0

        while attempt < MAX_RETRIES:
            try:
                client = self._get_client()

                start = time.time()

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    max_tokens=max_toks,
                    timeout=TIMEOUT_SECONDS,
                )

                latency = time.time() - start

                text = response.choices[0].message.content.strip()

                if not text:
                    raise ValueError("Empty LLM response")

                self._fail_count = 0

                tokens_est = int(len(text) / 4)

                logger.info(
                    f"[LLM] success | req={request_id} | latency={latency:.2f}s | tokens≈{tokens_est}"
                )

                return {
                    "text": text,
                    "latency": round(latency, 3),
                    "tokens_est": tokens_est,
                    "request_id": request_id,
                }

            except Exception as e:
                attempt += 1
                self._fail_count += 1
                self._last_fail_time = time.time()

                wait = min(10, BACKOFF_BASE ** attempt)

                logger.warning(
                    f"[LLM] fail | req={request_id} | attempt={attempt} | error={str(e)} | retry={wait:.1f}s"
                )

                time.sleep(wait)

        # 🔥 CRITICAL CHANGE: NO FALLBACK
        raise RuntimeError("LLM failed after retries")

    # ─────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────
    def health_check(self) -> bool:
        try:
            res = self.generate("ping", max_tokens=5)
            return bool(res.get("text"))
        except Exception:
            return False
