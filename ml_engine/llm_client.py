# llm_client.py — ELITE v3 (PRODUCTION + RESEARCH + HARDENED)

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
TIMEOUT_SECONDS = 12
BACKOFF_BASE = 1.8

CIRCUIT_BREAK_THRESHOLD = 5
CIRCUIT_RESET_TIME = 60

MAX_PROMPT_CHARS = 12000  # 🔥 safety


# ─────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────
class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.api_key = api_key or getattr(LLM_CFG, "api_key", None)
        self.model = model or LLM_CFG.model
        self.temperature = LLM_CFG.temperature
        self.max_tokens = LLM_CFG.max_tokens

        if not self.api_key:
            logger.warning("[LLM] Missing GROQ_API_KEY in CONFIG. LLM will run in degraded/fallback mode.")

        self._client = None

        # circuit breaker
        self._fail_count = 0
        self._last_fail_time = 0

        logger.info(f"[LLM] Initialized | model={self.model}")

    # ─────────────────────────────────────────
    # LAZY INIT
    # ─────────────────────────────────────────
    def _get_client(self):
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
            return self._fallback("empty_prompt", request_id)

        if self._circuit_open():
            logger.error(f"[LLM] circuit open | req={request_id}")
            return self._fallback("circuit_open", request_id)

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
                    timeout=TIMEOUT_SECONDS,  # 🔥 FIXED
                )

                latency = time.time() - start

                text = response.choices[0].message.content.strip()

                if not text:
                    raise ValueError("Empty LLM response")

                self._fail_count = 0

                tokens_est = int(len(text) / 4)  # 🔥 better heuristic

                logger.info(
                    f"[LLM] success | req={request_id} | latency={latency:.2f}s | tokens≈{tokens_est}"
                )

                return {
                    "text": text,
                    "latency": round(latency, 3),
                    "tokens_est": tokens_est,
                    "request_id": request_id,
                    "fallback": False,
                }

            except Exception as e:
                attempt += 1
                self._fail_count += 1
                self._last_fail_time = time.time()

                # 🔥 smarter backoff
                wait = min(10, BACKOFF_BASE ** attempt)

                logger.warning(
                    f"[LLM] fail | req={request_id} | attempt={attempt} | error={str(e)} | retry={wait:.1f}s"
                )

                time.sleep(wait)

        logger.error(f"[LLM] all retries failed | req={request_id}")

        return self._fallback("llm_failure", request_id)

    # ─────────────────────────────────────────
    # FALLBACK
    # ─────────────────────────────────────────
    def _fallback(self, reason: str, request_id: str) -> Dict[str, Any]:
        logger.error(f"[LLM] fallback | req={request_id} | reason={reason}")

        return {
            "text": (
                "I'm unable to generate a detailed response at the moment. "
                "Please consult a healthcare professional."
            ),
            "latency": 0.0,
            "tokens_est": 0,
            "request_id": request_id,
            "fallback": True,
        }

    # ─────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────
    def health_check(self) -> bool:
        try:
            res = self.generate("ping", max_tokens=5)
            return not res["fallback"]
        except Exception:
            return False