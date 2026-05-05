# llm_client.py — PRODUCTION v6 (L7/L9 HARDENED)

import os
import time
import logging
from typing import Dict, Any, Optional

from groq import Groq

logger = logging.getLogger("menoeaze.llm")


# ───────── CONFIG ─────────
DEFAULT_MODEL = "llama3-70b-8192"
DEFAULT_TEMP = 0.4
DEFAULT_MAX_TOKENS = 600

MAX_RETRIES = 2
BACKOFF_BASE = 1.6
TIMEOUT_GUARD = 20  # soft timeout tracking

CIRCUIT_FAIL_THRESHOLD = 5
CIRCUIT_RESET_SECONDS = 60


# ───────── CLIENT ─────────
class LLMClient:

    def __init__(self):

        self.api_key = os.getenv("GROQ_API_KEY")

        self.model = DEFAULT_MODEL
        self.temperature = DEFAULT_TEMP
        self.max_tokens = DEFAULT_MAX_TOKENS

        self._client: Optional[Groq] = None

        # circuit breaker state
        self._fail_count = 0
        self._last_fail_time = 0

        if not self.api_key:
            logger.error("❌ GROQ_API_KEY missing → LLM disabled")
        else:
            try:
                self._client = Groq(api_key=self.api_key)
                logger.info("✅ LLM initialized")
            except Exception as e:
                logger.error(f"❌ Groq init failed: {e}")
                self._client = None

    # ───────── CIRCUIT BREAKER ─────────
    def _circuit_open(self) -> bool:
        if self._fail_count < CIRCUIT_FAIL_THRESHOLD:
            return False

        if time.time() - self._last_fail_time > CIRCUIT_RESET_SECONDS:
            logger.info("🔁 Circuit reset")
            self._fail_count = 0
            return False

        return True

    # ───────── CORE GENERATION ─────────
    def generate(self, prompt: str) -> Dict[str, Any]:

        if not prompt or not prompt.strip():
            return {
                "text": "",
                "fallback": True,
                "error": "empty_prompt"
            }

        if not self._client:
            logger.warning("⚠️ LLM unavailable → fallback")
            return self._fallback("no_client")

        if self._circuit_open():
            logger.warning("⚠️ Circuit open → skipping LLM")
            return self._fallback("circuit_open")

        attempt = 0

        while attempt <= MAX_RETRIES:
            try:
                start = time.time()

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                latency = time.time() - start

                text = (
                    response.choices[0].message.content.strip()
                    if response and response.choices
                    else ""
                )

                if not text:
                    raise ValueError("empty_response")

                # reset circuit on success
                self._fail_count = 0

                return {
                    "text": text,
                    "latency": round(latency, 2),
                    "fallback": False
                }

            except Exception as e:
                attempt += 1
                self._fail_count += 1
                self._last_fail_time = time.time()

                wait = min(5, BACKOFF_BASE ** attempt)

                logger.warning(
                    f"❌ LLM attempt {attempt} failed: {e} | retry in {wait:.2f}s"
                )

                time.sleep(wait)

        # final failure
        return self._fallback("max_retries_exceeded")

    # ───────── SAFE STRING ─────────
    def safe_generate(self, prompt: str) -> str:
        res = self.generate(prompt)
        return res.get("text") or self._safe_message()

    # ───────── FALLBACK ─────────
    def _fallback(self, reason: str) -> Dict[str, Any]:
        logger.error(f"⚠️ LLM fallback triggered: {reason}")

        return {
            "text": self._safe_message(),
            "fallback": True,
            "error": reason
        }

    # ───────── SAFE MESSAGE ─────────
    def _safe_message(self) -> str:
        return (
            "Based on the available information, your symptoms may be related "
            "to hormonal changes. For accurate evaluation and personalized care, "
            "please consult a qualified healthcare professional."
        )

    # ───────── HEALTH CHECK ─────────
    def health_check(self) -> bool:
        if not self._client:
            return False

        try:
            res = self.generate("Respond with: OK")
            return "ok" in res.get("text", "").lower()
        except Exception:
            return False
