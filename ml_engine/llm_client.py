# llm_client.py — PRODUCTION FIXED (RAILWAY SAFE + NO CRASH)

import os
import time
import logging
from typing import Dict, Any

from groq import Groq

logger = logging.getLogger("menoeaze.llm")


class LLMClient:

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            logger.error("❌ GROQ_API_KEY missing → LLM disabled")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                logger.error(f"❌ Groq init failed: {e}")
                self.client = None

        self.model = "llama3-70b-8192"
        self.temperature = 0.4
        self.max_tokens = 600

    # ───────── CORE SAFE ─────────
    def generate(self, prompt: str) -> Dict[str, Any]:

        if not prompt:
            return {"text": "", "fallback": True}

        if not self.client:
            logger.warning("⚠️ LLM client unavailable → fallback")
            return {
                "text": "Please consult a healthcare professional.",
                "fallback": True
            }

        try:
            start = time.time()

            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            text = res.choices[0].message.content.strip()

            latency = time.time() - start

            if not text:
                logger.warning("⚠️ Empty LLM response")
                return {
                    "text": "Please consult a healthcare professional.",
                    "fallback": True
                }

            return {
                "text": text,
                "latency": round(latency, 2),
                "fallback": False
            }

        except Exception as e:
            logger.error(f"❌ LLM failed: {e}")

            return {
                "text": "Please consult a healthcare professional.",
                "fallback": True
            }

    # ───────── SAFE STRING ─────────
    def safe_generate(self, prompt: str) -> str:
        res = self.generate(prompt)
        return res.get("text", "")

    # ───────── HEALTH ─────────
    def health_check(self) -> bool:
        if not self.client:
            return False

        try:
            res = self.generate("Say OK")
            return "ok" in res.get("text", "").lower()
        except:
            return False
