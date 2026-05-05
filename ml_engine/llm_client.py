# llm_client.py — PRODUCTION v5 (STABLE + FALLBACK SAFE)

import os
import time
import logging
from typing import Dict, Any, Optional

from groq import Groq

logger = logging.getLogger("menoeaze.llm")


class LLMClient:

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            logger.error("❌ GROQ_API_KEY missing")

        self.client = Groq(api_key=self.api_key)

        self.model = "llama3-70b-8192"
        self.temperature = 0.4
        self.max_tokens = 600

    # ───────── CORE ─────────
    def generate(self, prompt: str) -> Dict[str, Any]:

        if not prompt:
            raise ValueError("Empty prompt")

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

            return {
                "text": text,
                "latency": round(latency, 2)
            }

        except Exception as e:
            logger.error(f"LLM failed: {e}")
            raise

    # ───────── SAFE FALLBACK ─────────
    def safe_generate(self, prompt: str) -> str:
        try:
            return self.generate(prompt)["text"]
        except:
            return "I'm unable to generate a detailed response right now. Please consult a healthcare professional."

    # ───────── HEALTH ─────────
    def health_check(self) -> bool:
        try:
            return bool(self.generate("ping")["text"])
        except:
            return False
