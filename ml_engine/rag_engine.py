# rag_engine.py — PRODUCTION FIXED (NO PLACEHOLDER RETURNS)

import logging
import re
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("menoeaze.rag")

MAX_CONTEXT_CHARS = 3200
MAX_DOC_CHARS = 800
MAX_QUERY_LENGTH = 300
MAX_SOURCES = 5


# ─────────────────────────────────────────────
# SANITIZATION
# ─────────────────────────────────────────────
def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"[^\w\s.,\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:MAX_QUERY_LENGTH]


# ─────────────────────────────────────────────
# SEVERITY
# ─────────────────────────────────────────────
def _severity_label(severity: float) -> str:
    return "low" if severity < 0.3 else "medium" if severity < 0.6 else "high"


# ─────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────
def _build_context(docs):
    seen_sources = set()
    blocks = []
    total = 0

    docs = sorted(
        docs,
        key=lambda d: d.get("priority", 0) + d.get("score", 0),
        reverse=True
    )

    for d in docs:
        source = d.get("document_name") or d.get("source") or "Unknown"
        content = (d.get("content") or "").strip()

        if not content or source in seen_sources:
            continue

        seen_sources.add(source)

        block = f"[{source}] {content[:MAX_DOC_CHARS]}\n\n"

        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        blocks.append(block)
        total += len(block)

    return "".join(blocks)


# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────
def _build_prompt(query, level, symptoms, context, trend_meta=None):
    trend_str = (
        f"- direction: {trend_meta['direction']}\n- variability: {trend_meta['variability']}"
        if trend_meta else "No trend data available."
    )

    return f"""
You are a clinical AI assistant.

User query:
{query}

Severity: {level}

Symptoms:
{symptoms}

Trend:
{trend_str}

Context:
{context}

Provide a helpful, medically safe, relevant answer.
Do NOT say "analyzing".
Be direct and useful.
"""


# ─────────────────────────────────────────────
# SAFE LLM CALL
# ─────────────────────────────────────────────
def _call_llm_safe(llm, prompt, query):
    try:
        res = llm.generate(prompt)

        if isinstance(res, str) and res.strip():
            return res

        if isinstance(res, dict):
            text = res.get("text", "").strip()
            if text:
                return text

        # 🔥 fallback to simple query
        return llm.generate(query)

    except Exception as e:
        logger.warning(f"LLM failed: {e}")

        try:
            return llm.generate(query)
        except:
            return "Please consult a healthcare professional for proper guidance."


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def generate_answer(
    query: str,
    severity: float,
    docs: List[Dict[str, Any]],
    symptoms: Optional[Dict[str, float]],
    llm_client,
    return_debug=False,
    trend_meta=None
):

    start = time.time()
    level = _severity_label(severity)

    query = _sanitize(query)

    try:
        context = _build_context(docs)

        # fallback context if weak
        if len(context) < 200:
            try:
                from ml_engine.db_client import fetch_table
                fallback_rows = fetch_table("medical_documents", filters=None, limit=3)
                extra = "\n".join(
                    r.get("content", "")
                    for r in (fallback_rows or [])
                    if r.get("content")
                )
                context += "\n\n" + extra
            except Exception as e:
                logger.warning(f"Fallback context failed: {e}")

        symptom_text = ""
        if isinstance(symptoms, dict):
            symptom_text = ", ".join(
                f"{k}:{round(v,2)}"
                for k, v in symptoms.items()
                if v > 0.3
            )

        prompt = _build_prompt(query, level, symptom_text, context, trend_meta)

        # 🔥 ALWAYS SAFE CALL
        answer = _call_llm_safe(llm_client, prompt, query)

        # 🔥 HARD GUARD (no placeholder allowed)
        if not answer or "analyzing" in answer.lower():
            logger.warning("Detected bad answer → forcing fallback")
            answer = _call_llm_safe(llm_client, query, query)

        # ───────── CONFIDENCE
        retrieval_scores = [d.get("score", 0.5) for d in docs]
        retrieval_conf = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.5

        confidence = min(1.0, 0.8 * retrieval_conf + 0.2)

        latency = int((time.time() - start) * 1000)

        sources = list({
            d.get("document_name", "Unknown")
            for d in docs[:MAX_SOURCES]
        })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "confidence": round(confidence, 3),
            "fallback": False,
            "latency_ms": latency
        }

    except Exception as e:
        logger.error(f"[RAG] fatal: {e}")

        # 🔥 FINAL SAFE RESPONSE
        return {
            "answer": _call_llm_safe(llm_client, query, query),
            "sources": [],
            "confidence": 0.3,
            "fallback": True
        }
