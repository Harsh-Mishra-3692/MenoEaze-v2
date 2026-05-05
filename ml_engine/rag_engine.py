# rag_engine.py — PRODUCTION v7 (L7/L9 HARDENED, DEPLOYMENT SAFE)

import logging
import re
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("menoeaze.rag")

MAX_CONTEXT_CHARS = 3200
MAX_DOC_CHARS = 800
MAX_QUERY_LENGTH = 300
MAX_SOURCES = 5

MIN_CONTEXT_THRESHOLD = 180


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
def _build_context(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return ""

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
        f"- direction: {trend_meta.get('direction')}\n- variability: {trend_meta.get('variability')}"
        if isinstance(trend_meta, dict)
        else "No trend data available."
    )

    return f"""
You are a clinical decision-support assistant for menopause.

User Query:
{query}

Severity Level:
{level}

Symptoms:
{symptoms if symptoms else "Not explicitly provided"}

Trend:
{trend_str}

Context:
{context if context else "General clinical knowledge may be applied."}

Instructions:
- Provide a clear, medically grounded explanation
- Do NOT hallucinate facts not supported by context
- Do NOT say "analyzing"
- Be concise but informative
"""


# ─────────────────────────────────────────────
# SAFE LLM CALL
# ─────────────────────────────────────────────
def _call_llm_safe(llm, prompt: str, fallback_query: str) -> str:

    try:
        res = llm.generate(prompt)

        # normalize output
        if isinstance(res, dict):
            text = res.get("text", "").strip()
        else:
            text = str(res).strip()

        if text:
            return text

        # fallback 1: simple query
        res2 = llm.generate(fallback_query)
        if isinstance(res2, dict):
            return res2.get("text", "").strip()

        return str(res2).strip()

    except Exception as e:
        logger.warning(f"[RAG] LLM primary failed: {e}")

        try:
            res = llm.generate(fallback_query)
            if isinstance(res, dict):
                return res.get("text", "").strip()
            return str(res).strip()
        except Exception:
            return _safe_default_message()


# ─────────────────────────────────────────────
# SAFE DEFAULT
# ─────────────────────────────────────────────
def _safe_default_message() -> str:
    return (
        "Your symptoms may be related to hormonal changes associated with menopause. "
        "For a more accurate assessment and personalized guidance, please consult a qualified healthcare professional."
    )


# ─────────────────────────────────────────────
# CONTEXT ENHANCEMENT (CRITICAL FOR RAILWAY)
# ─────────────────────────────────────────────
def _enhance_context_if_weak(context: str) -> str:

    if len(context) >= MIN_CONTEXT_THRESHOLD:
        return context

    # inject minimal domain grounding
    base_knowledge = (
        "Menopause commonly involves symptoms such as hot flashes, sleep disturbances, "
        "mood fluctuations, fatigue, and hormonal changes."
    )

    return context + "\n\n" + base_knowledge


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

    try:
        query = _sanitize(query)
        level = _severity_label(severity)

        # ───────── CONTEXT ─────────
        context = _build_context(docs)
        context = _enhance_context_if_weak(context)

        # ───────── SYMPTOMS ─────────
        symptom_text = ""
        if isinstance(symptoms, dict):
            symptom_text = ", ".join(
                f"{k}:{round(v,2)}"
                for k, v in symptoms.items()
                if v > 0.3
            )

        # ───────── PROMPT ─────────
        prompt = _build_prompt(query, level, symptom_text, context, trend_meta)

        # ───────── LLM ─────────
        answer = _call_llm_safe(llm_client, prompt, query)

        # ───────── HARD GUARD ─────────
        if not answer or "analyzing" in answer.lower():
            logger.warning("[RAG] Weak output detected → forcing fallback")
            answer = _call_llm_safe(llm_client, query, query)

        if not answer:
            answer = _safe_default_message()

        # ───────── CONFIDENCE ─────────
        retrieval_scores = [d.get("score", 0.5) for d in docs] if docs else [0.5]
        retrieval_conf = sum(retrieval_scores) / len(retrieval_scores)

        confidence = min(1.0, 0.75 * retrieval_conf + 0.25)

        latency = int((time.time() - start) * 1000)

        sources = list({
            d.get("document_name") or d.get("source") or "Unknown"
            for d in docs[:MAX_SOURCES]
        }) if docs else []

        return {
            "answer": answer.strip(),
            "sources": sources,
            "confidence": round(confidence, 3),
            "fallback": False,
            "latency_ms": latency
        }

    except Exception as e:
        logger.error(f"[RAG] fatal: {e}")

        return {
            "answer": _safe_default_message(),
            "sources": [],
            "confidence": 0.3,
            "fallback": True
        }
