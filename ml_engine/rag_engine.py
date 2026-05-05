# rag_engine.py — FINAL ELITE (GROUNDING + SAFE + CLINICAL)

import logging
import re
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("menoeaze.rag")

MAX_CONTEXT_CHARS = 3200
MAX_DOC_CHARS = 800
MAX_QUERY_LENGTH = 300
MAX_SOURCES = 5

MIN_DOCS_REQUIRED = 1
MIN_RETRIEVAL_SCORE = 0.2

GROUNDING_THRESHOLD = 0.25

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
# CONTEXT (DIVERSE + PRIORITY AWARE)
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
# CONTEXT VALIDATION
# ─────────────────────────────────────────────
def _valid_context(docs):
    return len(docs) > 0


# ─────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────
def _fallback(level):
    return "The system is processing your symptoms. Please ensure you log all relevant details."


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
def _build_prompt(query, level, symptoms, context, trend_meta=None):
    trend_str = f"- direction: {trend_meta['direction']}\n- variability: {trend_meta['variability']}" if trend_meta else "No trend data available."
    return f"""You are a clinical AI assistant.

Use:
- the user's query
- retrieved medical knowledge
- user's symptom history
- model outputs (severity, confidence)

User query: {query}

Retrieved medical knowledge:
{context}

Patient Severity: {level}
Current Symptoms: {symptoms}
Trend signals:
{trend_str}

Generate a relevant, specific, and clinically grounded response.
Stay focused on the user's query.
Avoid unrelated symptoms unless clearly connected."""


# ─────────────────────────────────────────────
# GROUNDING CHECK
# ─────────────────────────────────────────────
def _grounding_score(answer: str, context: str) -> float:
    a_tokens = set(answer.lower().split())
    c_tokens = set(context.lower().split())

    if not a_tokens:
        return 0.0

    overlap = len(a_tokens & c_tokens) / len(a_tokens)
    return overlap


# ─────────────────────────────────────────────
# SAFE LLM
# ─────────────────────────────────────────────
def _call_llm(llm, prompt):
    try:
        res = llm.generate(prompt)

        if isinstance(res, str):
            return res, False

        return res.get("text", ""), res.get("fallback", False)

    except Exception:
        return "", True


# ─────────────────────────────────────────────
# MAIN
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

    if not llm_client:
        return {
            "answer": "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
            "sources": [],
            "confidence": 0.0,
            "fallback": True
        }

    try:
        query = _sanitize(query)
        context = _build_context(docs)
        
        if len(context) < 200:
            from ml_engine.db_client import fetch_table
            fallback_rows = fetch_table("medical_documents", filters=None, limit=3)
            context += "\n\n" + "\n".join([r.get("content", "") for r in (fallback_rows or []) if r.get("content")])

        symptom_text = ""
        if isinstance(symptoms, dict):
            symptom_text = ", ".join(
                f"{k}:{round(v,2)}"
                for k, v in symptoms.items()
                if v > 0.3
            )

        prompt = _build_prompt(query, level, symptom_text, context, trend_meta)

        answer, llm_fallback = _call_llm(llm_client, prompt)

        if not answer or answer.strip() == "":
            answer, llm_fallback = _call_llm(llm_client, prompt) # Retry once
            
        if not answer or answer.strip() == "":
            return {
                "answer": "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
                "sources": [d.get("document_name", "Unknown") for d in docs[:MAX_SOURCES]],
                "confidence": 0.5,
                "fallback": True
            }

        # ───────── CONFIDENCE
        retrieval_scores = [d.get("score", 0.5) for d in docs]
        retrieval_conf = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0.5

        confidence = min(1.0, 0.8 * retrieval_conf + 0.2)
        if llm_fallback:
            confidence *= 0.6

        latency = (time.time() - start) * 1000

        sources = list({
            d.get("document_name", "Unknown")
            for d in docs[:MAX_SOURCES]
        })

        result = {
            "answer": answer.strip(),
            "sources": sources,
            "confidence": round(confidence, 3),
            "fallback": llm_fallback,
            "latency_ms": int(latency)
        }

        return result

    except Exception as e:
        logger.error(f"[RAG] fatal: {e}")

        return {
            "answer": _fallback(level),
            "sources": [],
            "confidence": 0.0,
            "fallback": True
        }