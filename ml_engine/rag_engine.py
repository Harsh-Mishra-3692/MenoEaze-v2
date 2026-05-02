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

        source = d.get("document_name")
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
    if len(docs) < MIN_DOCS_REQUIRED:
        return False

    scores = [d.get("score", 0) for d in docs]
    return (sum(scores) / len(scores)) > MIN_RETRIEVAL_SCORE


# ─────────────────────────────────────────────
# FALLBACK
# ─────────────────────────────────────────────
def _fallback(level):
    base = "I don’t have enough reliable medical evidence to answer precisely."

    if level == "low":
        return base + " General healthy habits may help."
    if level == "medium":
        return base + " Monitor symptoms and consider lifestyle adjustments."
    return base + " Please consult a healthcare professional."


# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
def _build_prompt(query, level, symptoms, context):
    return f"""You are a clinical assistant.

STRICT:
- Use ONLY the provided CONTEXT
- If unsure → say you don't know
- No hallucinations

Query: {query}
Severity: {level}
Symptoms: {symptoms}

CONTEXT:
{context}

Answer in 2 concise paragraphs.
"""


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
    return_debug=False
):

    start = time.time()
    level = _severity_label(severity)

    if not llm_client or not docs or not _valid_context(docs):
        return {
            "answer": _fallback(level),
            "sources": [],
            "confidence": 0.0,
            "fallback": True
        }

    try:
        query = _sanitize(query)
        context = _build_context(docs)

        symptom_text = ""
        if isinstance(symptoms, dict):
            symptom_text = ", ".join(
                f"{k}:{round(v,2)}"
                for k, v in symptoms.items()
                if v > 0.3
            )

        prompt = _build_prompt(query, level, symptom_text, context)

        answer, llm_fallback = _call_llm(llm_client, prompt)

        if not answer:
            return {
                "answer": _fallback(level),
                "sources": [],
                "confidence": 0.0,
                "fallback": True
            }

        # ───────── GROUNDING CHECK
        grounding = _grounding_score(answer, context)

        if grounding < GROUNDING_THRESHOLD:
            logger.warning("[RAG] hallucination detected")
            answer = _fallback(level)
            llm_fallback = True

        # ───────── CONFIDENCE
        retrieval_scores = [d.get("score", 0.5) for d in docs]
        retrieval_conf = sum(retrieval_scores) / len(retrieval_scores)

        confidence = (
            0.6 * retrieval_conf +
            0.4 * grounding
        )

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
            "confidence": round(min(1.0, confidence), 3),
            "latency_ms": round(latency, 2),
            "fallback": llm_fallback
        }

        if return_debug:
            result["debug"] = {
                "grounding": round(grounding, 3),
                "docs": len(docs),
                "context_len": len(context)
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