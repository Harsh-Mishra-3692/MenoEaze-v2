# rag_engine.py — ELITE v3 (PRODUCTION + RESEARCH READY)

import logging
import re
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger("menoeaze.rag")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_CONTEXT_CHARS = 3500
MAX_QUERY_LENGTH = 300
MIN_DOCS_REQUIRED = 1

MIN_RETRIEVAL_SCORE = 0.2


# ─────────────────────────────────────────────
# SANITIZATION
# ─────────────────────────────────────────────
# Patterns that indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"you\s+are\s+now",
    r"new\s+instruction",
    r"system\s*:",
    r"assistant\s*:",
    r"human\s*:",
    r"forget\s+(everything|all)",
    r"disregard\s+(all|previous|above)",
    r"override\s+(system|prompt|rules)",
    r"act\s+as\s+(if|a|an)",
    r"pretend\s+(you|to)",
]

def _sanitize(text: str) -> str:
    text = (text or "").strip()

    # Prompt injection defense: strip adversarial override attempts
    text_lower = text.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"[RAG] Prompt injection attempt detected and neutralized.")
            text = re.sub(pattern, "", text_lower, flags=re.IGNORECASE)

    text = re.sub(r"[^a-zA-Z0-9\s\-_,.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_QUERY_LENGTH]


# ─────────────────────────────────────────────
# SEVERITY
# ─────────────────────────────────────────────
def _severity_label(severity: float) -> str:
    if severity < 0.3:
        return "low"
    elif severity < 0.6:
        return "medium"
    return "high"


# ─────────────────────────────────────────────
# CONTEXT BUILDER (IMPROVED: PRIORITY BASED)
# ─────────────────────────────────────────────
def _build_context(docs: List[Dict[str, Any]]) -> str:
    # prioritize high-score docs
    docs = sorted(
        docs,
        key=lambda d: d.get("final_score", d.get("rerank_score", 0)),
        reverse=True
    )

    total = 0
    chunks = []

    for d in docs:
        content = (d.get("content") or "").strip()
        title = d.get("title", "Unknown")

        if not content:
            continue

        block = f"[{title}] {content}\n\n"

        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        chunks.append(block)
        total += len(block)

    return "".join(chunks)


# ─────────────────────────────────────────────
# CONTEXT VALIDATION
# ─────────────────────────────────────────────
def _is_context_valid(docs: List[Dict[str, Any]]) -> bool:
    if not docs or len(docs) < MIN_DOCS_REQUIRED:
        return False

    scores = [
        d.get("final_score", d.get("rerank_score", d.get("similarity", 0.0)))
        for d in docs
    ]

    avg_score = sum(scores) / len(scores)
    return avg_score > MIN_RETRIEVAL_SCORE


# ─────────────────────────────────────────────
# FALLBACK (SAFETY-FIRST MODE)
# ─────────────────────────────────────────────
def _fallback(level: str) -> str:
    _doctor_note = " Please consult a healthcare professional for personalized medical advice."
    if level == "low":
        return ("I do not have sufficient clinical evidence for this specific query. "
                "Based on general wellness guidelines, focus on sleep hygiene, hydration, "
                "and balanced nutrition." + _doctor_note)
    elif level == "medium":
        return ("I do not have sufficient clinical evidence for this specific query. "
                "Moderate symptoms may benefit from lifestyle adjustments and stress management." + _doctor_note)
    return ("I do not have sufficient clinical evidence for this specific query. "
            "Given the severity level, it is strongly recommended that you seek "
            "professional medical evaluation promptly." + _doctor_note)


# ─────────────────────────────────────────────
# PROMPT BUILDER (STRONGER)
# ─────────────────────────────────────────────
def _build_prompt(query: str, level: str, symptoms: str, context: str) -> str:
    return f"""You are a STRICT clinical assistant for menopause health.

ABSOLUTE RULES:
- Answer ONLY using the provided CONTEXT below. Do not use prior knowledge.
- If the answer is not in the CONTEXT, state clearly: "I do not have sufficient evidence to answer this."
- DO NOT hallucinate, speculate, or provide medical advice outside the CONTEXT.
- Be medically safe, concise, and actionable.
- Tailor your response to the patient's severity level.

User Query: {query}
Severity Level: {level}
Symptoms: {symptoms}

CONTEXT:
{context}

Answer:
"""


# ─────────────────────────────────────────────
# SAFE LLM CALL (COMPATIBLE)
# ─────────────────────────────────────────────
def _safe_llm_call(llm_client, prompt: str):
    try:
        res = llm_client.generate(prompt)

        # handle both string + dict (robust)
        if isinstance(res, str):
            return {
                "text": res,
                "latency": None,
                "request_id": None,
                "fallback": False
            }

        return res

    except Exception as e:
        logger.error(f"[RAG] LLM call failed: {e}")
        return {
            "text": "",
            "latency": None,
            "request_id": None,
            "fallback": True
        }


# ─────────────────────────────────────────────
# MAIN GENERATION
# ─────────────────────────────────────────────
def generate_answer(
    query: str,
    severity: float,
    docs: List[Dict[str, Any]],
    symptoms: Optional[Dict[str, float]],
    llm_client,
    return_debug: bool = False  # 🔥 NEW (for research)
) -> Dict[str, Any]:

    start = time.time()
    level = _severity_label(severity)

    # ── Guard: LLM missing
    if llm_client is None:
        return {
            "answer": _fallback(level),
            "sources": [],
            "confidence": 0.0,
            "fallback": True,
            "reason": "no_llm",
        }

    # ── Guard: retrieval weak
    if not _is_context_valid(docs):
        logger.warning("[RAG] Weak retrieval")

        return {
            "answer": _fallback(level),
            "sources": [],
            "confidence": 0.0,
            "fallback": True,
            "reason": "low_retrieval",
        }

    try:
        query = _sanitize(query)
        context = _build_context(docs)

        # ── Symptoms formatting
        symptom_text = ""
        if symptoms:
            symptom_text = ", ".join(
                f"{k.replace('_',' ')} ({v:.0%})"
                for k, v in sorted(symptoms.items(), key=lambda x: x[1], reverse=True)
                if v > 0.3
            )

        prompt = _build_prompt(query, level, symptom_text, context)

        # ── LLM CALL (SAFE)
        res = _safe_llm_call(llm_client, prompt)

        answer = res["text"].strip()

        # ── Confidence (IMPROVED)
        scores = [
            d.get("final_score", d.get("rerank_score", d.get("similarity", 0.5)))
            for d in docs
        ]

        retrieval_conf = sum(scores) / len(scores)

        # penalize fallback / empty answer
        penalty = 0.3 if res["fallback"] or not answer else 0.0

        confidence = max(0.0, min(1.0, retrieval_conf * (1 - penalty)))

        latency = (time.time() - start) * 1000

        result = {
            "answer": answer if answer else _fallback(level),
            "sources": list({d.get("title", "Unknown") for d in docs}),
            "confidence": round(confidence, 3),
            "latency_ms": round(latency, 2),
            "fallback": res["fallback"],
        }

        # 🔥 DEBUG MODE (for report)
        if return_debug:
            result["debug"] = {
                "num_docs": len(docs),
                "avg_score": round(retrieval_conf, 3),
                "context_length": len(context),
                "llm_latency": res.get("latency"),
                "request_id": res.get("request_id"),
            }

        return result

    except Exception as e:
        logger.error(f"[RAG] Generation failed: {e}")

        return {
            "answer": _fallback(level),
            "sources": [],
            "confidence": 0.0,
            "fallback": True,
            "reason": "exception",
        }