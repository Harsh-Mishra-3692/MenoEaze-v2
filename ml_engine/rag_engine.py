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
    _doctor_note = " And if things feel overwhelming, talking to your doctor can really help — you deserve that support."
    if level == "low":
        return ("I don't have specific research on that right now, but I want you to know that what you're experiencing is real and valid. "
                "In the meantime, gentle things like prioritizing sleep, staying hydrated, and nourishing your body with whole foods "
                "can make a quiet but meaningful difference." + _doctor_note)
    elif level == "medium":
        return ("I wish I had more specific information for you on this one. What I can say is that when symptoms are at this level, "
                "small, consistent shifts — like stress-relief techniques, gentle movement, and being intentional about rest — "
                "can start to move the needle." + _doctor_note)
    return ("I don't have enough information to give you the thorough answer you deserve on this, and I'm sorry about that. "
            "With what you're going through right now, I'd really encourage you to reach out to your healthcare provider soon. "
            "You don't have to push through this alone, and getting professional support is a sign of strength, not weakness. 💜")


# ─────────────────────────────────────────────
# PROMPT BUILDER (EMPATHETIC + EVIDENCE-GROUNDED)
# ─────────────────────────────────────────────
def _build_prompt(query: str, level: str, symptoms: str, context: str) -> str:
    return f"""You are MenoEaze — a compassionate, knowledgeable women's health companion who speaks with warmth and genuine care. You feel like a trusted friend who also happens to have deep medical knowledge.

CORE PRINCIPLES:
- ONLY use information from the CONTEXT provided below. Do not draw on outside knowledge.
- If the answer isn't in the CONTEXT, be honest and gentle: "I don't have specific information on that right now, but here's what might help..."
- Never speculate or fabricate medical claims.
- Prioritize natural remedies and lifestyle approaches first, then mention clinical options.
- Tailor your warmth and urgency to the person's severity level.
- Incorporate their symptoms and historical severity into your response for personalization, but medical evidence from CONTEXT must take absolute priority. Do NOT hallucinate advice.

YOUR VOICE:
- Open by acknowledging what the person is going through — show you heard them
- Use warm, conversational language. Say "you're" not "you are". Say "that's" not "that is".
- Keep it to 2-3 short paragraphs of flowing prose. No bullet lists, no markdown headers, no tables.
- Close with gentle encouragement or a caring thought — never a legal disclaimer.

User's question: {query}
Their severity level: {level}
Their symptoms: {symptoms}

CONTEXT:
{context}

Respond with warmth, brevity, and evidence from the context above. Be the caring, knowledgeable friend she needs right now."""


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