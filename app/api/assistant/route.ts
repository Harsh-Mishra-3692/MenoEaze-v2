import { NextResponse } from "next/server"

export const dynamic = "force-dynamic";

/* ================= CONFIG ================= */
const REQUEST_TIMEOUT_MS = 15000
const MAX_MESSAGE_LENGTH = 500

/* ================= HELPERS ================= */

function getBackendUrl(): string {
  const url = process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000"
  return url.replace(/\/+$/, "")
}

function sanitizeInput(input: unknown): string {
  if (typeof input !== "string") return ""
  return input.trim().slice(0, MAX_MESSAGE_LENGTH)
}

function safeArray<T>(arr: unknown): T[] {
  return Array.isArray(arr) ? arr : []
}

function safeNum(v: unknown, d: number): number {
  const n = Number(v)
  return Number.isNaN(n) ? d : n
}

function generateFallbackSequence(): number[][] {
  return Array.from({ length: 5 }, () => Array.from({ length: 11 }, () => 0))
}

/* ================= DOMAIN HANDLING ================= */

// Clearly off-domain: topics with zero health/body/wellness overlap
const OFF_DOMAIN = [
  "weather", "forecast", "stock", "crypto", "bitcoin", "football", "soccer",
  "basketball", "movie", "film", "recipe", "cook", "code", "program",
  "javascript", "python", "java", "react", "deploy", "server", "database",
  "politics", "election", "president", "math", "calcul", "algebra",
  "homework", "essay", "history of", "geography", "capital of",
  "translate", "song", "music", "lyrics", "game", "minecraft", "fortnite"
]

function isCompletelyOffDomain(query: string): boolean {
  const lower = query.toLowerCase()
  const hasOffDomain = OFF_DOMAIN.some(kw => lower.includes(kw))
  if (!hasOffDomain) return false

  const healthSignals = [
    "symptom", "health", "body", "pain", "sleep", "tired", "mood",
    "anxiety", "stress", "feel", "ache", "hormone", "menopause",
    "hot flash", "sweat", "fatigue", "headache", "joint", "bone",
    "woman", "medical", "doctor", "treatment", "wellness", "exercise",
    "diet", "weight", "period", "cycle", "depress", "emotion", "cramp"
  ]
  return !healthSignals.some(h => lower.includes(h))
}

/* ================= SIGNAL COMPUTATION ================= */

function computeTrend(seq: number[][]): string {
  const sums = seq.map(r => r.reduce((a, b) => a + b, 0))
  if (sums.length < 2) return "insufficient data"
  const delta = sums[sums.length - 1] - sums[0]
  if (delta > 5) return "worsening"
  if (delta < -5) return "improving"
  return "stable"
}

function computeVariability(seq: number[][]): string {
  const flat = seq.flat()
  if (flat.length === 0) return "unknown"
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length
  const variance = flat.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / flat.length
  if (variance > 8) return "highly variable"
  if (variance > 3) return "moderately variable"
  return "consistent"
}

function countHistoryDepth(seq: number[][]): number {
  return seq.filter(r => r.some(v => v > 0)).length
}

/* ================= ENRICHED QUERY ================= */

function buildEnrichedQuery(message: string, seq: number[][]): string {
  const trend = computeTrend(seq)
  const variability = computeVariability(seq)
  const historyDepth = countHistoryDepth(seq)

  return `User question:\n${message}\n\nPatient context:\n- trend: ${trend}\n- variability: ${variability}\n- history: ${historyDepth} recent logs\n\nInstruction:\nProvide a clinically grounded and personalized explanation using both the patient context and relevant medical knowledge.`
}

/* ================= RESPONSE NORMALIZATION ================= */

function normalizeReply(resJson: any): string {
  const text = resJson?.rag?.answer ?? resJson?.answer ?? resJson?.response
  if (typeof text !== "string" || text.trim().length === 0) {
    return "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights..."
  }
  return text.trim()
}

/* ================= MAIN ROUTE ================= */

export async function POST(request: Request) {
  try {
    const body = await request.json()
    const message = sanitizeInput(body?.message)
    const userId = typeof body?.userId === "string" ? body.userId : ""

    if (!message) {
      return NextResponse.json({
        reply: "Please ask a question about menopause or your symptoms.",
        citations: [],
        severity: 0,
        confidence: 0,
        degradedReasons: ["empty_query"]
      })
    }

    // Domain check
    if (isCompletelyOffDomain(message)) {
      return NextResponse.json({
        reply: "I specialize in menopause and women's health. Could you ask a health-related question?",
        citations: [],
        severity: 0,
        confidence: 0,
        degradedReasons: ["off_domain"]
      })
    }

    // Build sequence from client or fallback
    let sequence: number[][] = body?.sequence
    if (!Array.isArray(sequence) || sequence.length !== 5) {
      sequence = generateFallbackSequence()
    }

    // Enrich query
    const enrichedQuery = buildEnrichedQuery(message, sequence)

    // Call backend /run
    const backendUrl = getBackendUrl()
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

    const degradedReasons: string[] = []

    let resJson: any
    try {
      const res = await fetch(`${backendUrl}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          query: message, // Pass clean query to prevent drift
          sequence,
        }),
        signal: controller.signal,
      })
      clearTimeout(timeout)

      if (!res.ok) {
        degradedReasons.push(`backend_${res.status}`)
        return NextResponse.json({
          reply: "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
          citations: [],
          severity: 0,
          confidence: 0,
          degradedReasons,
        })
      }

      resJson = await res.json()
    } catch (fetchErr: any) {
      clearTimeout(timeout)
      degradedReasons.push(fetchErr?.name === "AbortError" ? "timeout" : "network")
      return NextResponse.json({
        reply: "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
        citations: [],
        severity: 0,
        confidence: 0,
        degradedReasons,
      })
    }

    // Normalize response
    const reply = normalizeReply(resJson)
    const severity = safeNum(resJson?.prediction?.severity ?? resJson?.severity, 0)
    const confidence = safeNum(resJson?.prediction?.confidence ?? resJson?.confidence, 0)

    const citations = safeArray(resJson?.rag?.sources).map((c: any) => {
      // Backend may return strings (doc names) or objects — handle both
      if (typeof c === "string") {
        return { title: c, snippet: "", url: undefined }
      }
      return {
        title: typeof c?.title === "string" ? c.title : "Clinical Source",
        snippet: typeof c?.snippet === "string" ? c.snippet : "",
        url: typeof c?.url === "string" ? c.url : undefined,
      }
    })

    return NextResponse.json({
      reply,
      citations,
      severity,
      confidence,
      degradedReasons,
    })

  } catch (error: any) {
    console.error("[Assistant API] Fatal:", error)
    return NextResponse.json({
      reply: "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
      citations: [],
      severity: 0,
      confidence: 0,
      degradedReasons: ["fatal"],
    })
  }
}