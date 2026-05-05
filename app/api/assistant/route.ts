// app/api/assistant/route.ts — PRODUCTION v3 (L7/L9 HARDENED)

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

/* ================= DOMAIN ================= */

const OFF_DOMAIN = [
  "weather","stock","crypto","bitcoin","football","movie",
  "recipe","code","javascript","python","react","politics",
  "math","algebra","homework","history","geography","song"
]

function isOffDomain(query: string): boolean {
  const q = query.toLowerCase()
  if (!OFF_DOMAIN.some(k => q.includes(k))) return false

  const healthSignals = [
    "symptom","health","pain","sleep","mood","stress",
    "hormone","menopause","hot flash","fatigue","headache",
    "doctor","treatment","wellness"
  ]

  return !healthSignals.some(h => q.includes(h))
}

/* ================= SIGNAL ================= */

function computeTrend(seq: number[][]): string {
  const sums = seq.map(r => r.reduce((a, b) => a + b, 0))
  if (sums.length < 2) return "insufficient"
  const delta = sums[sums.length - 1] - sums[0]
  if (delta > 5) return "worsening"
  if (delta < -5) return "improving"
  return "stable"
}

function computeVariability(seq: number[][]): string {
  const flat = seq.flat()
  if (!flat.length) return "unknown"
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length
  const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length
  if (variance > 8) return "high"
  if (variance > 3) return "moderate"
  return "low"
}

function buildEnrichedQuery(message: string, seq: number[][]): string {
  return `
User question:
${message}

Patient context:
- trend: ${computeTrend(seq)}
- variability: ${computeVariability(seq)}
- history: ${seq.length} logs

Instruction:
Provide a clinically grounded, personalized explanation.
Avoid generic responses.
`
}

/* ================= RESPONSE ================= */

function extractAnswer(res: any): string {
  const text =
    res?.answer ||
    res?.rag?.answer ||
    res?.response

  if (typeof text === "string" && text.trim().length > 20) {
    return text.trim()
  }

  return ""
}

/* ================= MAIN ================= */

export async function POST(request: Request) {
  try {
    const body = await request.json()

    const message = sanitizeInput(body?.message)
    const userId = typeof body?.userId === "string" ? body.userId : ""

    if (!message) {
      return NextResponse.json({
        reply: "Please ask a menopause or health-related question.",
        citations: [],
        severity: 0,
        confidence: 0,
        degraded: true
      })
    }

    if (isOffDomain(message)) {
      return NextResponse.json({
        reply: "I focus on menopause and women's health. Please ask a related question.",
        citations: [],
        severity: 0,
        confidence: 0,
        degraded: true
      })
    }

    let sequence = Array.isArray(body?.sequence) ? body.sequence : generateFallbackSequence()

    const enrichedQuery = buildEnrichedQuery(message, sequence)

    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

    let resJson: any = null

    try {
      const res = await fetch(`${getBackendUrl()}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: userId,
          query: enrichedQuery,   // 🔥 FIXED: now using enriched context
          sequence
        }),
        signal: controller.signal
      })

      clearTimeout(timeout)

      if (!res.ok) throw new Error(`Backend ${res.status}`)

      resJson = await res.json()

    } catch (err) {
      clearTimeout(timeout)

      return NextResponse.json({
        reply: "I'm unable to retrieve a detailed response right now. Please try again.",
        citations: [],
        severity: 0,
        confidence: 0,
        degraded: true
      })
    }

    const answer = extractAnswer(resJson)

    const reply = answer || "Your symptoms may be related to hormonal changes. For accurate evaluation, consult a healthcare professional."

    const severity = safeNum(resJson?.severity ?? resJson?.prediction?.severity, 0)
    const confidence = safeNum(resJson?.confidence ?? resJson?.prediction?.confidence, 0)

    const citations = safeArray(resJson?.citations ?? resJson?.rag?.sources).map((c: any) =>
      typeof c === "string"
        ? { title: c, snippet: "", url: undefined }
        : {
            title: c?.title || "Clinical Source",
            snippet: c?.snippet || "",
            url: c?.url
          }
    )

    return NextResponse.json({
      reply,
      citations,
      severity,
      confidence,
      degraded: false
    })

  } catch (error) {
    console.error("[Assistant API Fatal]", error)

    return NextResponse.json({
      reply: "Unable to process request. Please try again.",
      citations: [],
      severity: 0,
      confidence: 0,
      degraded: true
    })
  }
}
