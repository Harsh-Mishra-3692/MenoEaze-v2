// lib/mlClient.ts

import { supabase } from "@/lib/supabase"

const ML_API_URL =
  process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000"

/* ================= TYPES ================= */

type ActionType = "log" | "predict" | "feedback"

interface RunPayload {
  action: ActionType
  user_id?: string
  query?: string
  sequence?: number[][]
  feedback?: unknown
}

export interface RunResponse {
  severity: number
  confidence: number | null
  answer: string
  trend_summary?: string
  error?: string
  raw?: unknown // preserved backend response (debug-safe)
}

/* ================= CONSTANTS ================= */

const REQUEST_TIMEOUT = 15000 // 15s hard timeout
const MAX_QUERY_LENGTH = 500
const DEFAULT_USER = "demo_user"
const SEQ_LEN = 5
const FEATURES = 11

/* ================= SEQUENCE BUILDER ================= */

/**
 * Fetch last 5 symptom logs from Supabase and build a 5x11 matrix.
 * Pads with zero-rows if fewer than 5 logs exist.
 * Returns a guaranteed 5x11 number[][] — NEVER throws.
 */
export async function getUserSequence(userId: string): Promise<number[][]> {
  const zeroRow = (): number[] => Array.from({ length: FEATURES }, () => 0)

  try {
    if (!userId) {
      return Array.from({ length: SEQ_LEN }, zeroRow)
    }

    const { data, error } = await supabase
      .from("symptom_logs")
      .select("feature_vector")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(SEQ_LEN)

    if (error || !data || data.length === 0) {
      return Array.from({ length: SEQ_LEN }, zeroRow)
    }

    // Reverse so oldest first (chronological order)
    const rows = data.reverse()

    const vectors: number[][] = rows
      .map((row: any) => {
        const fv = row.feature_vector
        if (!Array.isArray(fv) || fv.length !== FEATURES) return null
        const nums = fv.map((v: any) => {
          const n = Number(v)
          return Number.isNaN(n) ? 0 : n
        })
        return nums
      })
      .filter((v): v is number[] => v !== null)

    // Pad with zero rows if we have fewer than 5
    while (vectors.length < SEQ_LEN) {
      vectors.unshift(zeroRow())
    }

    // Take last 5 only
    return vectors.slice(-SEQ_LEN)
  } catch {
    return Array.from({ length: SEQ_LEN }, zeroRow)
  }
}

/* ================= HELPERS ================= */

function isValidSequence(seq: unknown): seq is number[][] {
  return (
    Array.isArray(seq) &&
    seq.length === 5 &&
    seq.every(
      (row) =>
        Array.isArray(row) &&
        row.length === 11 &&
        row.every((x) => typeof x === "number" && !Number.isNaN(x))
    )
  )
}

function sanitizeString(input: unknown): string {
  if (typeof input !== "string") return ""
  return input.trim().slice(0, MAX_QUERY_LENGTH)
}

function normalizeResponse(data: any): RunResponse {
  const conf = data?.prediction?.confidence ?? data?.confidence;
  
  return {
    severity: safeNumber(
      data?.prediction?.severity ?? data?.severity,
      0.5
    ),
    confidence: (conf === null || conf === undefined) ? null : safeNumber(conf, 0.5),
    answer: normalizeAnswer(data),
    trend_summary: data?.trend_summary,
    raw: data
  }
}

function normalizeAnswer(data: any): string {
  const text = data?.rag?.answer ?? data?.answer

  if (typeof text !== "string" || text.trim().length === 0) {
    return "Insufficient clinical evidence to provide a reliable answer."
  }

  return text.trim()
}

function safeNumber(value: unknown, fallback: number): number {
  const n = Number(value)
  if (Number.isNaN(n)) return fallback
  return n
}

/* ================= CORE FETCH ================= */

async function safeFetch(payload: RunPayload): Promise<RunResponse> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT)

  try {
    const res = await fetch(`${ML_API_URL}/run`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
      cache: "no-store"
    })

    clearTimeout(timeout)

    if (!res.ok) {
      const text = await res.text().catch(() => "")
      throw new Error(`HTTP ${res.status}: ${text}`)
    }

    const data = await res.json()

    return normalizeResponse(data)

  } catch (err: any) {
    clearTimeout(timeout)

    return {
      severity: 0.5,
      confidence: null,
      answer: "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
      error: err?.message || "Unknown error"
    }
  }
}

/* ================= MAIN ENTRY ================= */

export async function runAction({
  action,
  userId,
  query,
  sequence,
  feedback
}: {
  action: ActionType
  userId?: string
  query?: string
  sequence?: number[][]
  feedback?: unknown
}): Promise<RunResponse> {
  /* ---- VALIDATION ---- */

  if (!["log", "predict", "feedback"].includes(action)) {
    throw new Error(`Invalid action: ${action}`)
  }

  if (action === "predict") {
    if (!isValidSequence(sequence)) {
      return {
        severity: 0,
        confidence: 0,
        answer: "Invalid input sequence.",
        error: "invalid_sequence"
      }
    }
  }

  /* ---- BUILD PAYLOAD ---- */

  const payload: RunPayload = {
    action,
    user_id: userId || DEFAULT_USER
  }

  if (query) payload.query = sanitizeString(query)
  if (sequence) payload.sequence = sequence
  if (feedback) payload.feedback = feedback

  /* ---- EXECUTE ---- */

  return safeFetch(payload)
}

/* ================= WRAPPERS ================= */

// Prediction — auto-fetches sequence from Supabase if not provided
export async function runPrediction(
  sequenceOrUserId: number[][] | string,
  userId?: string,
  query?: string
) {
  let sequence: number[][]
  let uid: string

  if (typeof sequenceOrUserId === "string") {
    // Called with userId — fetch sequence from Supabase
    uid = sequenceOrUserId
    sequence = await getUserSequence(uid)
  } else {
    // Called with explicit sequence (backward compat)
    sequence = sequenceOrUserId
    uid = userId || DEFAULT_USER
  }

  return runAction({
    action: "predict",
    userId: uid,
    query: query || "menopause symptom analysis",
    sequence
  })
}

// Assistant — fetches sequence from Supabase
export async function askAssistant(
  message: string,
  userId: string
) {
  const sequence = await getUserSequence(userId)

  return runAction({
    action: "predict",
    userId,
    query: message,
    sequence
  })
}

// Logging
export async function logSymptoms(data: unknown) {
  return runAction({
    action: "log",
    feedback: data
  })
}

// Feedback loop
export async function sendFeedback(data: unknown) {
  return runAction({
    action: "feedback",
    feedback: data
  })
}