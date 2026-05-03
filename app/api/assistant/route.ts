// code file of route.ts

import { NextResponse } from "next/server"

const DEFAULT_BACKEND = "http://localhost:8000"
const REQUEST_TIMEOUT_MS = 15000

function sanitizeUrl(url: string) {
  return url.replace(/\/+$/, "") // remove trailing slashes
}

async function fetchWithTimeout(url: string, options: RequestInit, timeout = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeout)

  try {
    const res = await fetch(url, {
      ...options,
      signal: controller.signal,
    })
    return res
  } finally {
    clearTimeout(id)
  }
}

export async function POST(req: Request) {
  try {
    // ---------- INPUT PARSING ----------
    let body
    try {
      body = await req.json()
    } catch {
      return NextResponse.json(
        { reply: "Invalid JSON request body." },
        { status: 400 }
      )
    }

    const { userId, message } = body || {}

    if (!userId || !message || typeof message !== "string") {
      return NextResponse.json(
        { reply: "Missing or invalid user ID / message." },
        { status: 400 }
      )
    }

    // ---------- BACKEND URL ----------
    const rawUrl = process.env.NEXT_PUBLIC_ML_API_URL || DEFAULT_BACKEND
    const baseUrl = sanitizeUrl(rawUrl)

    const endpoint = `${baseUrl}/run`

    // ---------- REQUEST ----------
    let response: Response

    try {
      response = await fetchWithTimeout(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          action: "predict",
          payload: {
            user_id: userId,
            symptoms: message,
          },
        }),
      })
    } catch (err: any) {
      if (err.name === "AbortError") {
        throw new Error("Backend timeout")
      }
      throw new Error("Backend unreachable")
    }

    // ---------- STATUS CHECK ----------
    if (!response.ok) {
      console.error(`[Assistant API] Backend error ${response.status}`)
      throw new Error(`Backend returned ${response.status}`)
    }

    // ---------- SAFE JSON PARSE ----------
    let resJson: any
    try {
      resJson = await response.json()
    } catch {
      throw new Error("Invalid backend response format")
    }

    // ---------- SCHEMA GUARD ----------
    if (!resJson || typeof resJson !== "object") {
      throw new Error("Malformed backend response")
    }

    // ---------- DEGRADED / ERROR HANDLING ----------
    if (resJson.status === "error") {
      console.error("[Assistant API] Backend error:", resJson.reason)
      throw new Error(resJson.reason || "Processing error")
    }

    if (resJson.status === "degraded") {
      console.warn("[Assistant API] Degraded mode:", resJson.reasons)
    }

    // ---------- RESPONSE EXTRACTION ----------
    const reply =
      resJson?.rag?.answer ||
      resJson?.response ||
      "I'm sorry, I couldn't process that right now."

    const citations = Array.isArray(resJson?.rag?.sources)
      ? resJson.rag.sources
      : []

    const degradedReasons =
      resJson?.status === "degraded" && Array.isArray(resJson?.reasons)
        ? resJson.reasons
        : []

    return NextResponse.json({
      reply,
      citations,
      degradedReasons,
    })

  } catch (error: any) {
    console.error("[Assistant API] Fatal error:", error)

    const message =
      error?.message || "Unexpected error occurred"

    return NextResponse.json(
      {
        reply:
          "I'm having trouble connecting right now. Please try again in a moment. (" +
          message +
          ")",
      },
      { status: 500 }
    )
  }
}