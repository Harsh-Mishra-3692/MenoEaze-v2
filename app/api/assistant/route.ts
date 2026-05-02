import { NextResponse } from "next/server"
import { getGroq } from "../../../lib/groq/client"
import { isMenopauseRelated } from "../../../lib/assistant/domainGuard"

export async function POST(req: Request) {
  try {
    const { userId, message } = await req.json()

    if (!isMenopauseRelated(message)) {
      return NextResponse.json({
        reply:
          "I appreciate you reaching out — but I'm really only equipped to help with menopause and perimenopause-related things. If you have questions about symptoms, sleep changes, mood shifts, hormones, or anything your body is doing during this transition, I'm all ears and genuinely here for you. 💜"
      })
    }

    const mlApiUrl = process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000"
    
    // Fetch directly from the backend Python RAG engine
    const response = await fetch(`${mlApiUrl}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        query: message,
      })
    })

    if (!response.ok) {
      console.error(`Backend returned HTTP ${response.status}`);
      throw new Error(`Backend returned ${response.status}`)
    }

    const resJson = await response.json()
    
    if (resJson?.status === "error" || resJson?.status === "degraded") {
      console.error("Backend running in degraded mode or returned error:", resJson?.reason);
      if (resJson?.status === "error") {
        throw new Error(resJson?.reason || "Backend processing error");
      }
    }

    const reply = resJson?.data?.rag?.answer || "I'm sorry, I couldn't process that right now."
    const retrievedDocs = resJson?.data?.rag?.docs_used ? [{ text: `Used ${resJson.data.rag.docs_used} docs from backend` }] : []

    // Save to chat history (non-blocking, won't fail the response)
    try {
      const { supabaseAdmin } = await import("../../../lib/vector/vectorClient")
      supabaseAdmin
        .from("chat_messages")
        .insert([
          { user_id: userId, role: "user", content: message },
          { user_id: userId, role: "assistant", content: reply }
        ])
        .then(({ error }) => {
          if (error)
            console.warn("Failed to save chat history:", error.message)
        })
    } catch {
      // Supabase not configured — skip saving
    }

    return NextResponse.json({ reply, citations: retrievedDocs })
  } catch (error) {
    console.error("Assistant API error:", error)
    const message =
      error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json(
      { reply: `I'm having trouble connecting right now. Please try again in a moment. (${message})` },
      { status: 500 }
    )
  }
}