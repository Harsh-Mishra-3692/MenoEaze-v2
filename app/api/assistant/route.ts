import { NextResponse } from "next/server"

export async function POST(req: Request) {
  try {
    const { userId, message } = await req.json()

    if (!userId || !message) {
      return NextResponse.json(
        { reply: "Missing user ID or message." },
        { status: 400 }
      )
    }

    const mlApiUrl = process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000"
    
    // STRICT /run PROXY
    // Format: { user_id, action: "predict", payload: { symptoms } }
    const response = await fetch(`${mlApiUrl}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        action: "predict",
        payload: {
          symptoms: message
        }
      })
    })

    if (!response.ok) {
      console.error(`Backend returned HTTP ${response.status}`)
      throw new Error(`Backend returned ${response.status}`)
    }

    const resJson = await response.json()
    
    // Strictly handle schema
    if (resJson?.status === "error" || resJson?.status === "degraded") {
      console.error("Backend running in degraded mode or returned error:", resJson?.reason)
      if (resJson?.status === "error") {
        throw new Error(resJson?.reason || "Backend processing error")
      }
    }

    const reply = resJson?.rag?.answer || "I'm sorry, I couldn't process that right now."
    const citations = resJson?.rag?.sources || []
    const degradedReasons = (resJson?.status === 'degraded' && Array.isArray(resJson?.reasons))
      ? resJson.reasons
      : []

    return NextResponse.json({ reply, citations, degradedReasons })
  } catch (error) {
    console.error("Assistant API error:", error)
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json(
      { reply: `I'm having trouble connecting right now. Please try again in a moment. (${message})` },
      { status: 500 }
    )
  }
}