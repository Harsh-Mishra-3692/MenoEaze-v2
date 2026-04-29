import { NextResponse } from "next/server"
import { getGroq } from "@/lib/groq/client"
import { isMenopauseRelated } from "@/lib/assistant/domainGuard"

export async function POST(req: Request) {
  try {
    const { userId, message } = await req.json()

    if (!isMenopauseRelated(message)) {
      return NextResponse.json({
        reply:
          "I specialize only in menopause-related health concerns. Please ask me about menopause symptoms, sleep, mood, hormones, or related wellness topics."
      })
    }

    const groq = getGroq()

    // Try to build full context, but gracefully degrade if services are unavailable
    let contextPrompt = ""
    let retrievedDocs: any[] = []

    try {
      const { buildAssistantContext } = await import(
        "@/lib/assistant/contextBuilder"
      )
      const { buildPrompt } = await import("@/lib/assistant/promptBuilder")

      const { ml, retrievedDocs: docs, memory, symptomHistory } =
        await buildAssistantContext(userId, message)

      retrievedDocs = docs
      contextPrompt = buildPrompt(message, ml, retrievedDocs, memory, symptomHistory)
    } catch (contextError) {
      console.warn(
        "Context building failed, using fallback prompt:",
        contextError instanceof Error ? contextError.message : contextError
      )

      // Fallback prompt when Supabase/OpenAI services are unavailable
      contextPrompt = `You are MenoEaze — a compassionate women's health companion specializing exclusively in menopause and perimenopause care. You speak like a warm, knowledgeable older sister — never like a chatbot or clinical system.

HOW YOU RESPOND:
- Start by genuinely acknowledging how the person feels
- Keep responses SHORT — 2-4 natural paragraphs max
- Prioritize natural, holistic remedies FIRST (herbal teas, breathing exercises, dietary changes, yoga, lifestyle adjustments)
- Weave evidence naturally: "Studies have found..." or "Many women find that..."
- NEVER use tables, long lists, or markdown headers. Use flowing prose with max 3 bullet points
- End with gentle encouragement, never a legal disclaimer
- Never say "I'm an AI" or use clinical jargon without explaining it simply

User says: "${message}"

Respond as MenoEaze — warm, brief, evidence-informed, natural-remedy-first.`
    }

    const completion = await groq.chat.completions.create({
      model: "openai/gpt-oss-120b",
      messages: [{ role: "system", content: contextPrompt }],
      temperature: 0.5,
      max_completion_tokens: 1024,
      top_p: 1
    })

    const reply = completion.choices[0].message.content || ""

    // Save to chat history (non-blocking, won't fail the response)
    try {
      const { supabaseAdmin } = await import("@/lib/vector/vectorClient")
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