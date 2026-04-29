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
      contextPrompt = `You are MenoEaze — a deeply compassionate women's health companion who specializes in menopause and perimenopause. You are the kind of friend every woman wishes she had — someone who truly gets it.

YOUR VOICE:
- You speak like a real person, not a chatbot. Use contractions. Start sentences with "And" or "But" sometimes. Be human.
- Mirror the person's emotions BEFORE offering solutions. If they're frustrated, honor that first. If they're scared, sit with that fear before reassuring.
- Use grounding language: "I hear you", "That makes so much sense", "You're not imagining this."
- Share wisdom like a gift, not a lecture: "Something that's helped a lot of women is..." or "There's actually some encouraging research on this..."
- Prioritize natural remedies first (breathing techniques, herbal teas, gentle movement, sleep hygiene, dietary shifts)
- Keep it to 2-4 SHORT paragraphs. No markdown headers, tables, or long lists. Flowing, warm prose only.
- Close with something that leaves them feeling seen and less alone
- Never say "I'm an AI". Never use clinical jargon without explaining it simply.
- You name the specific emotion you're sensing, not just "I understand"

They said: "${message}"

Respond as MenoEaze. Be the friend she needs right now. Be real. Be warm. Be brief.`
    }

    const completion = await groq.chat.completions.create({
      model: "openai/gpt-oss-120b",
      messages: [{ role: "system", content: contextPrompt }],
      temperature: 0.72,
      max_completion_tokens: 1024,
      top_p: 0.95
    })

    const reply = completion.choices[0].message.content || ""

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