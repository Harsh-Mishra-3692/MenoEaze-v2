import { NextResponse } from "next/server"
import { getGroq } from "@/lib/groq/client"

export async function POST(req: Request) {
    try {
        const { userId } = await req.json()

        if (!userId) {
            return NextResponse.json(
                { summary: "Please log in to see personalized insights." },
                { status: 400 }
            )
        }

        // Fetch recent symptom logs
        let logs: any[] = []
        try {
            const { supabaseAdmin } = await import("@/lib/vector/vectorClient")

            const { data, error } = await supabaseAdmin
                .from("symptom_logs")
                .select("symptom_type, severity, mood, sleep, notes, created_at")
                .eq("user_id", userId)
                .eq("is_deleted", false)
                .order("created_at", { ascending: false })
                .limit(30)

            if (error) throw error
            logs = data || []
        } catch (dbError) {
            console.warn("Failed to fetch logs for insight:", dbError)
            return NextResponse.json({
                summary:
                    "Start logging your symptoms to receive personalized insights. Every entry helps us understand your unique journey better. 💜"
            })
        }

        if (logs.length === 0) {
            return NextResponse.json({
                summary:
                    "No symptoms logged yet. Begin tracking how you feel each day, and we'll provide personalized insights based on your patterns. 💜"
            })
        }

        // Build a structured summary of logs for the LLM
        const logSummary = logs
            .slice(0, 15)
            .map((l) => {
                const date = new Date(l.created_at).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric"
                })
                return `${date}: ${l.symptom_type} (severity ${l.severity}/10), mood: ${l.mood || "N/A"}, sleep: ${l.sleep || "N/A"}/10${l.notes ? `, notes: ${l.notes}` : ""}`
            })
            .join("\n")

        const avgSeverity = (
            logs.reduce((sum, l) => sum + (l.severity || 0), 0) / logs.length
        ).toFixed(1)

        const symptoms = [...new Set(logs.map((l) => l.symptom_type).filter(Boolean))]
        const moods = [...new Set(logs.map((l) => l.mood).filter(Boolean))]

        const groq = getGroq()

        const completion = await groq.chat.completions.create({
            model: "openai/gpt-oss-120b",
            messages: [
                {
                    role: "system",
                    content: `You are MenoEaze's health insight engine. Generate a brief, empathetic, personalized 2-3 sentence health summary for a woman tracking her menopause symptoms.

Based on her recent data:
- ${logs.length} symptom entries logged
- Average severity: ${avgSeverity}/10
- Common symptoms: ${symptoms.slice(0, 6).join(", ")}
- Recent moods: ${moods.slice(0, 5).join(", ")}

Recent log details:
${logSummary}

Write a warm, insightful 2-3 sentence summary that:
1. Acknowledges her specific patterns (mention her actual symptoms by name)
2. Offers one gentle, actionable observation
3. Feels like a caring friend, not a clinical report

Do NOT use markdown, headers, or bullet points. Just flowing prose. Keep it under 60 words.`
                }
            ],
            temperature: 0.6,
            max_completion_tokens: 150,
            top_p: 1
        })

        const summary =
            completion.choices[0].message.content ||
            "Keep tracking your symptoms — patterns will emerge that help us support you better. 💜"

        return NextResponse.json({ summary })
    } catch (error) {
        console.error("Insight summary error:", error)
        return NextResponse.json({
            summary:
                "Your wellness journey matters. Keep logging your symptoms and we'll provide personalized insights as your data grows. 💜"
        })
    }
}
