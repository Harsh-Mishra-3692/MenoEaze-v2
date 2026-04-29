// lib/assistant/contextBuilder.ts

import { analyzeUserData } from "../ml"
import { retrieveRelevantDocs } from "../rag/retriever"
import { supabaseAdmin } from "../vector/vectorClient"

export async function buildAssistantContext(userId: string, message: string) {
    // Fetch symptom logs with all fields for rich context
    const { data: logs } = await supabaseAdmin
        .from("symptom_logs")
        .select("severity, mood_score, sleep_quality, hot_flash_score, night_sweats_score, fatigue_score, anxiety_score, stress_level, notes, created_at")
        .eq("user_id", userId)
        .eq("is_deleted", false)
        .order("created_at", { ascending: false })
        .limit(30)

    const ml = analyzeUserData(logs || [])

    const retrievedDocs = await retrieveRelevantDocs(message)

    // Fetch recent chat history
    const { data: memoryData } = await supabaseAdmin
        .from("chat_messages")
        .select("role, content")
        .eq("user_id", userId)
        .order("created_at", { ascending: false })
        .limit(6)

    const memory =
        memoryData
            ?.reverse()
            .map(m => `${m.role}: ${m.content}`)
            .join("\n") || ""

    // Build a readable symptom history for the prompt
    const symptomHistory = (logs || [])
        .slice(0, 10)
        .map(l => {
            const date = new Date(l.created_at).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric"
            })
            
            // Collect the top symptoms for this day
            const activeSymptoms = []
            if (l.hot_flash_score > 3) activeSymptoms.push(`hot flashes (${l.hot_flash_score}/10)`)
            if (l.night_sweats_score > 3) activeSymptoms.push(`night sweats (${l.night_sweats_score}/10)`)
            if (l.fatigue_score > 3) activeSymptoms.push(`fatigue (${l.fatigue_score}/10)`)
            if (l.anxiety_score > 3) activeSymptoms.push(`anxiety (${l.anxiety_score}/10)`)
            if (l.stress_level > 3) activeSymptoms.push(`stress (${l.stress_level}/10)`)
            
            const symptomStr = activeSymptoms.length > 0 ? activeSymptoms.join(", ") : "mild/no specific symptoms"
            
            return `${date}: overall severity ${l.severity ?? "N/A"}/10, mood: ${l.mood_score ?? "N/A"}/10, sleep: ${l.sleep_quality ?? "N/A"}/10. Notable: ${symptomStr}${l.notes ? `. Notes: ${l.notes}` : ""}`
        })
        .join("\n")

    return { ml, retrievedDocs, memory, symptomHistory }
}