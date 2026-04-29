// lib/assistant/contextBuilder.ts

import { analyzeUserData } from "@/lib/ml"
import { retrieveRelevantDocs } from "@/lib/rag/retriever"
import { supabaseAdmin } from "@/lib/vector/vectorClient"

export async function buildAssistantContext(userId: string, message: string) {
    // Fetch symptom logs with all fields for rich context
    const { data: logs } = await supabaseAdmin
        .from("symptom_logs")
        .select("symptom_type, severity, mood, sleep, notes, created_at")
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
            return `${date}: ${l.symptom_type} (severity ${l.severity}/10), mood: ${l.mood || "N/A"}, sleep: ${l.sleep || "N/A"}/10${l.notes ? `, notes: ${l.notes}` : ""}`
        })
        .join("\n")

    return { ml, retrievedDocs, memory, symptomHistory }
}