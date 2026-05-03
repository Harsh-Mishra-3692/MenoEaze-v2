import { NextResponse } from "next/server"

// DISABLED: This endpoint violated architectural constraints:
// 1. Used supabaseAdmin to query symptom_logs directly (DB leak)
// 2. Used local Groq LLM instead of backend RAG pipeline
// All intelligence MUST flow through POST /run on the backend.

export async function POST() {
    return NextResponse.json({
        summary:
            "Keep tracking your symptoms daily — your data helps the AI engine provide more accurate and personalized insights over time. 💜"
    })
}
