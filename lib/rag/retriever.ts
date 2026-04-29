import { generateEmbedding } from "./embedder"
import { supabaseAdmin } from "@/lib/vector/vectorClient"

interface RetrievedDoc {
  title: string
  content: string
  source: string
  similarity: number
}

export async function retrieveRelevantDocs(
  query: string
): Promise<RetrievedDoc[]> {
  const queryEmbedding = await generateEmbedding(query)

  const { data, error } = await supabaseAdmin.rpc(
    "match_medical_documents",
    {
      query_embedding: queryEmbedding,
      match_count: 5
    }
  )

  if (error || !data) {
    console.error(error)
    return []
  }

  return data.map((doc: any) => ({
    title: doc.title,
    content: doc.content,
    source: doc.source,
    similarity: doc.similarity
  }))
}