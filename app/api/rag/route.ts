import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"
import { generateEmbedding } from "@/lib/rag/embedder"
import { supabaseAdmin } from "@/lib/vector/vectorClient"

const pdfParse = require("pdf-parse")

const PDF_DIRECTORY = path.join(process.cwd(), "public/pdfs")

function chunkText(text: string, size = 1200, overlap = 200) {
  const chunks = []
  let start = 0

  while (start < text.length) {
    const end = start + size
    chunks.push(text.slice(start, end))
    start += size - overlap
  }

  return chunks
}

export async function POST() {
  try {
    const files = fs.readdirSync(PDF_DIRECTORY)

    for (const file of files) {
      if (!file.endsWith(".pdf")) continue

      const filePath = path.join(PDF_DIRECTORY, file)
      const buffer = fs.readFileSync(filePath)

      const data = await pdfParse(buffer)
      const cleanedText = data.text.replace(/\s+/g, " ").trim()

      const chunks = chunkText(cleanedText)

      for (let i = 0; i < chunks.length; i++) {
        const embedding = await generateEmbedding(chunks[i])

        await supabaseAdmin.from("medical_documents").insert({
          title: file.replace(".pdf", ""),
          content: chunks[i],
          source: file,
          document_name: file,
          chunk_index: i,
          embedding
        })
      }
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error(error)
    return NextResponse.json({ success: false })
  }
}