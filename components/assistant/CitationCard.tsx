"use client"

import { useState, useMemo } from "react"
import { ChevronDown, ChevronUp, FileText } from "lucide-react"

/* ================= TYPES ================= */

export interface Citation {
  title?: string
  source?: string
  content?: string
  similarity?: number
}

/* ================= NORMALIZER ================= */

function normalizeCitation(input: any): Required<Citation> {
  if (!input) {
    return {
      title: "Clinical Source",
      source: "Unknown",
      content: "",
      similarity: 0
    }
  }

  // If backend sends string
  if (typeof input === "string") {
    return {
      title: input.slice(0, 60),
      source: "Clinical Source",
      content: input,
      similarity: 0
    }
  }

  return {
    title:
      typeof input.title === "string" && input.title.trim()
        ? input.title
        : input.source || "Clinical Source",

    source:
      typeof input.source === "string" && input.source.trim()
        ? input.source
        : "Clinical Source",

    content:
      typeof input.content === "string"
        ? input.content
        : "",

    similarity:
      typeof input.similarity === "number" &&
      isFinite(input.similarity)
        ? input.similarity
        : 0
  }
}

/* ================= COMPONENT ================= */

export default function CitationCard({ citation }: { citation: Citation }) {

  const [expanded, setExpanded] = useState(false)

  const safeCitation = useMemo(
    () => normalizeCitation(citation),
    [citation]
  )

  const hasContent = safeCitation.content.length > 0

  const previewText =
    safeCitation.content.length > 300
      ? safeCitation.content.slice(0, 300) + "…"
      : safeCitation.content

  return (
    <button
      onClick={() => setExpanded(p => !p)}
      className="w-full text-left bg-white/[0.04] hover:bg-white/[0.06] border border-white/[0.06] rounded-xl px-4 py-3 transition-all duration-200 group"
    >
      <div className="flex items-start gap-3">

        {/* ICON */}
        <div className="mt-0.5 w-7 h-7 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center flex-shrink-0">
          <FileText size={13} className="text-purple-400/70" />
        </div>

        {/* CONTENT */}
        <div className="flex-1 min-w-0">

          {/* HEADER */}
          <div className="flex items-center justify-between gap-2">

            <div className="flex items-center gap-2 overflow-hidden">
              <p className="text-sm text-gray-300 font-medium truncate">
                {safeCitation.title}
              </p>

              {safeCitation.similarity > 0 && (
                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 flex-shrink-0">
                  {Math.round(safeCitation.similarity * 100)}% match
                </span>
              )}
            </div>

            {expanded ? (
              <ChevronUp size={14} className="text-gray-600 flex-shrink-0" />
            ) : (
              <ChevronDown size={14} className="text-gray-600 flex-shrink-0" />
            )}

          </div>

          {/* SOURCE */}
          <p className="text-[11px] text-gray-600 mt-0.5">
            {safeCitation.source}
          </p>

          {/* CONTENT */}
          {expanded && hasContent && (
            <p className="text-xs text-gray-500 mt-2 leading-relaxed border-t border-white/[0.04] pt-2">
              {previewText}
            </p>
          )}

        </div>
      </div>
    </button>
  )
}
