"use client"

import { useState } from "react"
import { ChevronDown, ChevronUp, FileText } from "lucide-react"

interface Citation {
    title: string
    source: string
    content: string
    similarity?: number
}

export default function CitationCard({ citation }: { citation: Citation }) {
    const [expanded, setExpanded] = useState(false)

    return (
        <button
            onClick={() => setExpanded(p => !p)}
            className="w-full text-left bg-white/[0.04] hover:bg-white/[0.06] border border-white/[0.06] rounded-xl px-4 py-3 transition-all duration-200 group"
        >
            <div className="flex items-start gap-3">
                <div className="mt-0.5 w-7 h-7 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center flex-shrink-0">
                    <FileText size={13} className="text-purple-400/70" />
                </div>

                <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 overflow-hidden">
                            <p className="text-sm text-gray-300 font-medium truncate">
                                {citation.title || citation.source}
                            </p>
                            {citation.similarity !== undefined && (
                                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 flex-shrink-0">
                                    {Math.round(citation.similarity * 100)}% match
                                </span>
                            )}
                        </div>
                        {expanded ? (
                            <ChevronUp size={14} className="text-gray-600 flex-shrink-0" />
                        ) : (
                            <ChevronDown size={14} className="text-gray-600 flex-shrink-0" />
                        )}
                    </div>

                    <p className="text-[11px] text-gray-600 mt-0.5">
                        {citation.source}
                    </p>

                    {expanded && (
                        <p className="text-xs text-gray-500 mt-2 leading-relaxed border-t border-white/[0.04] pt-2">
                            {citation.content.slice(0, 300)}
                            {citation.content.length > 300 && "…"}
                        </p>
                    )}
                </div>
            </div>
        </button>
    )
}