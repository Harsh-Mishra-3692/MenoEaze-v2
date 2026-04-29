"use client"

import Link from "next/link"
import { ArrowLeft, Sparkles } from "lucide-react"

interface Props {
    userEmail: string
}

export default function ChatHeader({ userEmail }: Props) {
    const initial = userEmail.charAt(0).toUpperCase() || "U"

    return (
        <header className="relative z-10 flex items-center justify-between px-5 py-3.5 border-b border-white/[0.06] bg-[#0a0a0f]/60 backdrop-blur-xl">
            {/* Left — Back */}
            <Link
                href="/dashboard"
                className="flex items-center gap-2 text-gray-500 hover:text-gray-300 transition-colors group"
            >
                <ArrowLeft
                    size={18}
                    className="group-hover:-translate-x-0.5 transition-transform"
                />
                <span className="text-sm hidden sm:inline">Back</span>
            </Link>

            {/* Center — Branding */}
            <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2.5">
                <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-purple-500/30 to-pink-500/30 border border-white/10 flex items-center justify-center">
                    <Sparkles size={14} className="text-purple-400" />
                </div>
                <div>
                    <h1 className="text-sm font-medium text-white/90 tracking-tight leading-none">
                        MenoEaze AI
                    </h1>
                    <p className="text-[10px] text-emerald-400/70 tracking-wider mt-0.5">
                        ● Online
                    </p>
                </div>
            </div>

            {/* Right — User avatar */}
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500/40 to-pink-500/40 border border-white/10 flex items-center justify-center text-xs font-medium text-white/80">
                {initial}
            </div>
        </header>
    )
}