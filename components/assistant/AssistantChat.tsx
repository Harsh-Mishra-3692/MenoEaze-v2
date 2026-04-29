"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import ChatHeader from "./ChatHeader"
import CitationCard from "./CitationCard"
import { Send, Sparkles, ArrowDown } from "lucide-react"

interface Message {
    role: "user" | "assistant"
    content: string
    timestamp: Date
}

interface Citation {
    title: string
    source: string
    content: string
}

interface Props {
    userId: string
    userEmail: string
}

const SUGGESTIONS = [
    "What are common perimenopause symptoms?",
    "How does sleep quality affect menopause?",
    "Tips for managing hot flashes naturally",
    "How does mood change during menopause?"
]

export default function AssistantChat({ userId, userEmail }: Props) {
    const [messages, setMessages] = useState<Message[]>([])
    const [citations, setCitations] = useState<Citation[]>([])
    const [input, setInput] = useState("")
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [showScrollBtn, setShowScrollBtn] = useState(false)

    const messagesEndRef = useRef<HTMLDivElement>(null)
    const messagesContainerRef = useRef<HTMLDivElement>(null)
    const inputRef = useRef<HTMLTextAreaElement>(null)

    const scrollToBottom = useCallback((smooth = true) => {
        messagesEndRef.current?.scrollIntoView({
            behavior: smooth ? "smooth" : "instant"
        })
    }, [])

    useEffect(() => {
        scrollToBottom()
    }, [messages, scrollToBottom])

    useEffect(() => {
        const container = messagesContainerRef.current
        if (!container) return

        const handleScroll = () => {
            const { scrollTop, scrollHeight, clientHeight } = container
            setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 100)
        }

        container.addEventListener("scroll", handleScroll)
        return () => container.removeEventListener("scroll", handleScroll)
    }, [])

    const sendMessage = useCallback(
        async (text?: string) => {
            const messageText = text || input.trim()
            if (!messageText || loading) return

            setError(null)

            const userMessage: Message = {
                role: "user",
                content: messageText,
                timestamp: new Date()
            }

            setMessages(prev => [...prev, userMessage])
            setInput("")
            setLoading(true)

            // Reset textarea height
            if (inputRef.current) {
                inputRef.current.style.height = "auto"
            }

            try {
                const res = await fetch("/api/assistant", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ userId, message: messageText })
                })

                if (!res.ok) throw new Error("Failed to get response")

                const data = await res.json()

                if (data.error) throw new Error(data.error)

                const assistantMessage: Message = {
                    role: "assistant",
                    content: data.reply,
                    timestamp: new Date()
                }

                setMessages(prev => [...prev, assistantMessage])
                setCitations(data.citations || [])
            } catch (err) {
                setError(
                    err instanceof Error ? err.message : "Something went wrong"
                )
            } finally {
                setLoading(false)
                inputRef.current?.focus()
            }
        },
        [input, loading, userId]
    )

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault()
            sendMessage()
        }
    }

    const handleTextareaInput = (
        e: React.ChangeEvent<HTMLTextAreaElement>
    ) => {
        setInput(e.target.value)
        // Auto-resize
        const el = e.target
        el.style.height = "auto"
        el.style.height = Math.min(el.scrollHeight, 160) + "px"
    }

    const isEmptyState = messages.length === 0

    return (
        <div className="flex flex-col h-[100dvh] bg-[#0a0a0f] text-white overflow-hidden">
            {/* Ambient background gradients */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden">
                <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-purple-900/20 blur-[128px] animate-[drift_20s_ease-in-out_infinite_alternate]" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-pink-900/15 blur-[128px] animate-[drift_25s_ease-in-out_infinite_alternate-reverse]" />
                <div className="absolute top-[40%] left-[50%] w-[400px] h-[400px] rounded-full bg-indigo-900/10 blur-[128px] animate-[drift_30s_ease-in-out_infinite_alternate]" />
            </div>

            {/* Header */}
            <ChatHeader userEmail={userEmail} />

            {/* Messages Area */}
            <div
                ref={messagesContainerRef}
                className="flex-1 overflow-y-auto relative z-10 scrollbar-thin"
            >
                {isEmptyState ? (
                    <WelcomeState
                        onSuggestionClick={(s) => sendMessage(s)}
                    />
                ) : (
                    <div className="max-w-3xl mx-auto px-4 py-6 space-y-1">
                        <AnimatePresence mode="popLayout">
                            {messages.map((msg, idx) => (
                                <MessageBubble key={idx} message={msg} />
                            ))}
                        </AnimatePresence>

                        {loading && <TypingIndicator />}

                        {error && (
                            <motion.div
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex items-center gap-3 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-red-400 text-sm"
                            >
                                <span className="flex-1">{error}</span>
                                <button
                                    onClick={() => {
                                        setError(null)
                                        const lastUserMsg = [...messages]
                                            .reverse()
                                            .find(m => m.role === "user")
                                        if (lastUserMsg) sendMessage(lastUserMsg.content)
                                    }}
                                    className="text-xs bg-red-500/20 hover:bg-red-500/30 px-3 py-1.5 rounded-lg transition-colors"
                                >
                                    Retry
                                </button>
                            </motion.div>
                        )}

                        {/* Citations */}
                        {citations.length > 0 && (
                            <motion.div
                                initial={{ opacity: 0, y: 8 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="pt-3 space-y-2"
                            >
                                <p className="text-xs text-gray-500 uppercase tracking-wider font-medium px-1">
                                    Sources
                                </p>
                                {citations.map((c, i) => (
                                    <CitationCard key={i} citation={c} />
                                ))}
                            </motion.div>
                        )}

                        <div ref={messagesEndRef} />
                    </div>
                )}

                {/* Scroll to bottom button */}
                <AnimatePresence>
                    {showScrollBtn && (
                        <motion.button
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            onClick={() => scrollToBottom()}
                            className="fixed bottom-28 left-1/2 -translate-x-1/2 z-20 w-9 h-9 rounded-full bg-white/10 backdrop-blur-md border border-white/10 flex items-center justify-center text-gray-400 hover:text-white hover:bg-white/15 transition-all shadow-lg"
                        >
                            <ArrowDown size={16} />
                        </motion.button>
                    )}
                </AnimatePresence>
            </div>

            {/* Input Area */}
            <div className="relative z-10 border-t border-white/[0.06] bg-[#0a0a0f]/80 backdrop-blur-xl">
                <div className="max-w-3xl mx-auto px-4 py-4">
                    <div className="flex items-end gap-3 bg-white/[0.05] border border-white/[0.08] rounded-2xl px-4 py-3 focus-within:border-purple-500/40 focus-within:bg-white/[0.07] transition-all duration-300">
                        <textarea
                            ref={inputRef}
                            rows={1}
                            value={input}
                            onChange={handleTextareaInput}
                            onKeyDown={handleKeyDown}
                            placeholder="Ask me anything about menopause…"
                            disabled={loading}
                            className="flex-1 bg-transparent text-white/90 placeholder-gray-500 text-sm leading-relaxed resize-none focus:outline-none disabled:opacity-50 max-h-40"
                        />
                        <button
                            onClick={() => sendMessage()}
                            disabled={!input.trim() || loading}
                            className="flex-shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-white disabled:opacity-30 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-purple-500/25 active:scale-95 transition-all duration-200"
                        >
                            <Send size={16} />
                        </button>
                    </div>
                    <p className="text-[11px] text-gray-600 text-center mt-2.5 tracking-wide">
                        MenoEaze AI provides health insights, not medical diagnosis.
                        Always consult your doctor.
                    </p>
                </div>
            </div>
        </div>
    )
}

/* ============================================================
   SUB-COMPONENTS
   ============================================================ */

function WelcomeState({
    onSuggestionClick
}: {
    onSuggestionClick: (s: string) => void
}) {
    return (
        <div className="flex flex-col items-center justify-center h-full px-6">
            {/* Animated Orb */}
            <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="relative mb-10"
            >
                <div className="w-28 h-28 rounded-full bg-gradient-to-br from-purple-500/30 via-pink-500/20 to-indigo-500/30 blur-2xl absolute inset-0 animate-pulse" />
                <div className="relative w-28 h-28 rounded-full bg-gradient-to-br from-purple-600/40 via-pink-500/30 to-indigo-600/40 border border-white/10 flex items-center justify-center backdrop-blur-sm">
                    <Sparkles size={36} className="text-purple-300/80" />
                </div>
            </motion.div>

            <motion.h1
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.6 }}
                className="text-2xl font-light text-white/90 mb-2 tracking-tight"
            >
                How can I help you today?
            </motion.h1>

            <motion.p
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.35, duration: 0.6 }}
                className="text-sm text-gray-500 mb-10 max-w-md text-center leading-relaxed"
            >
                I&apos;m your personalized menopause health companion — ask me
                anything about symptoms, wellness, or your health data.
            </motion.p>

            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.6 }}
                className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg"
            >
                {SUGGESTIONS.map((s, i) => (
                    <button
                        key={i}
                        onClick={() => onSuggestionClick(s)}
                        className="text-left text-sm text-gray-400 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] hover:border-purple-500/30 rounded-xl px-4 py-3.5 transition-all duration-200 hover:text-gray-200 group"
                    >
                        <span className="inline-block mr-2 text-purple-400/60 group-hover:text-purple-400 transition-colors">
                            →
                        </span>
                        {s}
                    </button>
                ))}
            </motion.div>
        </div>
    )
}

function MessageBubble({ message }: { message: Message }) {
    const isUser = message.role === "user"

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className={`flex ${isUser ? "justify-end" : "justify-start"} py-1.5`}
        >
            <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${isUser
                    ? "bg-gradient-to-br from-purple-600/90 to-pink-600/80 text-white/95 rounded-br-md"
                    : "bg-white/[0.06] border border-white/[0.06] text-gray-300 rounded-bl-md"
                    }`}
            >
                {isUser ? (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                ) : (
                    <div className="prose prose-invert prose-sm max-w-none prose-p:my-1.5 prose-headings:text-gray-200 prose-strong:text-gray-200 prose-a:text-purple-400 prose-code:text-pink-400 prose-code:bg-white/5 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-pre:bg-white/5 prose-pre:border prose-pre:border-white/10 prose-li:my-0.5 prose-ul:my-2 prose-ol:my-2">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {message.content}
                        </ReactMarkdown>
                    </div>
                )}
            </div>
        </motion.div>
    )
}

function TypingIndicator() {
    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start py-1.5"
        >
            <div className="bg-white/[0.06] border border-white/[0.06] rounded-2xl rounded-bl-md px-5 py-4 flex items-center gap-1.5">
                {[0, 1, 2].map(i => (
                    <motion.span
                        key={i}
                        className="w-2 h-2 rounded-full bg-purple-400/60"
                        animate={{ opacity: [0.3, 1, 0.3], scale: [0.85, 1, 0.85] }}
                        transition={{
                            duration: 1.2,
                            repeat: Infinity,
                            delay: i * 0.2,
                            ease: "easeInOut"
                        }}
                    />
                ))}
            </div>
        </motion.div>
    )
}