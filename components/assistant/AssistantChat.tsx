"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import ChatHeader from "./ChatHeader"
import CitationCard, { Citation } from "./CitationCard"
import { Send, Sparkles, ArrowDown, Loader2 } from "lucide-react"
import { askAssistant } from "@/lib/mlClient"

/* ================= TYPES ================= */

interface Message {
  role: "user" | "assistant"
  content: string
  timestamp: Date
  citations?: Citation[]
  degradedReasons?: string[]
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

/* ================= COMPONENT ================= */

export default function AssistantChat({ userId, userEmail }: Props) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showScrollBtn, setShowScrollBtn] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const isMounted = useRef(true)
  const activeRequest = useRef(0)

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
    }
  }, [])

  /* ================= SAFE SCROLL ================= */

  const scrollToBottom = useCallback((smooth = true) => {
    messagesEndRef.current?.scrollIntoView({
      behavior: smooth ? "smooth" : "auto"
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

  /* ================= RESPONSE NORMALIZER ================= */

  const normalizeResponse = (data: any) => {
    const safeText =
      typeof data?.reply === "string" && data.reply.trim().length > 0
        ? data.reply
        : "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights..."

    return {
      content: safeText,
      citations: Array.isArray(data?.citations) ? data.citations : [],
      degradedReasons: Array.isArray(data?.degradedReasons)
        ? data.degradedReasons
        : undefined
    }
  }

  /* ================= SEND ================= */

  const sendMessage = useCallback(
    async (text?: string) => {
      const messageText = (text || input).trim()

      if (!messageText || loading) return

      const requestId = ++activeRequest.current
      setError(null)

      const userMessage: Message = {
        role: "user",
        content: messageText,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, userMessage])
      setInput("")
      setLoading(true)

      if (inputRef.current) inputRef.current.style.height = "auto"
      try {
        const mlResponse = await askAssistant(messageText, userId)

        // Ignore stale responses
        if (requestId !== activeRequest.current) return

        const assistantMessage: Message = {
          role: "assistant",
          content: mlResponse.answer || "I'm analyzing your symptoms. Based on available clinical context, here are some relevant insights...",
          timestamp: new Date(),
          citations: Array.isArray((mlResponse.raw as any)?.citations) ? (mlResponse.raw as any).citations : [],
          degradedReasons: undefined
        }

        if (isMounted.current) {
          setMessages(prev => [...prev, assistantMessage])
        }

      } catch (err: any) {
        if (!isMounted.current) return

        const fallbackMessage: Message = {
          role: "assistant",
          content: "System encountered a delay but your data has been received. Please consult your physician if your symptoms are severe.",
          timestamp: new Date(),
          degradedReasons: ["network"]
        }

        setMessages(prev => [...prev, fallbackMessage])
        setError(err?.message || "Request failed")
      } finally {
        if (isMounted.current) {
          setLoading(false)
          inputRef.current?.focus()
        }
      }
    },
    [input, loading, userId]
  )

  /* ================= INPUT ================= */

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

    const el = e.target
    el.style.height = "auto"
    el.style.height = Math.min(el.scrollHeight, 160) + "px"
  }

  const isEmptyState = messages.length === 0

  /* ================= UI ================= */

  return (
    <div className="flex flex-col h-[100dvh] bg-[#0a0a0f] text-white overflow-hidden">

      <ChatHeader userEmail={userEmail} />

      <div
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto relative z-10"
      >
        {isEmptyState ? (
          <WelcomeState onSuggestionClick={sendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-1">
            <AnimatePresence mode="popLayout">
              {messages.map((msg, idx) => (
                <MessageBubble key={idx} message={msg} />
              ))}
            </AnimatePresence>

            {loading && <TypingIndicator />}

            {error && (
              <div className="text-red-400 text-sm px-4">
                {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}

        {showScrollBtn && (
          <button
            onClick={() => scrollToBottom()}
            className="fixed bottom-28 left-1/2 -translate-x-1/2 w-9 h-9 rounded-full bg-white/10"
          >
            <ArrowDown size={16} />
          </button>
        )}
      </div>

      <div className="border-t border-white/[0.06] bg-[#0a0a0f]/80">
        <div className="max-w-3xl mx-auto px-4 py-4">
          <div className="flex items-end gap-3 bg-white/[0.05] rounded-2xl px-4 py-3">
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={handleTextareaInput}
              onKeyDown={handleKeyDown}
              placeholder="Ask about menopause..."
              disabled={loading}
              className="flex-1 bg-transparent text-white text-sm resize-none outline-none"
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              className="w-9 h-9 bg-purple-600 rounded-xl"
            >
              <Send size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ================= SUB COMPONENTS ================= */

function WelcomeState({ onSuggestionClick }: any) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-4">
      <Sparkles size={40} />
      {SUGGESTIONS.map((s: string, i: number) => (
        <button key={i} onClick={() => onSuggestionClick(s)}>
          {s}
        </button>
      ))}
    </div>
  )
}

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user"

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className="max-w-[80%] p-3 rounded-xl">
        {isUser ? (
          message.content
        ) : (
          <>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>

            {message.citations?.map((c, i) => (
              <CitationCard key={i} citation={c} />
            ))}
          </>
        )}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return <div className="text-gray-400">...</div>
}