"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import ChatHeader from "./ChatHeader"
import CitationCard, { Citation } from "./CitationCard"
import { Send, Sparkles, ArrowDown } from "lucide-react"
import { askAssistant } from "@/lib/mlClient"

/* ================= TYPES ================= */

interface Message {
  role: "user" | "assistant"
  content: string
  timestamp: Date
  citations?: Citation[]
  degraded?: boolean
}

interface Props {
  userId: string
  userEmail: string
}

const SUGGESTIONS = [
  "What are common perimenopause symptoms?",
  "How does sleep affect menopause?",
  "How to manage hot flashes naturally?",
  "Why am I feeling mood swings?"
]

/* ================= COMPONENT ================= */

export default function AssistantChat({ userId, userEmail }: Props) {

  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)
  const [showScrollBtn, setShowScrollBtn] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const isMounted = useRef(true)
  const activeRequest = useRef(0)

  useEffect(() => {
    return () => { isMounted.current = false }
  }, [])

  /* ================= SCROLL ================= */

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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

  /* ================= SAFE RESPONSE ================= */

  const extractSafeResponse = (res: any) => {
    const text =
      res?.reply ||
      res?.answer ||
      ""

    if (typeof text === "string" && text.trim().length > 20) {
      return text.trim()
    }

    return "Your symptoms may be related to hormonal changes during menopause. For accurate guidance, consult a healthcare professional."
  }

  const extractCitations = (res: any): Citation[] => {
    const raw = res?.citations || []

    if (!Array.isArray(raw)) return []

    return raw.map((c: any) =>
      typeof c === "string"
        ? { title: c, snippet: "", url: undefined }
        : {
            title: c?.title || "Clinical Source",
            snippet: c?.snippet || "",
            url: c?.url
          }
    )
  }

  /* ================= SEND ================= */

  const sendMessage = useCallback(async (text?: string) => {

    const messageText = (text || input).trim()
    if (!messageText || loading) return

    const requestId = ++activeRequest.current

    const userMessage: Message = {
      role: "user",
      content: messageText,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInput("")
    setLoading(true)

    try {
      const res = await askAssistant(messageText, userId)

      if (requestId !== activeRequest.current) return

      const assistantMessage: Message = {
        role: "assistant",
        content: extractSafeResponse(res),
        timestamp: new Date(),
        citations: extractCitations(res),
        degraded: res?.degraded === true
      }

      if (isMounted.current) {
        setMessages(prev => [...prev, assistantMessage])
      }

    } catch (err) {

      if (!isMounted.current) return

      const fallbackMessage: Message = {
        role: "assistant",
        content: "Unable to retrieve a detailed response at the moment. Please try again.",
        timestamp: new Date(),
        degraded: true
      }

      setMessages(prev => [...prev, fallbackMessage])

    } finally {
      if (isMounted.current) {
        setLoading(false)
        inputRef.current?.focus()
      }
    }

  }, [input, loading, userId])

  /* ================= INPUT ================= */

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const handleTextareaInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)

    const el = e.target
    el.style.height = "auto"
    el.style.height = Math.min(el.scrollHeight, 160) + "px"
  }

  const isEmptyState = messages.length === 0

  /* ================= UI ================= */

  return (
    <div className="flex flex-col h-[100dvh] bg-[#0a0a0f] text-white">

      <ChatHeader userEmail={userEmail} />

      <div ref={messagesContainerRef} className="flex-1 overflow-y-auto">

        {isEmptyState ? (
          <WelcomeState onSuggestionClick={sendMessage} />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-2">

            <AnimatePresence>
              {messages.map((msg, i) => (
                <MessageBubble key={i} message={msg} />
              ))}
            </AnimatePresence>

            {loading && <TypingIndicator />}

            <div ref={messagesEndRef} />
          </div>
        )}

        {showScrollBtn && (
          <button
            onClick={scrollToBottom}
            className="fixed bottom-28 left-1/2 -translate-x-1/2"
          >
            <ArrowDown size={18} />
          </button>
        )}

      </div>

      <div className="border-t border-white/10 p-4">
        <div className="flex gap-3 bg-white/5 rounded-xl p-3">
          <textarea
            ref={inputRef}
            value={input}
            onChange={handleTextareaInput}
            onKeyDown={handleKeyDown}
            placeholder="Ask about menopause..."
            disabled={loading}
            className="flex-1 bg-transparent outline-none resize-none"
          />
          <button
            onClick={() => sendMessage()}
            disabled={!input.trim() || loading}
            className="bg-purple-600 px-3 rounded-lg"
          >
            <Send size={16} />
          </button>
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
  return <div className="text-gray-400">Thinking...</div>
}
