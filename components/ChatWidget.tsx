'use client'

import { useState, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

interface Props {
  userId: string
}

const RATE_LIMIT_MS = 3000

export default function ChatWidget({ userId }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastSent, setLastSent] = useState(0)

  const bottomRef = useRef<HTMLDivElement>(null)

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return

    const now = Date.now()
    if (now - lastSent < RATE_LIMIT_MS) return
    setLastSent(now)

    const tempId = crypto.randomUUID()

    const userMsg: ChatMessage = {
      id: tempId,
      role: 'user',
      content: input.trim(),
      created_at: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      // ALL chat goes through /api/assistant -> backend /run predict
      const res = await fetch('/api/assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userId, message: userMsg.content })
      })

      const data = await res.json()

      const assistantMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: data.reply || "I'm sorry, I couldn't process that right now.",
        created_at: new Date().toISOString()
      }

      setMessages(prev => [...prev, assistantMsg])
    } catch {
      const errorMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Something went wrong. Please try again.',
        created_at: new Date().toISOString()
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setLoading(false)
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
    }
  }, [input, loading, lastSent, userId])

  return (
    <div className="bg-white rounded-2xl shadow-lg flex flex-col h-[550px]">
      <div className="p-5 border-b bg-gradient-to-r from-purple-600 to-pink-500 text-white rounded-t-2xl">
        <h3 className="font-semibold">Personal Support Assistant</h3>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-2xl ${
              msg.role === 'user'
                ? 'bg-gradient-to-r from-purple-600 to-pink-500 text-white rounded-br-sm'
                : 'bg-white text-gray-900 shadow-sm rounded-bl-sm border border-gray-200'
            }`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
            </div>
          </div>
        ))}

        {loading && <div className="text-sm text-gray-500">Assistant is typing...</div>}
        <div ref={bottomRef} />
      </div>

      <div className="p-4 border-t bg-white rounded-b-2xl">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
          }}
          className="w-full border-2 border-gray-200 rounded-xl px-4 py-2 focus:border-purple-500 focus:outline-none resize-none"
          placeholder="Ask about your symptoms..."
          rows={2}
        />
        <button onClick={handleSend} disabled={!input.trim() || loading}
          className="w-full mt-3 bg-gradient-to-r from-purple-600 to-pink-500 text-white px-6 py-3 rounded-xl font-semibold disabled:opacity-50 transition">
          {loading ? 'Sending...' : 'Send Message'}
        </button>
      </div>
    </div>
  )
}