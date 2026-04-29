'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { supabase } from '@/lib/supabase'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type MessageStatus = 'sending' | 'sent' | 'failed'

type ChatMessage = {
  id: string
  role: 'user' | 'assistant'
  content: string
  status: MessageStatus
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
  const abortRef = useRef<AbortController | null>(null)

  /* ------------------ Load Messages from DB ------------------ */
  useEffect(() => {
    const loadMessages = async () => {
      const { data } = await supabase
        .from('chat_messages')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: true })

      if (data) {
        setMessages(data)
      }
    }

    loadMessages()
  }, [userId])

  /* ------------------ Auto Scroll ------------------ */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  /* ------------------ Send Handler ------------------ */
  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return

    const now = Date.now()
    if (now - lastSent < RATE_LIMIT_MS) return
    setLastSent(now)

    const tempId = crypto.randomUUID()

    const optimisticMessage: ChatMessage = {
      id: tempId,
      role: 'user',
      content: input.trim(),
      status: 'sending',
      created_at: new Date().toISOString()
    }

    setMessages(prev => [...prev, optimisticMessage])
    setInput('')
    setLoading(true)

    abortRef.current?.abort()
    abortRef.current = new AbortController()

    try {
      const {
        data: { session }
      } = await supabase.auth.getSession()

      if (!session?.access_token) {
        throw new Error('Authentication required')
      }

      /* Save user message in DB */
      await supabase.from('chat_messages').insert({
        id: tempId,
        user_id: userId,
        role: 'user',
        content: optimisticMessage.content,
        status: 'sent'
      })

      /* Update optimistic status */
      setMessages(prev =>
        prev.map(m =>
          m.id === tempId ? { ...m, status: 'sent' } : m
        )
      )

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${session.access_token}`
        },
        body: JSON.stringify({
          userId,
          message: optimisticMessage.content
        }),
        signal: abortRef.current.signal
      })

      if (!response.ok) {
        throw new Error('AI response failed')
      }

      const data = await response.json()

      const assistantId = crypto.randomUUID()

      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: 'assistant',
        content: data.reply,
        status: 'sent',
        created_at: new Date().toISOString()
      }

      setMessages(prev => [...prev, assistantMessage])

      await supabase.from('chat_messages').insert({
        id: assistantId,
        user_id: userId,
        role: 'assistant',
        content: assistantMessage.content,
        status: 'sent'
      })
    } catch (err) {
      /* Rollback optimistic update */
      setMessages(prev =>
        prev.map(m =>
          m.id === tempId ? { ...m, status: 'failed' } : m
        )
      )
    } finally {
      setLoading(false)
    }
  }, [input, loading, lastSent, userId])

  /* ------------------ Retry Failed ------------------ */
  const retryMessage = async (msg: ChatMessage) => {
    setInput(msg.content)
    setMessages(prev => prev.filter(m => m.id !== msg.id))
  }

  /* ------------------ Status Icon ------------------ */
  const renderStatus = (msg: ChatMessage) => {
    if (msg.role !== 'user') return null

    if (msg.status === 'sending')
      return <span className="text-xs ml-2">✓</span>

    if (msg.status === 'sent')
      return <span className="text-xs ml-2">✓✓</span>

    if (msg.status === 'failed')
      return (
        <button
          onClick={() => retryMessage(msg)}
          className="text-xs ml-2 text-red-500"
        >
          ⚠ Retry
        </button>
      )

    return null
  }

  return (
    <div className="bg-white rounded-2xl shadow-lg flex flex-col h-[550px]">
      {/* Header */}
      <div className="p-5 border-b bg-gradient-to-r from-purple-600 to-pink-500 text-white rounded-t-2xl">
        <h3 className="font-semibold">
          Personal Support Assistant
        </h3>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-gray-50">
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex ${
              msg.role === 'user'
                ? 'justify-end'
                : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-2xl ${
                msg.role === 'user'
                  ? 'bg-gradient-to-r from-purple-600 to-pink-500 text-white rounded-br-sm'
                  : 'bg-white text-gray-900 shadow-sm rounded-bl-sm border border-gray-200'
              }`}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content}
              </ReactMarkdown>

              {renderStatus(msg)}
            </div>
          </div>
        ))}

        {loading && (
          <div className="text-sm text-gray-500">
            Assistant is typing...
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t bg-white rounded-b-2xl">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSend()
            }
          }}
          className="w-full border-2 border-gray-200 rounded-xl px-4 py-2 focus:border-purple-500 focus:outline-none resize-none"
          placeholder="Describe your question..."
          rows={2}
        />

        <button
          onClick={handleSend}
          disabled={!input.trim() || loading}
          className="w-full mt-3 bg-gradient-to-r from-purple-600 to-pink-500 text-white px-6 py-3 rounded-xl font-semibold disabled:opacity-50"
        >
          {loading ? 'Sending...' : 'Send Message'}
        </button>
      </div>
    </div>
  )
}