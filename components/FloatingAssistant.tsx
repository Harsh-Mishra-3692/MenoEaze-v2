'use client'

import { useRouter } from 'next/navigation'
import { useEffect, useState, useRef, useCallback } from 'react'
import { supabase } from '@/lib/supabase'

interface Props {
  userId?: string
}

export default function FloatingAssistant({ userId }: Props) {
  const router = useRouter()

  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null)
  const [hasUnread, setHasUnread] = useState(false)
  const [animate, setAnimate] = useState(false)
  const [idle, setIdle] = useState(false)
  const [contextVisible, setContextVisible] = useState(true)

  const inactivityTimer = useRef<NodeJS.Timeout | null>(null)

  /* ------------------ Auth Check ------------------ */
  useEffect(() => {
    const checkSession = async () => {
      const {
        data: { session }
      } = await supabase.auth.getSession()
      setIsAuthenticated(!!session)
    }

    checkSession()

    const {
      data: { subscription }
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsAuthenticated(!!session)
    })

    return () => subscription.unsubscribe()
  }, [])

  /* ------------------ Context Awareness ------------------ */
  useEffect(() => {
    const checkContext = async () => {
      if (!userId) return

      // Check recent symptom logs for severity-based context
      const { data } = await supabase
        .from('symptom_logs')
        .select('severity')
        .eq('user_id', userId)
        .eq('is_deleted', false)
        .order('created_at', { ascending: false })
        .limit(5)

      if (!data || data.length === 0) {
        setContextVisible(true)
        return
      }

      // Always visible, but could be conditionally hidden
      setContextVisible(true)
    }

    checkContext()
  }, [userId])

  /* ------------------ Real-time Subscription ------------------ */
  useEffect(() => {
    if (!userId) return

    const channel = supabase
      .channel('assistant-realtime')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'chat_messages',
          filter: `user_id=eq.${userId}`
        },
        payload => {
          if (payload.new.role === 'assistant') {
            setHasUnread(true)
            triggerAnimation()
          }
        }
      )
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [userId])

  /* ------------------ Animation Trigger ------------------ */
  const triggerAnimation = () => {
    setAnimate(true)
    setTimeout(() => setAnimate(false), 4000)
  }

  /* ------------------ Idle Detection ------------------ */
  useEffect(() => {
    const resetTimer = () => {
      setIdle(false)
      if (inactivityTimer.current) clearTimeout(inactivityTimer.current)

      inactivityTimer.current = setTimeout(() => {
        setIdle(true)
      }, 10000)
    }

    window.addEventListener('mousemove', resetTimer)
    window.addEventListener('keydown', resetTimer)

    resetTimer()

    return () => {
      window.removeEventListener('mousemove', resetTimer)
      window.removeEventListener('keydown', resetTimer)
    }
  }, [])

  /* ------------------ Click Handler ------------------ */
  const handleClick = useCallback(async () => {
    const {
      data: { session }
    } = await supabase.auth.getSession()

    if (!session) {
      router.push('/login')
      return
    }

    setHasUnread(false)
    router.push('/assistant')
  }, [router])

  if (!isAuthenticated || !contextVisible) return null

  return (
    <div className="fixed bottom-6 right-6 z-50 group">
      {/* Tooltip */}
      {idle && !hasUnread && (
        <div className="absolute bottom-20 right-0 bg-gray-900 text-white text-xs px-3 py-2 rounded-lg shadow-md opacity-90 animate-fade-in">
          Need help today?
        </div>
      )}

      <button
        onClick={handleClick}
        aria-label="Open Assistant"
        className={`relative transition-all duration-300 ${idle ? 'scale-90 opacity-80' : 'scale-100'
          }`}
      >
        {/* Unread Dot */}
        {hasUnread && (
          <span className="absolute top-0 right-0 w-4 h-4 bg-red-500 rounded-full border-2 border-white animate-pulse"></span>
        )}

        {/* Glow */}
        <div
          className={`absolute inset-0 rounded-full blur-lg opacity-60 transition ${animate
            ? 'bg-gradient-to-r from-blue-500 to-cyan-400'
            : 'bg-gradient-to-r from-purple-600 to-pink-500'
            }`}
        />

        {/* Main Button */}
        <div
          className={`relative w-16 h-16 rounded-full flex items-center justify-center shadow-xl text-white text-2xl transition-transform duration-300 ${animate
            ? 'bg-gradient-to-r from-blue-500 to-cyan-400 animate-bounce'
            : 'bg-gradient-to-r from-purple-600 to-pink-500'
            } hover:scale-110 active:scale-95`}
        >
          🤖
        </div>
      </button>
    </div>
  )
}