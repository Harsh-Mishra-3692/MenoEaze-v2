'use client'

import { useRouter } from 'next/navigation'
import { useEffect, useState, useRef, useCallback } from 'react'
import { supabase } from '@/lib/supabase'

interface Props {
  userId?: string
  doctorRecommend?: boolean
  doctorUrgency?: string
}

export default function FloatingAssistant({ userId, doctorRecommend: propRecommend = false, doctorUrgency: propUrgency = 'none' }: Props) {
  const router = useRouter()

  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null)
  const [idle, setIdle] = useState(false)
  const [doctorRecommend, setDoctorRecommend] = useState(propRecommend)
  const [doctorUrgency, setDoctorUrgency] = useState(propUrgency)

  // Feature 8: Poll localStorage for doctor urgency (set by AnalysisCard)
  useEffect(() => {
    const readUrgency = () => {
      try {
        const raw = localStorage.getItem('menoeaze_doctor_urgency')
        if (raw) {
          const parsed = JSON.parse(raw)
          if (typeof parsed.recommend === 'boolean') setDoctorRecommend(parsed.recommend)
          if (typeof parsed.urgency === 'string') setDoctorUrgency(parsed.urgency)
        }
      } catch {}
    }
    readUrgency()
    const interval = setInterval(readUrgency, 3000)
    return () => clearInterval(interval)
  }, [])

  const inactivityTimer = useRef<NodeJS.Timeout | null>(null)

  /* Auth check — supabase.auth is allowed */
  useEffect(() => {
    const checkSession = async () => {
      const { data: { session } } = await supabase.auth.getSession()
      setIsAuthenticated(!!session)
    }

    checkSession()

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsAuthenticated(!!session)
    })

    return () => subscription.unsubscribe()
  }, [])

  /* Idle Detection */
  useEffect(() => {
    const resetTimer = () => {
      setIdle(false)
      if (inactivityTimer.current) clearTimeout(inactivityTimer.current)
      inactivityTimer.current = setTimeout(() => setIdle(true), 10000)
    }

    window.addEventListener('mousemove', resetTimer)
    window.addEventListener('keydown', resetTimer)
    resetTimer()

    return () => {
      window.removeEventListener('mousemove', resetTimer)
      window.removeEventListener('keydown', resetTimer)
    }
  }, [])

  const handleClick = useCallback(async () => {
    const { data: { session } } = await supabase.auth.getSession()
    if (!session) { router.push('/login'); return }
    router.push('/assistant')
  }, [router])

  if (!isAuthenticated) return null

  // Feature 8: Urgency-driven visual state
  const isUrgent = doctorRecommend && doctorUrgency === 'immediate'
  const isWarning = doctorRecommend && doctorUrgency !== 'immediate' && doctorUrgency !== 'none'

  const glowColor = isUrgent
    ? 'bg-gradient-to-r from-red-500 to-orange-400'
    : isWarning
      ? 'bg-gradient-to-r from-amber-500 to-orange-400'
      : 'bg-gradient-to-r from-purple-600 to-pink-500'

  const buttonColor = isUrgent
    ? 'bg-gradient-to-r from-red-500 to-orange-400'
    : isWarning
      ? 'bg-gradient-to-r from-amber-500 to-orange-400'
      : 'bg-gradient-to-r from-purple-600 to-pink-500'

  const tooltipText = isUrgent
    ? '🚨 Urgent: Please consult your doctor'
    : isWarning
      ? '🩺 Doctor consultation recommended'
      : 'Need help today?'

  return (
    <div className="fixed bottom-6 right-6 z-50 group">
      {/* Tooltip */}
      {(idle || isUrgent) && (
        <div className={`absolute bottom-20 right-0 text-white text-xs px-3 py-2 rounded-lg shadow-md opacity-90 animate-fade-in ${
          isUrgent ? 'bg-red-700' : 'bg-gray-900'
        }`}>
          {tooltipText}
        </div>
      )}

      <button onClick={handleClick} aria-label="Open Assistant"
        className={`relative transition-all duration-300 ${idle && !isUrgent ? 'scale-90 opacity-80' : 'scale-100'}`}>

        {/* Urgency pulse ring */}
        {isUrgent && (
          <span className="absolute -inset-1 rounded-full bg-red-500/40 animate-ping" />
        )}

        {/* Unread dot for warnings */}
        {isWarning && !isUrgent && (
          <span className="absolute top-0 right-0 w-4 h-4 bg-amber-500 rounded-full border-2 border-white animate-pulse z-10" />
        )}

        {/* Glow */}
        <div className={`absolute inset-0 rounded-full blur-lg opacity-60 transition ${glowColor}`} />

        {/* Main Button */}
        <div className={`relative w-16 h-16 rounded-full flex items-center justify-center shadow-xl text-white text-2xl transition-transform duration-300 ${buttonColor} ${
          isUrgent ? 'animate-bounce' : ''
        } hover:scale-110 active:scale-95`}>
          {isUrgent ? '🚨' : '🤖'}
        </div>
      </button>
    </div>
  )
}