'use client'

import { useEffect, useState, useRef } from 'react'
import { supabase } from '@/lib/supabase'
import { X } from 'lucide-react'

interface AuthModalProps {
  isOpen: boolean
  onClose: () => void
  initialMode?: 'login' | 'signup'
}

export default function AuthModal({
  isOpen,
  onClose,
  initialMode = 'login'
}: AuthModalProps) {
  const [mode, setMode] = useState<'login' | 'signup'>(initialMode)
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const modalRef = useRef<HTMLDivElement>(null)
  const isMountedRef = useRef(true)

  /* ---------------- Sync Mode ---------------- */
  useEffect(() => {
    setMode(initialMode)
  }, [initialMode])

  /* ---------------- Track Mount ---------------- */
  useEffect(() => {
    isMountedRef.current = true
    return () => {
      isMountedRef.current = false
    }
  }, [])

  /* ---------------- Reset Form On Close ---------------- */
  useEffect(() => {
    if (!isOpen) {
      setEmail('')
      setPassword('')
      setError(null)
      setSuccess(null)
      setLoading(false)
    }
  }, [isOpen])

  /* ---------------- ESC CLOSE ---------------- */
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    if (isOpen) window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  /* ---------------- OUTSIDE CLICK ---------------- */
  const handleOutsideClick = (e: React.MouseEvent) => {
    if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
      onClose()
    }
  }

  /* ---------------- AUTH HANDLER ---------------- */
  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault()
    if (loading) return

    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      if (mode === 'login') {
        const { error } = await supabase.auth.signInWithPassword({
          email: email.trim(),
          password
        })

        if (error) throw error

        if (isMountedRef.current) {
          onClose()
        }
      } else {
        const { error } = await supabase.auth.signUp({
          email: email.trim(),
          password,
          options: {
            emailRedirectTo: `${window.location.origin}`
          }
        })

        if (error) throw error

        if (isMountedRef.current) {
          setSuccess('Check your email to confirm your account.')
        }
      }
    } catch (err: unknown) {
      if (isMountedRef.current) {
        setError(
          err instanceof Error
            ? err.message
            : 'Authentication failed.'
        )
      }
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
      }
    }
  }

  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center px-4"
      onClick={handleOutsideClick}
      role="dialog"
      aria-modal="true"
    >
      <div
        ref={modalRef}
        className="bg-white w-full max-w-md rounded-2xl shadow-xl p-8 relative"
      >
        {/* Close */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-700 transition"
          aria-label="Close modal"
        >
          <X size={20} />
        </button>

        {/* Title */}
        <h2 className="text-2xl font-bold mb-6 text-center">
          {mode === 'login' ? 'Welcome Back' : 'Create Account'}
        </h2>

        {/* Form */}
        <form onSubmit={handleAuth} className="space-y-4">

          <input
            type="email"
            required
            autoComplete="email"
            placeholder="Email address"
            value={email}
            onChange={e => setEmail(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-purple-500 focus:outline-none"
          />

          <input
            type="password"
            required
            minLength={6}
            autoComplete={
              mode === 'login' ? 'current-password' : 'new-password'
            }
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-purple-500 focus:outline-none"
          />

          {error && (
            <p className="text-sm text-red-600">{error}</p>
          )}

          {success && (
            <p className="text-sm text-green-600">{success}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-500 text-white py-2 rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-60"
          >
            {loading
              ? 'Processing...'
              : mode === 'login'
              ? 'Sign In'
              : 'Sign Up'}
          </button>
        </form>

        {/* Mode Switch */}
        <div className="mt-6 text-center text-sm text-gray-600">
          {mode === 'login' ? (
            <>
              Don’t have an account?{' '}
              <button
                onClick={() => setMode('signup')}
                className="text-purple-600 font-medium hover:underline"
              >
                Sign Up
              </button>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <button
                onClick={() => setMode('login')}
                className="text-purple-600 font-medium hover:underline"
              >
                Sign In
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}