'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'

export default function AuthPage() {
  const router = useRouter()

  const [mode, setMode] = useState<'login' | 'signup'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [username, setUsername] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [message, setMessage] = useState<string | null>(null)

  /* ---------------- Redirect If Logged In ---------------- */
  useEffect(() => {
    const checkSession = async () => {
      const { data } = await supabase.auth.getSession()
      if (data.session) {
        router.replace('/dashboard')
      }
    }
    checkSession()
  }, [router])

  /* ---------------- Submit Handler ---------------- */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (loading) return

    setLoading(true)
    setError(null)
    setMessage(null)

    try {
      if (mode === 'signup') {
        if (password.length < 8) {
          throw new Error('Password must be at least 8 characters.')
        }

        if (!username.trim()) {
          throw new Error('Please choose a display name.')
        }

        const { data, error } = await supabase.auth.signUp({
          email: email.trim(),
          password,
          options: {
            emailRedirectTo: window.location.origin
          }
        })

        if (error) throw error

        // Create profile with username
        if (data.user) {
          await supabase.from('profiles').upsert({
            id: data.user.id,
            username: username.trim()
          })
        }

        if (data.user && !data.session) {
          setMessage('Check your email to confirm your account.')
        } else {
          router.push('/dashboard')
        }
      } else {
        const { error } = await supabase.auth.signInWithPassword({
          email: email.trim(),
          password
        })

        if (error) throw error

        router.push('/dashboard')
      }
    } catch (err: unknown) {
      setError(
        err instanceof Error ? err.message : 'Authentication failed.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">

      <div className="w-full max-w-md bg-white rounded-2xl shadow-lg p-8">

        {/* Brand */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-500 bg-clip-text text-transparent">
            MenoEaze
          </h1>
          <p className="text-sm text-gray-500 mt-2">
            AI-powered menopause health companion
          </p>
        </div>

        {/* Tabs */}
        <div className="flex mb-6 border-b">
          <button
            onClick={() => setMode('login')}
            className={`flex-1 py-2 text-sm font-medium ${mode === 'login'
                ? 'border-b-2 border-purple-600 text-purple-600'
                : 'text-gray-500'
              }`}
          >
            Login
          </button>
          <button
            onClick={() => setMode('signup')}
            className={`flex-1 py-2 text-sm font-medium ${mode === 'signup'
                ? 'border-b-2 border-purple-600 text-purple-600'
                : 'text-gray-500'
              }`}
          >
            Sign Up
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">

          {/* Username — Signup only */}
          {mode === 'signup' && (
            <input
              type="text"
              required
              placeholder="Choose a display name"
              value={username}
              onChange={e => setUsername(e.target.value)}
              className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-purple-500 focus:outline-none"
            />
          )}

          <input
            type="email"
            required
            autoComplete="email"
            placeholder="Email"
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

          {message && (
            <p className="text-sm text-green-600">{message}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-purple-600 to-pink-500 text-white py-2 rounded-lg font-semibold hover:shadow-lg transition disabled:opacity-60"
          >
            {loading
              ? 'Processing...'
              : mode === 'login'
                ? 'Login'
                : 'Create Account'}
          </button>
        </form>

      </div>
    </div>
  )
}