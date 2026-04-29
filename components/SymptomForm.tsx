'use client'

import { useState, useEffect, useRef } from 'react'
import { supabase } from '@/lib/supabase'

interface Props {
  onSuccess?: () => void
}

/* ================= MENOPAUSE SYMPTOM SUGGESTIONS ================= */
const symptomSuggestions = [
  'Hot flashes', 'Night sweats', 'Irregular periods', 'Vaginal dryness',
  'Sleep problems', 'Fatigue', 'Brain fog', 'Mood swings',
  'Joint pain', 'Headaches', 'Heart palpitations', 'Weight gain',
  'Hair thinning', 'Dry skin', 'Itchiness', 'Bloating',
  'Breast tenderness', 'Muscle tension', 'Dizziness', 'Urinary issues',
  'Low libido', 'Tingling extremities', 'Anxiety', 'Depression',
  'Digestive issues', 'Bone pain', 'Memory lapses', 'Concentration issues'
]

/* ================= MOOD SUGGESTIONS ================= */
const moodSuggestions = [
  'Anxious', 'Irritable', 'Sad', 'Overwhelmed', 'Frustrated',
  'Hopeful', 'Calm', 'Grateful', 'Happy', 'Empowered',
  'Lonely', 'Tearful', 'Restless', 'Numb', 'Confident',
  'Exhausted', 'Peaceful', 'Worried', 'Content', 'Moody'
]

const quickEmojis = ['😊', '😢', '😡', '😴', '😰', '😌', '😭', '😤', '🤒', '💖']

export default function SymptomForm({ onSuccess }: Props) {
  const [symptomType, setSymptomType] = useState('')
  const [severity, setSeverity] = useState(5)
  const [mood, setMood] = useState('')
  const [sleep, setSleep] = useState(5)
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [showSymptomSuggestions, setShowSymptomSuggestions] = useState(false)
  const [showMoodSuggestions, setShowMoodSuggestions] = useState(false)

  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const resetForm = () => {
    setSymptomType('')
    setSeverity(5)
    setMood('')
    setSleep(5)
    setNotes('')
    setShowSymptomSuggestions(false)
    setShowMoodSuggestions(false)
  }

  const addEmoji = (emoji: string) => {
    setMood(prev => prev + emoji)
  }

  const selectSymptom = (symptom: string) => {
    setSymptomType(symptom)
    setShowSymptomSuggestions(false)
  }

  const selectMood = (m: string) => {
    setMood(m)
    setShowMoodSuggestions(false)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (submitting) return

    if (!symptomType.trim()) {
      setErrorMsg('Symptom type is required.')
      return
    }

    if (!mood.trim()) {
      setErrorMsg('Mood is required.')
      return
    }

    setSubmitting(true)
    setErrorMsg(null)
    setSuccess(false)

    try {
      const { data: authData } = await supabase.auth.getUser()
      const userId = authData.user?.id

      if (!userId) {
        throw new Error('User not authenticated.')
      }

      const { error } = await supabase.from('symptom_logs').insert([
        {
          user_id: userId,
          symptom_type: symptomType.trim(),
          severity,
          mood: mood.trim(),
          sleep,
          notes: notes.trim()
        }
      ])

      if (error) {
        if (error.message.includes('idx_unique_daily_log')) {
          throw new Error('You have already logged today\u2019s symptom.')
        }
        throw error
      }

      setSuccess(true)
      resetForm()

      setTimeout(() => {
        setSuccess(false)
        onSuccess?.()
      }, 1200)

    } catch (err: any) {
      setErrorMsg(err.message || 'Something went wrong.')
    } finally {
      setSubmitting(false)
    }
  }

  // Filter suggestions as user types
  const filteredSymptoms = symptomSuggestions.filter(s =>
    s.toLowerCase().includes(symptomType.toLowerCase())
  )
  const filteredMoods = moodSuggestions.filter(m =>
    m.toLowerCase().includes(mood.toLowerCase())
  )

  return (
    <form
      onSubmit={handleSubmit}
      className="space-y-6"
    >
      <h2 className="text-xl font-semibold text-gray-900">
        Log Today&apos;s Symptoms
      </h2>

      {/* Symptom Type with Suggestions */}
      <div className="relative">
        <input
          ref={inputRef}
          value={symptomType}
          onChange={(e) => {
            setSymptomType(e.target.value)
            setShowSymptomSuggestions(true)
          }}
          onFocus={() => setShowSymptomSuggestions(true)}
          placeholder="Type or select a symptom..."
          className="w-full border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition"
        />

        {/* Suggestion Chips */}
        {showSymptomSuggestions && (
          <div className="mt-2 max-h-40 overflow-y-auto">
            <div className="flex flex-wrap gap-1.5">
              {(symptomType.trim() ? filteredSymptoms : symptomSuggestions).map((symptom) => (
                <button
                  type="button"
                  key={symptom}
                  onClick={() => selectSymptom(symptom)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${symptomType === symptom
                      ? 'bg-purple-600 text-white shadow-sm'
                      : 'bg-purple-50 text-purple-700 hover:bg-purple-100 border border-purple-100'
                    }`}
                >
                  {symptom}
                </button>
              ))}
            </div>
            {symptomType.trim() && filteredSymptoms.length === 0 && (
              <p className="text-xs text-gray-400 mt-2 ml-1">
                Custom symptom: &ldquo;{symptomType}&rdquo; — that&apos;s fine, log it!
              </p>
            )}
          </div>
        )}
      </div>

      {/* Severity */}
      <div>
        <label className="text-sm font-medium text-gray-700 block mb-2">
          Severity: <span className="text-purple-600 font-semibold">{severity}/10</span>
        </label>
        <input
          type="range"
          min="1"
          max="10"
          value={severity}
          onChange={(e) => setSeverity(Number(e.target.value))}
          className="w-full accent-purple-600"
        />
      </div>

      {/* Mood with Suggestions */}
      <div className="relative">
        <label className="text-sm font-medium text-gray-700 block mb-2">
          Mood
        </label>

        <input
          value={mood}
          onChange={(e) => {
            setMood(e.target.value)
            setShowMoodSuggestions(true)
          }}
          onFocus={() => setShowMoodSuggestions(true)}
          placeholder="Type or select your mood..."
          className="w-full border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition"
        />

        {/* Mood Suggestion Chips */}
        {showMoodSuggestions && (
          <div className="mt-2 max-h-32 overflow-y-auto">
            <div className="flex flex-wrap gap-1.5">
              {(mood.trim() ? filteredMoods : moodSuggestions).map((m) => (
                <button
                  type="button"
                  key={m}
                  onClick={() => selectMood(m)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${mood === m
                      ? 'bg-pink-600 text-white shadow-sm'
                      : 'bg-pink-50 text-pink-700 hover:bg-pink-100 border border-pink-100'
                    }`}
                >
                  {m}
                </button>
              ))}
            </div>
            {mood.trim() && filteredMoods.length === 0 && (
              <p className="text-xs text-gray-400 mt-2 ml-1">
                Custom mood: &ldquo;{mood}&rdquo; — every feeling is valid 💜
              </p>
            )}
          </div>
        )}

        {/* Quick Emojis */}
        <div className="flex flex-wrap gap-2 mt-3">
          {quickEmojis.map((emoji) => (
            <button
              type="button"
              key={emoji}
              onClick={() => addEmoji(emoji)}
              className="text-xl hover:scale-110 active:scale-95 transition"
            >
              {emoji}
            </button>
          ))}
        </div>
      </div>

      {/* Sleep */}
      <div>
        <label className="text-sm font-medium text-gray-700 block mb-2">
          Sleep Quality: <span className="text-purple-600 font-semibold">{sleep}/10</span>
        </label>
        <input
          type="range"
          min="1"
          max="10"
          value={sleep}
          onChange={(e) => setSleep(Number(e.target.value))}
          className="w-full accent-purple-600"
        />
      </div>

      {/* Notes */}
      <div>
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          rows={3}
          placeholder="Anything else you want to note?"
          className="w-full border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 resize-none outline-none transition"
        />
      </div>

      {/* Feedback */}
      {errorMsg && (
        <div className="text-red-500 text-sm font-medium">
          {errorMsg}
        </div>
      )}

      {success && (
        <div className="text-green-600 text-sm font-medium">
          ✓ Logged successfully
        </div>
      )}

      {/* Submit */}
      <button
        type="submit"
        disabled={submitting}
        className="w-full bg-gradient-to-r from-purple-600 to-pink-500 text-white py-3 rounded-xl font-semibold shadow-sm hover:shadow-md transition disabled:opacity-50"
      >
        {submitting ? 'Logging...' : "Log Today's Symptom"}
      </button>
    </form>
  )
}