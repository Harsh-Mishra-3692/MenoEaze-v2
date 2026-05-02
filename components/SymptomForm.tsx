'use client'

import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'

interface Props {
  onSuccess?: () => void
}

/* ─────────────────────────────────────────────
   FEATURE TYPE (LOCKED — DO NOT CHANGE)
───────────────────────────────────────────── */
type UserSymptomFeatures = {
  hot_flash_score: number
  night_sweats_score: number
  sleep_quality: number
  mood_score: number
  fatigue_score: number
  anxiety_score: number
  physical_activity: number
  stress_level: number
  caffeine_intake: number
  age: number
  bmi: number
}

/* ─────────────────────────────────────────────
   STEPS (UNCHANGED UX)
───────────────────────────────────────────── */
const STEPS = [
  {
    title: 'Physical Symptoms',
    subtitle: 'How is your body feeling today?',
    icon: '🌡️',
    fields: [
      { key: 'hot_flash_score', label: 'Hot Flashes', emoji: '🔥', min: 0, max: 10 },
      { key: 'night_sweats_score', label: 'Night Sweats', emoji: '💧', min: 0, max: 10 },
      { key: 'fatigue_score', label: 'Fatigue', emoji: '😴', min: 0, max: 10 },
      { key: 'physical_activity', label: 'Physical Activity', emoji: '🏃‍♀️', min: 0, max: 10 },
    ]
  },
  {
    title: 'Emotional Wellness',
    subtitle: 'How are you feeling inside?',
    icon: '💜',
    fields: [
      { key: 'mood_score', label: 'Mood', emoji: '😊', min: 0, max: 10 },
      { key: 'anxiety_score', label: 'Anxiety', emoji: '😰', min: 0, max: 10 },
      { key: 'stress_level', label: 'Stress Level', emoji: '🧠', min: 0, max: 10 },
    ]
  },
  {
    title: 'Lifestyle & Sleep',
    subtitle: 'Your daily patterns matter',
    icon: '🌙',
    fields: [
      { key: 'sleep_quality', label: 'Sleep Quality', emoji: '😴', min: 0, max: 10 },
      { key: 'caffeine_intake', label: 'Caffeine (cups)', emoji: '☕', min: 0, max: 5 },
      { key: 'age', label: 'Age', emoji: '🎂', min: 30, max: 70, isNumber: true },
      { key: 'bmi', label: 'BMI', emoji: '⚖️', min: 15, max: 50, isNumber: true },
    ]
  }
]

/* ─────────────────────────────────────────────
   DEFAULT VALUES
───────────────────────────────────────────── */
const DEFAULT_FEATURES: UserSymptomFeatures = {
  hot_flash_score: 3,
  night_sweats_score: 2,
  sleep_quality: 6,
  mood_score: 6,
  fatigue_score: 3,
  anxiety_score: 3,
  physical_activity: 5,
  stress_level: 4,
  caffeine_intake: 2,
  age: 50,
  bmi: 25,
}

export default function SymptomForm({ onSuccess }: Props) {
  const [step, setStep] = useState(0)
  const [features, setFeatures] = useState<UserSymptomFeatures>({ ...DEFAULT_FEATURES })
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  useEffect(() => {
    try {
      const saved = localStorage.getItem('menoeaze_profile')
      if (saved) {
        const { age, bmi } = JSON.parse(saved)
        if (age) setFeatures(prev => ({ ...prev, age }))
        if (bmi) setFeatures(prev => ({ ...prev, bmi }))
      }
    } catch {}
  }, [])

  const updateFeature = (key: keyof UserSymptomFeatures, value: number) => {
    setFeatures(prev => ({ ...prev, [key]: value }))
  }

  const currentStep = STEPS[step]
  const isLastStep = step === STEPS.length - 1
  const progress = ((step + 1) / STEPS.length) * 100

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (submitting) return

    setSubmitting(true)
    setErrorMsg(null)

    try {
      const { data: authData } = await supabase.auth.getUser()
      const userId = authData.user?.id

      if (!userId) throw new Error('User not authenticated')

      // persist profile
      localStorage.setItem('menoeaze_profile', JSON.stringify({
        age: features.age,
        bmi: features.bmi
      }))

      const mlApiUrl = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000'

      const res = await fetch(`${mlApiUrl}/log-symptom`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          features,
          notes,
          timestamp: Date.now()
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err?.detail || 'Failed to log symptoms')
      }

      setSuccess(true)
      setStep(0)
      setFeatures({ ...DEFAULT_FEATURES })
      setNotes('')

      setTimeout(() => {
        setSuccess(false)
        onSuccess?.()
      }, 1500)

    } catch (err: any) {
      setErrorMsg(err.message || 'Something went wrong')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">

      {/* Progress */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Step {step + 1} of {STEPS.length}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Header */}
      <div className="flex items-center gap-3">
        <span className="text-2xl">{currentStep.icon}</span>
        <div>
          <h3 className="font-semibold">{currentStep.title}</h3>
          <p className="text-xs text-gray-500">{currentStep.subtitle}</p>
        </div>
      </div>

      {/* Inputs */}
      <div className="space-y-4">
        {currentStep.fields.map((field: any) => (
          <div key={field.key}>
            <label className="text-sm font-medium flex justify-between">
              <span>{field.emoji} {field.label}</span>
              <span>{features[field.key as keyof UserSymptomFeatures]}</span>
            </label>

            {field.isNumber ? (
              <input
                type="number"
                value={features[field.key as keyof UserSymptomFeatures]}
                onChange={(e) => {
                  const val = Number(e.target.value)
                  if (val >= field.min && val <= field.max) {
                    updateFeature(field.key, val)
                  }
                }}
                className="w-full border rounded-lg px-3 py-2"
              />
            ) : (
              <input
                type="range"
                min={field.min}
                max={field.max}
                value={features[field.key as keyof UserSymptomFeatures]}
                onChange={(e) => updateFeature(field.key, Number(e.target.value))}
                className="w-full"
              />
            )}
          </div>
        ))}
      </div>

      {/* Notes */}
      {isLastStep && (
        <textarea
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          placeholder="Notes..."
          className="w-full border rounded-lg px-3 py-2"
        />
      )}

      {/* Messages */}
      {errorMsg && <div className="text-red-500">{errorMsg}</div>}
      {success && <div className="text-green-600">✓ Logged successfully</div>}

      {/* Buttons */}
      <div className="flex gap-3">
        {step > 0 && (
          <button type="button" onClick={() => setStep(s => s - 1)}>
            Back
          </button>
        )}

        {isLastStep ? (
          <button type="submit" disabled={submitting}>
            {submitting ? 'Saving...' : 'Submit'}
          </button>
        ) : (
          <button type="button" onClick={() => setStep(s => s + 1)}>
            Next
          </button>
        )}
      </div>
    </form>
  )
}