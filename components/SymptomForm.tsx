'use client'

import { useState, useEffect } from 'react'
import { supabase } from '../lib/supabase'
import {
  UserSymptomFeatures,
  FEATURE_KEYS,
  buildSequenceMatrix,
  featuresToRow
} from '../lib/ml/types'

interface Props {
  onSuccess?: () => void
}

/* ═══════════════════════════════════════════════
   STEP DEFINITIONS — 3 grouped sections
   ═══════════════════════════════════════════════ */
const STEPS = [
  {
    title: 'Physical Symptoms',
    subtitle: 'How is your body feeling today?',
    icon: '🌡️',
    fields: [
      { key: 'hot_flash_score' as const, label: 'Hot Flashes', emoji: '🔥', min: 0, max: 10 },
      { key: 'night_sweats_score' as const, label: 'Night Sweats', emoji: '💧', min: 0, max: 10 },
      { key: 'fatigue_score' as const, label: 'Fatigue', emoji: '😴', min: 0, max: 10 },
      { key: 'physical_activity' as const, label: 'Physical Activity', emoji: '🏃‍♀️', min: 0, max: 10 },
    ]
  },
  {
    title: 'Emotional Wellness',
    subtitle: 'How are you feeling inside?',
    icon: '💜',
    fields: [
      { key: 'mood_score' as const, label: 'Mood', emoji: '😊', min: 0, max: 10 },
      { key: 'anxiety_score' as const, label: 'Anxiety', emoji: '😰', min: 0, max: 10 },
      { key: 'stress_level' as const, label: 'Stress Level', emoji: '🧠', min: 0, max: 10 },
    ]
  },
  {
    title: 'Lifestyle & Sleep',
    subtitle: 'Your daily patterns matter',
    icon: '🌙',
    fields: [
      { key: 'sleep_quality' as const, label: 'Sleep Quality', emoji: '😴', min: 0, max: 10 },
      { key: 'caffeine_intake' as const, label: 'Caffeine (cups)', emoji: '☕', min: 0, max: 5 },
      { key: 'age' as const, label: 'Age', emoji: '🎂', min: 30, max: 70, isNumber: true },
      { key: 'bmi' as const, label: 'BMI', emoji: '⚖️', min: 15, max: 50, isNumber: true },
    ]
  }
]

/* ═══════════════════════════════════════════════
   DEFAULT FEATURE VALUES
   ═══════════════════════════════════════════════ */
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

  // Load saved age/bmi from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('menoeaze_profile')
      if (saved) {
        const { age, bmi } = JSON.parse(saved)
        if (age) setFeatures(prev => ({ ...prev, age }))
        if (bmi) setFeatures(prev => ({ ...prev, bmi }))
      }
    } catch { /* ignore */ }
  }, [])

  const updateFeature = (key: keyof UserSymptomFeatures, value: number) => {
    setFeatures(prev => ({ ...prev, [key]: value }))
  }

  const currentStep = STEPS[step]
  const isLastStep = step === STEPS.length - 1
  const progress = ((step + 1) / STEPS.length) * 100

  const handleNext = () => {
    if (!isLastStep) {
      setStep(prev => prev + 1)
    }
  }

  const handleBack = () => {
    if (step > 0) {
      setStep(prev => prev - 1)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (submitting) return

    setSubmitting(true)
    setErrorMsg(null)
    setSuccess(false)

    try {
      const { data: authData } = await supabase.auth.getUser()
      const userId = authData.user?.id

      if (!userId) throw new Error('User not authenticated.')

      // Save age/bmi to localStorage for future sessions
      localStorage.setItem('menoeaze_profile', JSON.stringify({
        age: features.age,
        bmi: features.bmi
      }))

      // 1. Insert all 11 features into Supabase
      const { error } = await supabase.from('symptom_logs').insert([{
        user_id: userId,
        hot_flash_score: features.hot_flash_score,
        night_sweats_score: features.night_sweats_score,
        sleep_quality: features.sleep_quality,
        mood_score: features.mood_score,
        fatigue_score: features.fatigue_score,
        anxiety_score: features.anxiety_score,
        physical_activity: features.physical_activity,
        stress_level: features.stress_level,
        caffeine_intake: features.caffeine_intake,
        age: features.age,
        bmi: features.bmi,
        severity: computeOverallSeverity(features),
        notes: notes.trim(),
      }])

      if (error) {
        if (error.message.includes('idx_unique_daily_log')) {
          throw new Error('You\u2019ve already logged today\u2019s symptoms.')
        }
        throw error
      }

      // 2. Fetch past logs for forward-fill matrix
      const { data: pastLogs } = await supabase
        .from('symptom_logs')
        .select(FEATURE_KEYS.join(', '))
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(5)

      // Build (5, 11) matrix with forward-fill
      const logEntries: UserSymptomFeatures[] = pastLogs && pastLogs.length > 0
        ? pastLogs.reverse().map((row: any) => ({
          hot_flash_score: row.hot_flash_score ?? features.hot_flash_score,
          night_sweats_score: row.night_sweats_score ?? features.night_sweats_score,
          sleep_quality: row.sleep_quality ?? features.sleep_quality,
          mood_score: row.mood_score ?? features.mood_score,
          fatigue_score: row.fatigue_score ?? features.fatigue_score,
          anxiety_score: row.anxiety_score ?? features.anxiety_score,
          physical_activity: row.physical_activity ?? features.physical_activity,
          stress_level: row.stress_level ?? features.stress_level,
          caffeine_intake: row.caffeine_intake ?? features.caffeine_intake,
          age: row.age ?? features.age,
          bmi: row.bmi ?? features.bmi,
        }))
        : [features]

      const sequence = buildSequenceMatrix(logEntries)

      // 3. POST to ML /feedback for continual learning
      const mlApiUrl = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000'
      fetch(`${mlApiUrl}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          actual_severity: computeOverallSeverity(features) / 10,
          sequence,
        }),
      }).catch(() => { /* non-blocking — ML may not be running */ })

      setSuccess(true)
      setStep(0)
      setFeatures({ ...DEFAULT_FEATURES })
      setNotes('')

      setTimeout(() => {
        setSuccess(false)
        onSuccess?.()
      }, 1500)

    } catch (err: any) {
      setErrorMsg(err.message || 'Something went wrong.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-5">

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>Step {step + 1} of {STEPS.length}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Step Header */}
      <div className="flex items-center gap-3">
        <span className="text-2xl">{currentStep.icon}</span>
        <div>
          <h3 className="text-base font-semibold text-gray-800">{currentStep.title}</h3>
          <p className="text-xs text-gray-500">{currentStep.subtitle}</p>
        </div>
      </div>

      {/* Feature Inputs */}
      <div className="space-y-4">
        {currentStep.fields.map((field) => (
          <div key={field.key}>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-sm font-medium text-gray-700 flex items-center gap-1.5">
                <span>{field.emoji}</span>
                {field.label}
              </label>
              <span className="text-sm font-semibold text-purple-600">
                {features[field.key]}
                {!('isNumber' in field) && <span className="text-gray-400 font-normal">/{field.max}</span>}
              </span>
            </div>

            {'isNumber' in field && field.isNumber ? (
              <input
                type="number"
                value={features[field.key]}
                onChange={(e) => {
                  const val = Number(e.target.value)
                  if (val >= field.min && val <= field.max) {
                    updateFeature(field.key, val)
                  }
                }}
                min={field.min}
                max={field.max}
                className="w-full border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition text-sm min-h-[44px]"
              />
            ) : (
              <input
                type="range"
                min={field.min}
                max={field.max}
                value={features[field.key]}
                onChange={(e) => updateFeature(field.key, Number(e.target.value))}
                className="w-full accent-purple-600 min-h-[44px]"
              />
            )}
          </div>
        ))}
      </div>

      {/* Notes (last step only) */}
      {isLastStep && (
        <div>
          <label className="text-sm font-medium text-gray-700 block mb-1.5">
            📝 Notes (optional)
          </label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={2}
            placeholder="Anything else you want to note?"
            className="w-full border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 resize-none outline-none transition text-sm"
          />
        </div>
      )}

      {/* Feedback Messages */}
      {errorMsg && (
        <div className="text-red-500 text-sm font-medium bg-red-50 px-4 py-2.5 rounded-xl">
          {errorMsg}
        </div>
      )}

      {success && (
        <div className="text-green-600 text-sm font-medium bg-green-50 px-4 py-2.5 rounded-xl flex items-center gap-2">
          <span>✓</span> Logged successfully — your AI is learning from this!
        </div>
      )}

      {/* Navigation Buttons */}
      <div className="flex gap-3">
        {step > 0 && (
          <button
            type="button"
            onClick={handleBack}
            className="flex-1 border border-gray-300 text-gray-700 py-3 rounded-xl font-medium text-sm hover:bg-gray-50 transition min-h-[44px]"
          >
            ← Back
          </button>
        )}

        {isLastStep ? (
          <button
            type="submit"
            disabled={submitting}
            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-500 text-white py-3 rounded-xl font-semibold shadow-sm hover:shadow-md transition disabled:opacity-50 text-sm min-h-[44px]"
          >
            {submitting ? 'Logging...' : "Log Today's Symptoms 💜"}
          </button>
        ) : (
          <button
            type="button"
            onClick={handleNext}
            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-500 text-white py-3 rounded-xl font-semibold shadow-sm hover:shadow-md transition text-sm min-h-[44px]"
          >
            Next →
          </button>
        )}
      </div>
    </form>
  )
}

/* ═══════════════════════════════════════════════
   UTILITY: Overall severity composite
   ═══════════════════════════════════════════════ */
function computeOverallSeverity(f: UserSymptomFeatures): number {
  // Weighted average of negative symptoms (higher = worse)
  const raw = (
    f.hot_flash_score * 0.18 +
    f.night_sweats_score * 0.12 +
    (10 - f.sleep_quality) * 0.15 +
    (10 - f.mood_score) * 0.12 +
    f.fatigue_score * 0.13 +
    f.anxiety_score * 0.12 +
    f.stress_level * 0.10 +
    (10 - f.physical_activity) * 0.08
  )
  return Math.round(Math.max(0, Math.min(10, raw)))
}