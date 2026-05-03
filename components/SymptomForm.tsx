'use client'

import { useState, useEffect, useRef } from 'react'
import { supabase } from '../lib/supabase'
import { logSymptom } from '../lib/mlClient'

interface Props {
  onSuccess?: () => void
}

/* ───────────────────────────────────────────── */
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

type FeatureKey = keyof UserSymptomFeatures

type StepField = {
  key: FeatureKey
  label: string
  emoji: string
  min: number
  max: number
  isNumber?: boolean
}

type Step = {
  title: string
  subtitle: string
  icon: string
  fields: StepField[]
}

/* ───────────────────────────────────────────── */
const STEPS: Step[] = [
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

/* ───────────────────────────────────────────── */
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

/* ───────────────────────────────────────────── */
export default function SymptomForm({ onSuccess }: Props) {
  const [step, setStep] = useState(0)
  const [features, setFeatures] = useState({ ...DEFAULT_FEATURES })
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)
  const [sessionLogCount, setSessionLogCount] = useState(0)

  const submissionLock = useRef(false)

  /* ───────────────────────────────────────────── */
  useEffect(() => {
    try {
      const saved = localStorage.getItem('menoeaze_profile')
      if (saved) {
        const parsed: Partial<UserSymptomFeatures> = JSON.parse(saved)
        if (typeof parsed.age === 'number' && Number.isFinite(parsed.age)) {
          setFeatures(prev => ({ ...prev, age: parsed.age! }))
        }
        if (typeof parsed.bmi === 'number' && Number.isFinite(parsed.bmi)) {
          setFeatures(prev => ({ ...prev, bmi: parsed.bmi! }))
        }
      }
    } catch {}
  }, [])

  /* ───────────────────────────────────────────── */
  const updateFeature = (key: FeatureKey, value: number) => {
    if (!Number.isFinite(value)) return

    const field = STEPS.flatMap(s => s.fields).find(f => f.key === key)
    if (!field) return

    const safe = Math.max(field.min, Math.min(field.max, value))

    setFeatures(prev => ({ ...prev, [key]: safe }))
  }

  const validateFeatures = () =>
    Object.values(features).every(v => typeof v === 'number' && Number.isFinite(v))

  const buildVector = (): number[] => [
    features.hot_flash_score,
    features.night_sweats_score,
    features.sleep_quality,
    features.mood_score,
    features.fatigue_score,
    features.anxiety_score,
    features.physical_activity,
    features.stress_level,
    features.caffeine_intake,
    features.age,
    features.bmi
  ]

  /* ───────────────────────────────────────────── */
  const safeLog = async () => {
    let lastErr: unknown = null

    const vector = buildVector()
    if (vector.length !== 11) throw new Error('Invalid feature vector length')

    for (let i = 0; i < 2; i++) {
      try {
        const res = await logSymptom(vector, notes.slice(0, 300), '')

        if (!res || (res.status !== 'ok' && res.status !== 'degraded')) {
          throw new Error('Unexpected backend response')
        }

        return true
      } catch (err) {
        lastErr = err
        await new Promise(r => setTimeout(r, 300))
      }
    }

    throw lastErr
  }

  /* ───────────────────────────────────────────── */
  const normalizeError = (err: unknown): string => {
    if (err instanceof Error) {
      if (err.message.includes('Network')) return 'Network issue. Please try again.'
      if (err.message.includes('auth')) return 'Session expired. Please login again.'
      return err.message
    }
    return 'Something went wrong'
  }

  /* ───────────────────────────────────────────── */
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (submissionLock.current || submitting) return
    submissionLock.current = true

    setSubmitting(true)
    setErrorMsg(null)

    try {
      const { data } = await supabase.auth.getUser()

      if (!data.user?.id) {
        throw new Error('User not authenticated')
      }

      if (!validateFeatures()) {
        throw new Error('Invalid symptom values')
      }

      localStorage.setItem(
        'menoeaze_profile',
        JSON.stringify({ age: features.age, bmi: features.bmi })
      )

      await safeLog()

      setSuccess(true)
      setSessionLogCount(prev => prev + 1)

      setStep(0)
      setFeatures({ ...DEFAULT_FEATURES })
      setNotes('')

      setTimeout(() => {
        setSuccess(false)
        onSuccess?.()
      }, 1200)

    } catch (err) {
      setErrorMsg(normalizeError(err))
    } finally {
      setSubmitting(false)
      submissionLock.current = false
    }
  }

  /* ───────────────────────────────────────────── */
  const currentStep = STEPS[step]
  const isLastStep = step === STEPS.length - 1
  const progress = ((step + 1) / STEPS.length) * 100

  return (
    <form onSubmit={handleSubmit} className="space-y-5">

      {/* Feature 4: Analysis Readiness Tracker */}
      {sessionLogCount > 0 && sessionLogCount < 5 && (
        <div className="bg-indigo-50 p-3 rounded-xl border border-indigo-100">
          <div className="flex justify-between text-xs text-indigo-700 mb-1.5">
            <span className="font-medium">Analysis Readiness</span>
            <span>{sessionLogCount} of 5 logs this session</span>
          </div>
          <div className="w-full h-1.5 bg-indigo-100 rounded-full overflow-hidden">
            <div className="h-full bg-indigo-500 rounded-full transition-all duration-500"
              style={{ width: `${(sessionLogCount / 5) * 100}%` }} />
          </div>
        </div>
      )}
      {sessionLogCount >= 5 && (
        <div className="text-xs text-emerald-600 bg-emerald-50 p-2.5 rounded-xl border border-emerald-100">
          ✓ Ready for analysis — you have enough logs to generate insights.
        </div>
      )}

      {/* Progress */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Step {step + 1} of {STEPS.length}</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="w-full h-1.5 bg-gray-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-300"
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
        {currentStep.fields.map(field => {
          const value = features[field.key]

          return (
            <div key={field.key}>
              <label className="text-sm font-medium flex justify-between">
                <span>{field.emoji} {field.label}</span>
                <span className="text-purple-600 font-semibold">{value}</span>
              </label>

              {field.isNumber ? (
                <input
                  type="number"
                  value={value}
                  onChange={e => updateFeature(field.key, Number(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2 mt-1"
                />
              ) : (
                <div className="mt-1">
                  <input
                    type="range"
                    min={field.min}
                    max={field.max}
                    value={value}
                    onChange={e => updateFeature(field.key, Number(e.target.value))}
                    className="w-full accent-purple-500"
                  />
                  <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
                    <span>{field.min}</span>
                    <span>{field.max}</span>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* Notes */}
      {isLastStep && (
        <div>
          <label className="text-sm font-medium text-gray-700 block mb-1">
            📝 Additional notes (optional)
          </label>
          <textarea
            value={notes}
            onChange={e => setNotes(e.target.value.slice(0, 300))}
            placeholder="How are you feeling today? Any observations..."
            rows={3}
            className="w-full border rounded-lg px-3 py-2 text-sm resize-none"
          />
          <p className="text-[10px] text-gray-400 mt-1 text-right">{notes.length}/300</p>
        </div>
      )}

      {/* Errors */}
      {errorMsg && (
        <div className="text-red-500 text-sm bg-red-50 p-3 rounded-lg border border-red-100">
          ⚠️ {errorMsg}
        </div>
      )}
      {success && (
        <div className="text-green-600 text-sm bg-green-50 p-3 rounded-lg border border-green-100">
          ✓ Logged successfully
        </div>
      )}

      {/* Buttons */}
      <div className="flex gap-3">
        {step > 0 && (
          <button
            type="button"
            onClick={() => setStep(s => s - 1)}
            disabled={submitting}
            className="px-4 py-2 border rounded-lg text-sm font-medium text-gray-600 hover:bg-gray-50 transition disabled:opacity-50"
          >
            Back
          </button>
        )}

        {isLastStep ? (
          <button
            type="submit"
            disabled={submitting}
            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-500 text-white px-4 py-2.5 rounded-lg text-sm font-semibold disabled:opacity-50 transition hover:shadow-md"
          >
            {submitting ? 'Saving...' : 'Submit Symptoms'}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setStep(s => s + 1)}
            className="flex-1 bg-purple-600 text-white px-4 py-2.5 rounded-lg text-sm font-semibold hover:bg-purple-700 transition"
          >
            Next
          </button>
        )}
      </div>
    </form>
  )
}