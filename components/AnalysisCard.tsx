'use client'

import React, { useState, useEffect, useRef } from 'react'
import { runMLPipeline, submitFeedback, MLRunResponse, MLClientError } from '@/lib/mlClient'

interface Props {
  userId: string
  onDoctorUrgency?: (recommend: boolean, urgency: string) => void
}

export default function AnalysisCard({ userId, onDoctorUrgency }: Props) {
  const [showReasoning, setShowReasoning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [mlData, setMlData] = useState<MLRunResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Feature 9: Feedback Loop
  const [showFeedback, setShowFeedback] = useState(false)
  const [feedbackActual, setFeedbackActual] = useState(0.5)
  const [feedbackRating, setFeedbackRating] = useState(5)
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false)
  const [feedbackDone, setFeedbackDone] = useState(false)

  // Feature 10: Trust History (session-scoped)
  const [trustHistory, setTrustHistory] = useState<number[]>([])

  const isMounted = useRef(true)
  const loadingRef = useRef(false)

  useEffect(() => {
    isMounted.current = true
    return () => { isMounted.current = false }
  }, [])

  const handleRunAnalysis = async () => {
    if (loadingRef.current) return
    loadingRef.current = true

    try {
      if (isMounted.current) { setLoading(true); setError(null); setShowFeedback(false); setFeedbackDone(false) }

      const res = await runMLPipeline('Analyze my current symptom patterns and health status')

      if (isMounted.current) {
        setMlData(res)

        // Feature 8: Persist doctor urgency for FloatingAssistant (layout-level)
        const dr = res.doctor?.recommend || false
        const du = res.doctor?.urgency || 'none'
        try {
          localStorage.setItem('menoeaze_doctor_urgency', JSON.stringify({ recommend: dr, urgency: du }))
        } catch {}
        onDoctorUrgency?.(dr, du)

        // Feature 9: Show feedback if prediction_id exists (not guardrail override)
        const isOverride = res.reasoning?.override === true
        if (!isOverride && res.prediction?.prediction_id) {
          setShowFeedback(true)
          setFeedbackActual(res.prediction.severity || 0.5)
        }
      }
    } catch (err: unknown) {
      if (!isMounted.current) return
      if (err instanceof MLClientError) setError(err.message)
      else if (err instanceof Error) setError(err.message)
      else setError('Analysis failed')
      setMlData(null)
    } finally {
      loadingRef.current = false
      if (isMounted.current) setLoading(false)
    }
  }

  // Feature 9: Submit feedback
  const handleFeedback = async () => {
    if (!mlData?.prediction?.prediction_id || feedbackSubmitting) return
    setFeedbackSubmitting(true)

    try {
      const res = await submitFeedback(
        mlData.prediction.prediction_id,
        mlData.prediction.severity || 0.5,
        feedbackActual,
        feedbackRating
      )

      // Feature 10: Append trust score to history
      if (typeof res.trust_score === 'number') {
        setTrustHistory(prev => [...prev, res.trust_score!])
      }

      setFeedbackDone(true)
      setShowFeedback(false)
    } catch {
      // Feedback failure is non-critical
    } finally {
      setFeedbackSubmitting(false)
    }
  }

  // Derived values — pure reads, no computation
  const severity = safe(mlData?.prediction?.severity, 0.5)
  const confidence = safe(mlData?.prediction?.confidence, 0.0)
  const answer = mlData?.rag?.answer || ''
  const anomaly = mlData?.reasoning?.anomaly || false
  const trend = mlData?.reasoning?.trend || ''
  const isOverride = mlData?.reasoning?.override === true
  const personalized = mlData?.prediction?.personalized || false
  const trust = mlData?.reasoning?.factors?.trust
  const bandLower = mlData?.reasoning?.confidence_band?.lower
  const bandUpper = mlData?.reasoning?.confidence_band?.upper
  const doctorRecommend = mlData?.doctor?.recommend || false
  const doctorUrgency = mlData?.doctor?.urgency || 'none'
  const isDegraded = mlData?.status === 'degraded'
  const degradedReasons = mlData?.reasons || []

  return (
    <div className="space-y-4">

      {/* Header + Button */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
        <h3 className="text-lg font-semibold text-gray-800">Health Analysis</h3>
        <button onClick={handleRunAnalysis} disabled={loading}
          className="w-full sm:w-auto bg-gradient-to-r from-purple-600 to-pink-500 hover:from-purple-700 hover:to-pink-600 text-white px-5 py-2.5 min-h-[44px] rounded-xl text-sm font-semibold disabled:opacity-50 transition-all shadow-sm hover:shadow-md">
          {loading ? 'Analyzing…' : '✨ Run Analysis'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-50 text-red-600 p-4 rounded-xl text-sm border border-red-100">
          <p className="font-medium">Analysis Failed</p>
          <p className="mt-1">{error}</p>
        </div>
      )}

      {/* Empty State */}
      {!mlData && !loading && !error && (
        <div className="text-center text-gray-400 py-10 bg-gray-50/50 rounded-xl border border-dashed border-gray-200">
          <p className="text-2xl mb-2">🔬</p>
          <p className="text-sm">Click &quot;Run Analysis&quot; to generate insights</p>
        </div>
      )}

      {/* ═══════════ RESULTS ═══════════ */}
      {mlData && (
        <div className="space-y-4">

          {/* Feature 1: Global Anomaly Alert */}
          {anomaly && (
            <div className="flex items-center gap-2 bg-orange-50 text-orange-700 p-3 rounded-xl border border-orange-200 text-sm">
              <span className="text-lg">⚠️</span>
              <span><strong>Anomaly detected</strong> — Your current symptoms deviate significantly from your historical baseline.</span>
            </div>
          )}

          {/* Feature 12: Guardrail Override Notification */}
          {isOverride && (
            <div className="flex items-center gap-2 bg-rose-50 text-rose-700 p-3 rounded-xl border border-rose-200 text-sm">
              <span className="text-lg">🛡️</span>
              <span><strong>Clinical safety protocol active</strong> — This response was generated by the safety guardrail system.</span>
            </div>
          )}

          {/* Feature 3: Severity Bar with Confidence Bands */}
          <SeverityBar severity={severity} bandLower={bandLower} bandUpper={bandUpper} />

          {/* Feature 2 + 5 + 6: Trend + Personalization + Calibration Row */}
          <div className="flex flex-wrap gap-2">
            {/* Feature 2: Trend */}
            <TrendBadge trend={trend} />

            {/* Feature 5: Personalization State */}
            <span className={`inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full border ${
              personalized
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : 'bg-gray-50 text-gray-500 border-gray-200'
            }`}>
              {personalized ? '🎯 Personalized' : '📊 General Baseline'}
            </span>

            {/* Feature 6: System Calibration */}
            {typeof trust === 'number' && (
              <span className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200">
                ⚙️ Calibration: {(trust * 100).toFixed(0)}%
              </span>
            )}

            {/* Confidence */}
            <span className="inline-flex items-center text-xs px-2.5 py-1 rounded-full bg-gray-50 text-gray-600 border border-gray-200">
              Confidence: {(confidence * 100).toFixed(0)}%
            </span>
          </div>

          {/* RAG Answer */}
          {answer && (
            <div className="text-sm bg-purple-50/50 p-4 rounded-xl border border-purple-100 leading-relaxed text-gray-700">
              {answer}
            </div>
          )}

          {/* Feature 11: Urgency Escalation */}
          {doctorRecommend && (
            <div className={`text-sm p-4 rounded-xl border ${
              doctorUrgency === 'immediate'
                ? 'bg-red-50 text-red-800 border-red-300'
                : 'bg-amber-50 text-amber-800 border-amber-200'
            }`}>
              <div className="flex items-center gap-2 font-medium mb-1">
                <span>{doctorUrgency === 'immediate' ? '🚨' : '🩺'}</span>
                <span>Doctor consultation recommended</span>
              </div>
              <p>Urgency: <strong>{doctorUrgency}</strong>
                {doctorUrgency === 'immediate' && ' — Please seek medical attention promptly.'}
              </p>
            </div>
          )}

          {/* Degradation Transparency (Feature 7 — dashboard version) */}
          {isDegraded && degradedReasons.length > 0 && (
            <div className="text-xs bg-yellow-50 text-yellow-700 p-3 rounded-xl border border-yellow-200">
              <span className="font-medium">⚡ System note:</span> Some subsystems operated in fallback mode ({degradedReasons.join(', ')}).
            </div>
          )}

          {/* Reasoning Toggle */}
          <button onClick={() => setShowReasoning(!showReasoning)}
            className="text-xs text-purple-600 hover:text-purple-700 font-medium transition">
            {showReasoning ? '▾ Hide reasoning details' : '▸ Show reasoning details'}
          </button>

          {showReasoning && (
            <div className="text-xs bg-gray-50 p-3 rounded-xl border border-gray-100 text-gray-600 space-y-1">
              <p>Confidence: {(confidence * 100).toFixed(0)}%</p>
              {trend && <p>Trend: {trend}</p>}
              {anomaly && <p>⚠️ Anomaly flag active</p>}
              {typeof trust === 'number' && <p>Trust: {(trust * 100).toFixed(0)}%</p>}
              {mlData.reasoning?.factors?.rag_quality !== undefined && (
                <p>RAG Quality: {(mlData.reasoning.factors.rag_quality * 100).toFixed(0)}%</p>
              )}
              {bandLower !== undefined && bandUpper !== undefined && (
                <p>Band: [{(bandLower * 100).toFixed(0)}% – {(bandUpper * 100).toFixed(0)}%]</p>
              )}
            </div>
          )}

          {/* Feature 9: Feedback Prompt */}
          {showFeedback && !feedbackDone && (
            <div className="bg-indigo-50/50 p-4 rounded-xl border border-indigo-100 space-y-3">
              <p className="text-sm font-medium text-gray-700">How accurate was this analysis?</p>

              <div>
                <label className="text-xs text-gray-500 flex justify-between mb-1">
                  <span>Your perceived severity</span>
                  <span className="font-medium text-indigo-600">{(feedbackActual * 100).toFixed(0)}%</span>
                </label>
                <input type="range" min={0} max={100} value={feedbackActual * 100}
                  onChange={e => setFeedbackActual(Number(e.target.value) / 100)}
                  className="w-full accent-indigo-500" />
              </div>

              <div>
                <label className="text-xs text-gray-500 flex justify-between mb-1">
                  <span>Overall satisfaction</span>
                  <span className="font-medium text-indigo-600">{feedbackRating}/10</span>
                </label>
                <input type="range" min={1} max={10} value={feedbackRating}
                  onChange={e => setFeedbackRating(Number(e.target.value))}
                  className="w-full accent-indigo-500" />
              </div>

              <button onClick={handleFeedback} disabled={feedbackSubmitting}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white text-sm py-2 rounded-lg font-medium disabled:opacity-50 transition">
                {feedbackSubmitting ? 'Submitting…' : 'Submit Feedback'}
              </button>
            </div>
          )}

          {feedbackDone && (
            <div className="text-sm text-emerald-600 bg-emerald-50 p-3 rounded-xl border border-emerald-100">
              ✓ Feedback submitted — thank you for helping improve your experience.
            </div>
          )}

          {/* Feature 10: Trust History */}
          {trustHistory.length > 0 && (
            <div className="bg-blue-50/50 p-3 rounded-xl border border-blue-100">
              <p className="text-xs font-medium text-gray-600 mb-2">System Alignment History</p>
              <div className="flex items-end gap-1 h-8">
                {trustHistory.map((t, i) => (
                  <div key={i} className="flex-1 bg-blue-400 rounded-t transition-all" style={{ height: `${t * 100}%`, minWidth: 4, maxWidth: 20 }}
                    title={`${(t * 100).toFixed(0)}%`} />
                ))}
              </div>
            </div>
          )}

        </div>
      )}
    </div>
  )
}

/* ═══════════ SUB-COMPONENTS ═══════════ */

function SeverityBar({ severity, bandLower, bandUpper }: {
  severity: number
  bandLower?: number
  bandUpper?: number
}) {
  const pct = severity * 100
  const color = pct < 30 ? 'bg-green-500' : pct < 60 ? 'bg-yellow-500' : 'bg-red-500'

  const hasband = typeof bandLower === 'number' && typeof bandUpper === 'number'
  const lo = hasband ? safe(bandLower!, 0) * 100 : 0
  const hi = hasband ? safe(bandUpper!, 1) * 100 : 0

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="font-medium text-gray-700">Severity</span>
        <span className="font-semibold text-gray-800">{pct.toFixed(0)}%</span>
      </div>
      <div className="relative w-full h-3 bg-gray-100 rounded-full overflow-hidden">
        {/* Feature 3: Confidence Band */}
        {hasband && (
          <div className="absolute h-full bg-gray-300/40 rounded-full" style={{ left: `${lo}%`, width: `${hi - lo}%` }} />
        )}
        {/* Severity Fill */}
        <div className={`absolute h-full ${color} rounded-full transition-all duration-700 ease-out`} style={{ width: `${pct}%` }} />
      </div>
      {hasband && (
        <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
          <span>{lo.toFixed(0)}%</span>
          <span>{hi.toFixed(0)}%</span>
        </div>
      )}
    </div>
  )
}

function TrendBadge({ trend }: { trend: string }) {
  if (!trend || trend === 'unknown') {
    return (
      <span className="inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full bg-gray-50 text-gray-500 border border-gray-200">
        ➖ Trend: unknown
      </span>
    )
  }

  const config: Record<string, { icon: string; bg: string; text: string; border: string }> = {
    improving: { icon: '📈', bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200' },
    worsening: { icon: '📉', bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200' },
    stable:    { icon: '➡️', bg: 'bg-blue-50', text: 'text-blue-700', border: 'border-blue-200' },
  }

  const c = config[trend] || config.stable

  return (
    <span className={`inline-flex items-center gap-1 text-xs px-2.5 py-1 rounded-full border ${c.bg} ${c.text} ${c.border}`}>
      {c.icon} Trend: {trend}
    </span>
  )
}

function safe(v: number | undefined, d: number) {
  if (typeof v !== 'number' || Number.isNaN(v)) return d
  return Math.max(0, Math.min(1, v))
}