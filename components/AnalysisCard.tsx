'use client'

import React, { useState } from 'react'

type RiskLevel = 'low' | 'medium' | 'high'
type Trend = 'increasing' | 'decreasing' | 'stable' | 'unknown'

interface HistoryPoint {
  severity: number
  confidence?: number
  timestamp?: string
  actual?: number // 🔥 user feedback
}

interface Props {
  data?: {
    prediction?: {
      severity: number
      confidence: number
    }
    risk?: {
      risk_level?: RiskLevel
      trend?: Trend
      doctor_recommended?: boolean
      urgency?: 'none' | 'optional' | 'moderate' | 'soon' | 'immediate'
      message?: string
    }
    history?: HistoryPoint[]
    reasoning?: string
  }
}

export default function AnalysisCard({ data }: Props) {
  const [showReasoning, setShowReasoning] = useState(false)

  const severity = safe(data?.prediction?.severity, 0.5)
  const confidence = safe(data?.prediction?.confidence, 0.0)

  const history = (data?.history || []).slice(-5)

  const riskLevel: RiskLevel = data?.risk?.risk_level || 'low'
  const trend: Trend = data?.risk?.trend || 'unknown'
  const doctorRecommended = data?.risk?.doctor_recommended || false
  const urgency = data?.risk?.urgency || 'none'
  const message = data?.risk?.message || ''
  const reasoning = data?.reasoning || 'No additional insights available.'

  const riskColor = getRiskColor(riskLevel)
  const trendIcon = getTrendIcon(trend)
  const urgencyColor = getUrgencyColor(urgency)

  return (
    <div className="bg-white rounded-2xl shadow-md p-5 space-y-6 border">

      {/* HEADER */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Health Analysis</h3>
        <div className={`px-3 py-1 rounded-full text-xs font-semibold ${riskColor}`}>
          {riskLevel.toUpperCase()} RISK
        </div>
      </div>

      {/* SEVERITY */}
      <Bar label="Severity" value={severity} />

      {/* CONFIDENCE */}
      <div className="text-sm text-gray-600">
        Confidence: {(confidence * 100).toFixed(0)}%
        <span className="ml-2 text-gray-400">
          ±{Math.round((1 - confidence) * 20)}%
        </span>
      </div>

      {/* TREND */}
      <div className="flex items-center gap-2 text-sm">
        Trend: <span>{trendIcon}</span> {trend}
      </div>

      {/* 📊 TIME SERIES + CONFIDENCE BAND */}
      {history.length > 1 && (
        <TimeSeriesChart history={history} />
      )}

      {/* 📊 FEEDBACK VS PREDICTION */}
      {history.some(h => h.actual !== undefined) && (
        <FeedbackChart history={history} />
      )}

      {/* DOCTOR CTA */}
      {doctorRecommended && (
        <div className={`p-3 rounded-xl text-sm font-medium ${urgencyColor}`}>
          🩺 {getDoctorMessage(urgency)}
        </div>
      )}

      {/* MESSAGE */}
      {message && (
        <div className="text-sm bg-gray-50 p-3 rounded-xl">{message}</div>
      )}

      {/* AI REASONING */}
      <div>
        <button
          onClick={() => setShowReasoning(!showReasoning)}
          className="text-sm text-purple-600 font-medium"
        >
          {showReasoning ? 'Hide AI reasoning ▲' : 'Show AI reasoning ▼'}
        </button>

        {showReasoning && (
          <div className="mt-2 text-sm bg-gray-50 p-3 rounded-xl">
            {reasoning}
          </div>
        )}
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────
   📊 TIME SERIES WITH CONFIDENCE BAND
   ───────────────────────────────────────────── */

function TimeSeriesChart({ history }: { history: HistoryPoint[] }) {
  const width = 100
  const height = 40

  const points = history.map((p, i) => {
    const x = (i / (history.length - 1)) * width
    const y = height - p.severity * height
    return { x, y, conf: p.confidence ?? 0.5 }
  })

  const line = points.map(p => `${p.x},${p.y}`).join(' ')

  const bandTop = points.map(p =>
    `${p.x},${height - Math.min(1, p.severity + (1 - p.conf) * 0.2) * height}`
  ).join(' ')

  const bandBottom = points.map(p =>
    `${p.x},${height - Math.max(0, p.severity - (1 - p.conf) * 0.2) * height}`
  ).reverse().join(' ')

  return (
    <div>
      <p className="text-sm font-medium mb-2">Trend Over Time</p>
      <svg viewBox="0 0 100 40" className="w-full h-24">

        {/* CONFIDENCE BAND */}
        <polygon
          points={`${bandTop} ${bandBottom}`}
          fill="rgba(139,92,246,0.2)"
        />

        {/* LINE */}
        <polyline
          fill="none"
          stroke="#8b5cf6"
          strokeWidth="2"
          points={line}
        />

        {/* POINTS */}
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r="2" fill="#8b5cf6" />
        ))}
      </svg>
    </div>
  )
}

/* ─────────────────────────────────────────────
   📊 FEEDBACK VS PREDICTION
   ───────────────────────────────────────────── */

function FeedbackChart({ history }: { history: HistoryPoint[] }) {
  const valid = history.filter(h => h.actual !== undefined)

  return (
    <div>
      <p className="text-sm font-medium mb-2">
        Prediction vs Actual Feedback
      </p>

      <svg viewBox="0 0 100 40" className="w-full h-24">

        {/* PREDICTED */}
        <polyline
          fill="none"
          stroke="#8b5cf6"
          strokeWidth="2"
          points={valid.map((p, i) =>
            `${(i / (valid.length - 1)) * 100},${40 - p.severity * 40}`
          ).join(' ')}
        />

        {/* ACTUAL */}
        <polyline
          fill="none"
          stroke="#ef4444"
          strokeWidth="2"
          points={valid.map((p, i) =>
            `${(i / (valid.length - 1)) * 100},${40 - (p.actual ?? 0) * 40}`
          ).join(' ')}
        />
      </svg>

      <div className="text-xs text-gray-500 mt-1">
        Purple = Prediction | Red = Actual
      </div>
    </div>
  )
}

/* ───────────────────────────────────────────── */

function Bar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span>{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}

function safe(v: number | undefined, d: number) {
  if (typeof v !== 'number' || isNaN(v)) return d
  return Math.max(0, Math.min(1, v))
}

function getRiskColor(level: RiskLevel) {
  return level === 'high'
    ? 'bg-red-100 text-red-700'
    : level === 'medium'
    ? 'bg-yellow-100 text-yellow-700'
    : 'bg-green-100 text-green-700'
}

function getTrendIcon(trend: Trend) {
  return trend === 'increasing'
    ? '📈'
    : trend === 'decreasing'
    ? '📉'
    : trend === 'stable'
    ? '➖'
    : '❔'
}

function getUrgencyColor(u: string) {
  return u === 'immediate'
    ? 'bg-red-100 text-red-700'
    : u === 'soon'
    ? 'bg-orange-100 text-orange-700'
    : u === 'moderate'
    ? 'bg-yellow-100 text-yellow-700'
    : u === 'optional'
    ? 'bg-blue-100 text-blue-700'
    : 'bg-gray-100 text-gray-700'
}

function getDoctorMessage(u: string) {
  return u === 'immediate'
    ? 'Seek immediate medical attention.'
    : u === 'soon'
    ? 'Symptoms worsening — consult a doctor soon.'
    : u === 'moderate'
    ? 'Persistent symptoms — medical consultation recommended.'
    : u === 'optional'
    ? 'Consider consulting a healthcare provider.'
    : ''
}