'use client'

import { useEffect, useMemo, useState } from 'react'
import { supabase } from '@/lib/supabase'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend
} from 'recharts'

type Symptom = {
  severity: number
  mood_score: number | null
  symptom_type: string | null
  created_at: string
  hot_flash_score: number | null
  night_sweats_score: number | null
  sleep_quality: number | null
  fatigue_score: number | null
  anxiety_score: number | null
  physical_activity: number | null
  stress_level: number | null
  caffeine_intake: number | null
  age: number | null
  bmi: number | null
}

export default function AnalysisCard({ userId }: { userId: string }) {
  const [logs, setLogs] = useState<Symptom[]>([])
  const [loading, setLoading] = useState(true)
  const [view, setView] = useState<'weekly' | 'monthly'>('weekly')
  const [mlForecast, setMlForecast] = useState<number | null>(null)
  const [mlStrategy, setMlStrategy] = useState<string | null>(null)
  const [mlConfidence, setMlConfidence] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [aiInsight, setAiInsight] = useState<string | null>(null)
  const [insightLoading, setInsightLoading] = useState(false)

  useEffect(() => {
    let isMounted = true

    const fetchLogs = async () => {
      const { data, error } = await supabase
        .from('symptom_logs')
        .select('severity, mood_score, symptom_type, created_at, hot_flash_score, night_sweats_score, sleep_quality, fatigue_score, anxiety_score, physical_activity, stress_level, caffeine_intake, age, bmi')
        .eq('user_id', userId)
        .eq('is_deleted', false)
        .order('created_at', { ascending: true })

      if (!isMounted) return

      if (error) {
        setError('Unable to load your analytics.')
      } else {
        setLogs(data || [])
      }

      setLoading(false)
    }

    fetchLogs()
    return () => { isMounted = false }
  }, [userId])

  // Fetch AI insight summary
  useEffect(() => {
    if (logs.length === 0) return

    const fetchInsight = async () => {
      setInsightLoading(true)
      try {
        const res = await fetch('/api/insight-summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ userId })
        })
        const data = await res.json()
        if (data.summary) {
          setAiInsight(data.summary)
        }
      } catch {
        // Silent fail — fallback text shown
      } finally {
        setInsightLoading(false)
      }
    }

    fetchInsight()
  }, [userId, logs.length])

  useEffect(() => {
    if (logs.length <= 5) return

    const controller = new AbortController()

    const callML = async () => {
      try {
        // Build a (5, 11) sequence from the last 5 logs using real data
        const recent = logs.slice(-5)
        const sequence = recent.map(l => {
          const sev = (l.severity ?? 0) / 10
          return [
            (l.hot_flash_score ?? sev * 10) / 10,
            (l.night_sweats_score ?? sev * 9) / 10,
            (l.sleep_quality ?? (1 - sev) * 10) / 10,
            (l.mood_score ?? (1 - sev) * 10) / 10,
            (l.fatigue_score ?? sev * 8) / 10,
            (l.anxiety_score ?? sev * 7) / 10,
            (l.physical_activity ?? Math.max(0, 5 - sev * 3)) / 10,
            (l.stress_level ?? sev * 6) / 10,
            (l.caffeine_intake ?? 2) / 5,
            (l.age ?? 50 - 30) / 40,
            (l.bmi ?? 25 - 15) / 35,
          ]
        })

        const mlApiUrl = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000'

        const response = await fetch(`${mlApiUrl}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: userId,
            sequence,
            request_guidance: false,
          }),
          signal: controller.signal
        })

        if (!response.ok) return

        const data = await response.json()
        if (typeof data.severity === 'number') {
          setMlForecast(Number((data.severity * 10).toFixed(2)))
        }
        if (data.strategy) {
          setMlStrategy(data.strategy)
        }
        if (typeof data.confidence === 'number') {
          setMlConfidence(data.confidence)
        }
      } catch {
        // silent fail — ML backend may not be running
      }
    }

    callML()
    return () => controller.abort()
  }, [logs, userId])

  const analytics = useMemo(() => {
    if (logs.length === 0) return null

    const now = Date.now()
    const range =
      view === 'weekly'
        ? 7 * 24 * 60 * 60 * 1000
        : 30 * 24 * 60 * 60 * 1000

    const filtered = logs.filter(
      l => new Date(l.created_at).getTime() >= now - range
    )

    if (filtered.length === 0) return null

    const movingAverage = filtered.map((_, index) => {
      const start = Math.max(0, index - 6)
      const subset = filtered.slice(start, index + 1)
      const avg = subset.reduce((sum, l) => sum + l.severity, 0) / subset.length

      return {
        date: new Date(filtered[index].created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        severity: filtered[index].severity,
        smoothed: Number(avg.toFixed(2))
      }
    })

    const n = filtered.length
    if (n < 2) {
      return { movingAverage, forecastNext: filtered[n - 1].severity, trend: 'Stable' }
    }

    const x = Array.from({ length: n }, (_, i) => i)
    const y = filtered.map(l => l.severity)
    const xMean = x.reduce((a, b) => a + b, 0) / n
    const yMean = y.reduce((a, b) => a + b, 0) / n
    const numerator = x.reduce((sum, xi, i) => sum + (xi - xMean) * (y[i] - yMean), 0)
    const denominator = x.reduce((sum, xi) => sum + (xi - xMean) ** 2, 0)
    const slope = denominator === 0 ? 0 : numerator / denominator
    const intercept = yMean - slope * xMean
    const forecastNext = Math.max(0, Math.min(10, intercept + slope * (n + 1)))

    const trend = slope > 0.1 ? 'Increasing' : slope < -0.1 ? 'Decreasing' : 'Stable'

    return { movingAverage, forecastNext: Number(forecastNext.toFixed(2)), trend }
  }, [logs, view])

  if (loading) {
    return (
      <div className="h-72 rounded-2xl bg-purple-50/50 animate-pulse flex items-center justify-center">
        <p className="text-sm text-gray-400">Loading your insights…</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 rounded-2xl bg-rose-50 text-rose-600 text-sm">
        {error}
      </div>
    )
  }

  if (!analytics) {
    return (
      <div className="py-16 text-center">
        <p className="text-gray-400 text-sm mb-1">No insights yet</p>
        <p className="text-gray-400 text-xs">Start logging symptoms to see your trends and patterns 💜</p>
      </div>
    )
  }

  const trendColor =
    analytics.trend === 'Decreasing' ? 'text-emerald-600' :
      analytics.trend === 'Increasing' ? 'text-rose-600' :
        'text-gray-600'

  const trendEmoji =
    analytics.trend === 'Decreasing' ? '📉' :
      analytics.trend === 'Increasing' ? '📈' :
        '➡️'

  // Build fallback insight text
  const fallbackInsight =
    analytics.trend === 'Decreasing'
      ? "Your symptoms appear to be easing — that's wonderful progress. Keep up with the habits that are helping you feel better."
      : analytics.trend === 'Increasing'
        ? "Your symptoms seem to be intensifying recently. Consider tracking any new triggers, and don't hesitate to reach out to your healthcare provider."
        : "Your symptoms are holding steady. Consistent tracking helps us spot patterns — you're doing great by staying engaged with your health."

  return (
    <div className="space-y-6">

      {/* View Toggle */}
      <div className="flex gap-2">
        {(['weekly', 'monthly'] as const).map(mode => (
          <button
            key={mode}
            onClick={() => setView(mode)}
            className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all duration-200 ${view === mode
              ? 'bg-purple-600 text-white shadow-sm'
              : 'bg-purple-50 text-purple-600 hover:bg-purple-100'
              }`}
          >
            {mode.charAt(0).toUpperCase() + mode.slice(1)}
          </button>
        ))}
      </div>

      {/* Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={analytics.movingAverage}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f3e8ff" />
            <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9ca3af' }} />
            <YAxis domain={[0, 10]} tick={{ fontSize: 11, fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(255,255,255,0.95)',
                borderRadius: '12px',
                border: '1px solid #e9d5ff',
                boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                fontSize: '12px'
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }}
            />
            <Line
              type="monotone"
              dataKey="smoothed"
              stroke="#a855f7"
              strokeWidth={2.5}
              dot={false}
              name="7-Day Avg"
            />
            <Line
              type="monotone"
              dataKey="severity"
              stroke="#ec4899"
              strokeWidth={1.5}
              dot={{ r: 3, fill: '#ec4899' }}
              name="Severity"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Insights Row */}
      <div className="grid sm:grid-cols-2 gap-3">
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-r from-purple-50 to-indigo-50 rounded-xl p-4 border border-purple-100/60"
        >
          <p className="text-xs text-gray-500 font-medium mb-1">
            {trendEmoji} Forecast
          </p>
          <p className="text-lg font-semibold text-gray-800">
            {analytics.forecastNext}<span className="text-sm text-gray-400 font-normal">/10</span>
          </p>
          <p className={`text-xs font-medium mt-1 ${trendColor}`}>
            Trend: {analytics.trend}
          </p>
        </motion.div>

        {mlForecast !== null && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-4 border border-blue-100/60"
          >
            <div className="flex items-center justify-between mb-1">
              <p className="text-xs text-gray-500 font-medium">
                🤖 ML Prediction
              </p>
              {mlStrategy === 'maml' && (
                <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-gradient-to-r from-amber-400 to-orange-400 text-white animate-pulse">
                  ⚡ MAML Active
                </span>
              )}
              {mlStrategy === 'bias' && (
                <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-purple-100 text-purple-600">
                  Bias-Corrected
                </span>
              )}
            </div>
            <p className="text-lg font-semibold text-gray-800">
              {mlForecast}<span className="text-sm text-gray-400 font-normal">/10</span>
            </p>
            <div className="flex items-center gap-2 mt-1">
              <p className="text-xs text-gray-500">
                {mlStrategy === 'maml' ? 'Hyper-Personalized' : 'AI severity estimate'}
              </p>
              {mlConfidence !== null && (
                <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-blue-100 text-blue-600">
                  {Math.round(mlConfidence * 100)}% conf
                </span>
              )}
            </div>
          </motion.div>
        )}
      </div>

      {/* AI Insight Summary */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="bg-gradient-to-r from-rose-50/80 to-pink-50/80 rounded-xl p-4 border border-pink-100/60"
      >
        <p className="text-xs text-gray-500 font-medium mb-1">💜 AI Insight</p>
        {insightLoading ? (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-purple-400 animate-pulse" />
            <p className="text-sm text-gray-400">Analyzing your patterns…</p>
          </div>
        ) : (
          <p className="text-sm text-gray-600 leading-relaxed">
            {aiInsight || fallbackInsight}
          </p>
        )}
      </motion.div>
    </div>
  )
}