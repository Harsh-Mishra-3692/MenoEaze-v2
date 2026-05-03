'use client'

import React, { useState, useEffect, useRef } from 'react'
import { supabase } from '@/lib/supabase'
import { runMLPipeline, MLRunResponse } from '@/lib/mlClient'

interface Props {
  userId: string
}

export default function AnalysisCard({ userId }: Props) {
  const [showReasoning, setShowReasoning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [mlData, setMlData] = useState<MLRunResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const isMounted = useRef(true)
  const loadingRef = useRef(false)

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
    }
  }, [])

  const handleRunAnalysis = async () => {
    if (!userId || loadingRef.current) return

    try {
      loadingRef.current = true

      if (isMounted.current) {
        setLoading(true)
        setError(null)
      }

      console.log('[ML][START]', userId)

      const { count, error } = await supabase
        .from('symptom_logs')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', userId)

      if (error) throw new Error('Failed to fetch symptom logs')

      if (!count || count < 5) {
        throw new Error('At least 5 days of data required for analysis')
      }

      const res = await runMLPipeline(userId, 'Run structured health analysis')

      // STRICT VALIDATION
      if (
        !res ||
        !res.prediction ||
        typeof res.prediction.severity !== 'number' ||
        typeof res.prediction.confidence !== 'number'
      ) {
        throw new Error('Invalid ML response')
      }

      const severity = safe(res.prediction.severity, 0.5)
      const confidence = safe(res.prediction.confidence, 0.0)

      console.log('[ML][SUCCESS]', userId, severity)

      if (isMounted.current) {
        setMlData(res)
      }

    } catch (err: any) {
      console.error('[ML][ERROR]', userId, err)

      if (isMounted.current) {
        setError(err.message || 'Analysis failed')
        setMlData(null)
      }

    } finally {
      loadingRef.current = false
      if (isMounted.current) {
        setLoading(false)
      }
    }
  }

  const severity = safe(mlData?.prediction?.severity, 0.5)
  const confidence = safe(mlData?.prediction?.confidence, 0.0)

  const message = mlData?.rag?.answer || ''

  const reasoning =
    'AI Confidence Score: ' +
    (confidence * 100).toFixed(0) +
    '%. ' +
    (mlData?.reasoning?.anomaly ? 'Anomaly detected.' : '')

  return (
    <div className="bg-white rounded-2xl shadow-md p-5 space-y-6 border">

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h3 className="text-lg font-semibold">Health Analysis</h3>

        <button
          onClick={handleRunAnalysis}
          disabled={loading}
          className="w-full sm:w-auto bg-purple-600 hover:bg-purple-700 text-white px-4 py-3 min-h-[44px] rounded-lg text-sm font-semibold disabled:opacity-50"
        >
          {loading ? 'Running...' : 'Run Analysis'}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 text-red-600 p-4 rounded-xl text-sm border">
          <p className="font-medium">Analysis Failed</p>
          <p>{error}</p>
        </div>
      )}

      {!mlData && !loading && !error && (
        <div className="text-center text-gray-500 py-8 bg-gray-50 rounded-xl border">
          Click "Run Analysis" to generate insights
        </div>
      )}

      {mlData && (
        <>
          <Bar label="Severity" value={severity} />

          <div className="text-sm text-gray-600">
            Confidence: {(confidence * 100).toFixed(0)}%
          </div>

          {message && (
            <div className="text-sm bg-gray-50 p-3 rounded-xl">
              {message}
            </div>
          )}

          <button
            onClick={() => setShowReasoning(!showReasoning)}
            className="text-sm text-purple-600"
          >
            {showReasoning ? 'Hide reasoning' : 'Show reasoning'}
          </button>

          {showReasoning && (
            <div className="text-sm bg-gray-50 p-3 rounded-xl">
              {reasoning}
            </div>
          )}
        </>
      )}
    </div>
  )
}

function Bar({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span>{label}</span>
        <span>{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="w-full h-2 bg-gray-100 rounded-full">
        <div
          className="h-full bg-purple-500"
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