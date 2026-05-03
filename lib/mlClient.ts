// code file of mlClient.ts — ELITE v5 (ZERO-TRUST + BACKWARD COMPAT + FAIL-SAFE)

export const ML_API_URL =
  process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000'

// ─────────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────────
export interface MLPrediction {
  severity: number
  confidence: number
  prediction_id?: string | null
  personalized?: boolean
}

export interface MLRag {
  answer: string
  sources?: any[]
}

export interface MLDoctor {
  recommend: boolean
  urgency?: 'none' | 'optional' | 'moderate' | 'soon' | 'immediate'
}

export interface MLReasoning {
  trend?: string
  anomaly?: boolean
  override?: boolean
  confidence_band?: {
    lower: number
    upper: number
  }
  factors?: {
    trust?: number
    rag_quality?: number
  }
}

export interface MLMeta {
  avg_severity?: number
  trend?: string
  volatility?: number
  signal_strength?: number
}

export interface MLRunResponse {
  status: 'ok' | 'degraded'
  action: 'predict' | 'log' | 'feedback'
  prediction?: MLPrediction
  rag?: MLRag
  doctor?: MLDoctor
  meta?: MLMeta
  reasoning?: MLReasoning
  latency_ms?: number
  request_id?: string
  trust_score?: number
  reasons?: string[]
}

// ─────────────────────────────────────────────
// ERROR
// ─────────────────────────────────────────────
export class MLClientError extends Error {
  code?: string
  constructor(message: string, code?: string) {
    super(message)
    this.name = 'MLClientError'
    this.code = code
  }
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────
function clamp(v: any): number {
  const x = Number(v)
  if (!isFinite(x)) return 0.5
  return Math.max(0, Math.min(1, x))
}

function safeString(v: any, max = 500): string {
  try {
    return String(v ?? '').trim().slice(0, max)
  } catch {
    return ''
  }
}

function safeObject<T = Record<string, any>>(x: any): T {
  if (!x || typeof x !== 'object') return {} as T
  return x as T
}

// ─────────────────────────────────────────────
// AUTH (SAFE + NON-BLOCKING)
// ─────────────────────────────────────────────
async function getAuthTokenSafe(): Promise<string | null> {
  try {
    const { createClient } = await import('@supabase/supabase-js')

    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    )

    const { data } = await supabase.auth.getSession()
    return data?.session?.access_token || null
  } catch {
    return null // never hard fail UI
  }
}

// ─────────────────────────────────────────────
// NETWORK LAYER
// ─────────────────────────────────────────────
async function safeFetch(
  body: any,
  timeout = 10000,
  retries = 1
): Promise<any> {
  let lastError: any

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const token = await getAuthTokenSafe()

      const controller = new AbortController()
      const id = setTimeout(() => controller.abort(), timeout)

      const res = await fetch(`${ML_API_URL}/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      })

      clearTimeout(id)

      if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new MLClientError(
          `HTTP ${res.status}: ${text || 'Request failed'}`,
          'HTTP'
        )
      }

      return await res.json()
    } catch (err: any) {
      lastError = err
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 300))
      }
    }
  }

  throw new MLClientError(
    lastError?.message || 'Network failure',
    'NETWORK'
  )
}

// ─────────────────────────────────────────────
// CORE EXECUTION
// ─────────────────────────────────────────────
async function runAction(
  action: 'log' | 'predict' | 'feedback',
  payload: Record<string, any>
): Promise<MLRunResponse> {
  const raw = await safeFetch({ action, payload })
  const data = safeObject<MLRunResponse>(raw)

  if (data.status !== 'ok' && data.status !== 'degraded') {
    throw new MLClientError('Invalid API status')
  }

  return data
}

// ─────────────────────────────────────────────
// ACTION: LOG
// ─────────────────────────────────────────────
export async function logSymptom(
  featureVector: number[],
  notes: string = '',
  emoji: string = ''
) {
  const vector = featureVector

  if (!Array.isArray(vector) || vector.length !== 11) {
    throw new MLClientError('Feature vector must be length 11')
  }

  const safeVector = vector.map(v =>
    Math.max(0, Math.min(10, Number(v)))
  )

  return runAction('log', {
    feature_vector: safeVector,
    notes: safeString(notes, 300),
    emoji: safeString(emoji, 10),
  })
}

// ─────────────────────────────────────────────
// ACTION: PREDICT
// ─────────────────────────────────────────────
export async function runMLPipeline(
  symptoms: string
): Promise<MLRunResponse> {
  const clean = safeString(symptoms, 500)

  if (!clean || clean.length < 3) {
    throw new MLClientError('Invalid symptoms')
  }

  const data = await runAction('predict', { symptoms: clean })

  // SAFE PARSING (never crash UI)
  const pred = safeObject<MLPrediction>(data.prediction)
  const rag = safeObject<MLRag>(data.rag)

  pred.severity = typeof pred.severity === 'number' ? pred.severity : 0.5
  pred.confidence = typeof pred.confidence === 'number' ? pred.confidence : 0.5
  rag.answer = typeof rag.answer === 'string' ? rag.answer : 'No response available.'

  pred.severity = clamp(pred.severity)
  pred.confidence = clamp(pred.confidence)

  data.prediction = pred
  data.rag = rag

  return data
}

// ─────────────────────────────────────────────
// ACTION: FEEDBACK
// ─────────────────────────────────────────────
export async function submitFeedback(
  predictionId: string,
  predicted: number,
  actual: number,
  rating: number
) {
  if (!predictionId) {
    throw new MLClientError('Missing prediction_id')
  }

  return runAction('feedback', {
    prediction_id: predictionId,
    predicted: clamp(predicted),
    actual: clamp(actual),
    rating: Math.max(1, Math.min(10, Number(rating))),
  })
}