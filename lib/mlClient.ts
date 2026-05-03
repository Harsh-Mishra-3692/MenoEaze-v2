export const ML_API_URL = process.env.NEXT_PUBLIC_ML_API_URL || 'http://localhost:8000'

export interface MLPrediction {
  severity: number
  confidence: number
  prediction_id?: string
}

export interface MLRag {
  answer: string
  sources: any[]
}

export interface MLDoctor {
  recommend: boolean
  urgency?: 'none' | 'optional' | 'moderate' | 'soon' | 'immediate'
}

export interface MLRunResponse {
  prediction: MLPrediction
  rag: MLRag
  doctor?: MLDoctor
  latency_ms: number
  reasoning?: any
}

export class MLClientError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MLClientError'
  }
}

export async function runMLPipeline(userId: string, symptoms: string): Promise<MLRunResponse> {
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!userId || !uuidRegex.test(userId)) {
    console.error(`[ML][ERROR][${userId || 'unknown'}][reason=Invalid user_id UUID]`);
    throw new MLClientError('Invalid input: A valid user account is required.')
  }
  if (!symptoms || symptoms.length < 3) {
    console.error(`[ML][ERROR][${userId}][reason=Invalid symptoms length]`);
    throw new MLClientError('Invalid input: symptoms are required.')
  }

  console.log(`[ML][START][${userId}]`)

  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 10000) // 10s timeout

    const res = await fetch(`${ML_API_URL}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, symptoms }),
      signal: controller.signal
    })
    
    clearTimeout(timeoutId)

    if (!res.ok) {
      throw new MLClientError(`ML API returned ${res.status}: ${res.statusText}`)
    }

    const data = await res.json()

    // Strict schema validation
    if (
      !data ||
      !data.prediction ||
      typeof data.prediction.severity !== 'number' ||
      typeof data.prediction.confidence !== 'number' ||
      !data.rag ||
      typeof data.rag.answer !== 'string'
    ) {
      throw new MLClientError('Malformed response from ML API')
    }

    const sev = data.prediction.severity
    const conf = data.prediction.confidence

    // Value bounds validation
    if (sev < 0 || sev > 1 || conf < 0 || conf > 1) {
      throw new MLClientError('Invalid prediction bounds (severity/confidence must be 0-1)')
    }

    console.log(`[ML][SUCCESS][${userId}][severity=${sev.toFixed(4)}]`)
    return data as MLRunResponse
  } catch (err: any) {
    console.error(`[ML][ERROR][${userId}][reason=${err.message}]`)
    throw new MLClientError(err.message || 'ML Pipeline failed')
  }
}

export async function logSymptom(userId: string, featureVector: number[], notes: string = '', emoji: string = '') {
  if (featureVector.length !== 11) {
    throw new MLClientError('Feature vector must be exactly 11 elements')
  }

  const res = await fetch(`${ML_API_URL}/log-symptom`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: userId, feature_vector: featureVector, notes, emoji }),
  })

  if (!res.ok) {
    throw new MLClientError(`Failed to log symptom: ${res.statusText}`)
  }

  return res.json()
}
