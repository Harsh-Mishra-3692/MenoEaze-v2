// lib/ml/types.ts
// ═══════════════════════════════════════════════
// Shared types for ML backend ↔ Next.js frontend
// Synced with: ml_engine/preprocess.py ALL_FEATURES
// ═══════════════════════════════════════════════

/**
 * The 11 clinical features used by the GRU model.
 * Order MUST match preprocess.py ALL_FEATURES:
 *   TEMPORAL: hot_flash_score, night_sweats_score, sleep_quality,
 *             mood_score, fatigue_score, anxiety_score,
 *             physical_activity, stress_level, caffeine_intake
 *   STATIC:  age, bmi
 */
export interface UserSymptomFeatures {
  hot_flash_score: number      // 0-10
  night_sweats_score: number   // 0-10
  sleep_quality: number        // 0-10
  mood_score: number           // 0-10
  fatigue_score: number        // 0-10
  anxiety_score: number        // 0-10
  physical_activity: number    // 0-10
  stress_level: number         // 0-10
  caffeine_intake: number      // 0-5 cups
  age: number                  // 30-70
  bmi: number                  // 15-50
}

/** Feature keys in the exact order expected by the GRU model */
export const FEATURE_KEYS: (keyof UserSymptomFeatures)[] = [
  'hot_flash_score',
  'night_sweats_score',
  'sleep_quality',
  'mood_score',
  'fatigue_score',
  'anxiety_score',
  'physical_activity',
  'stress_level',
  'caffeine_intake',
  'age',
  'bmi',
]

/** Normalization ranges for converting raw values to [0,1] for the model */
export const FEATURE_RANGES: Record<keyof UserSymptomFeatures, [number, number]> = {
  hot_flash_score:   [0, 10],
  night_sweats_score:[0, 10],
  sleep_quality:     [0, 10],
  mood_score:        [0, 10],
  fatigue_score:     [0, 10],
  anxiety_score:     [0, 10],
  physical_activity: [0, 10],
  stress_level:      [0, 10],
  caffeine_intake:   [0, 5],
  age:               [30, 70],
  bmi:               [15, 50],
}

/** Normalize a raw feature value to [0, 1] */
export function normalizeFeature(key: keyof UserSymptomFeatures, value: number): number {
  const [min, max] = FEATURE_RANGES[key]
  return Math.max(0, Math.min(1, (value - min) / (max - min)))
}

/** Convert a full feature set to a normalized [1, 11] row vector */
export function featuresToRow(features: UserSymptomFeatures): number[] {
  return FEATURE_KEYS.map(k => normalizeFeature(k, features[k]))
}

/**
 * Build the (5, 11) matrix required by the GRU model.
 * If history has < 5 entries, forward-fills by repeating the most recent entry.
 */
export function buildSequenceMatrix(logs: UserSymptomFeatures[]): number[][] {
  if (logs.length === 0) return []

  const rows = logs.map(featuresToRow)

  // Forward-fill: repeat last entry to pad to 5 rows
  while (rows.length < 5) {
    rows.unshift([...rows[0]])
  }

  // Take last 5
  return rows.slice(-5)
}

/** Response from the /predict endpoint */
export interface PredictionResponse {
  severity: number
  strategy: 'none' | 'bias' | 'maml' | 'adapt'
  confidence?: number
  base_severity?: number
  bias?: number
  variance?: number
  latency_ms?: number
  sources?: any[]
}

/** Payload for the /feedback endpoint */
export interface FeedbackPayload {
  user_id: string
  actual_severity: number
  sequence: number[][]
}
