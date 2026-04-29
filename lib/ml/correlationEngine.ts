// lib/ml/correlationEngine.ts

import { SymptomLog } from "./riskEngine"

export interface CorrelationResult {
    sleepVsSeverity: number
    moodVsSeverity: number
}

function pearson(x: number[], y: number[]) {
    const n = x.length
    if (n < 3) return 0

    const meanX = x.reduce((a, b) => a + b, 0) / n
    const meanY = y.reduce((a, b) => a + b, 0) / n

    const numerator = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0)
    const denomX = Math.sqrt(x.reduce((sum, xi) => sum + (xi - meanX) ** 2, 0))
    const denomY = Math.sqrt(y.reduce((sum, yi) => sum + (yi - meanY) ** 2, 0))

    return denomX * denomY === 0 ? 0 : numerator / (denomX * denomY)
}

export function calculateCorrelations(logs: SymptomLog[]): CorrelationResult {
    const sleep = logs.map(l => l.sleep ?? l.sleep_hours ?? 0)
    const severity = logs.map(l => l.severity ?? l.symptom_severity ?? 0)
    const mood = logs.map(l => l.mood_score ?? (typeof l.mood === 'string' ? 5 : 0))

    return {
        sleepVsSeverity: pearson(sleep, severity),
        moodVsSeverity: pearson(mood, severity)
    }
}