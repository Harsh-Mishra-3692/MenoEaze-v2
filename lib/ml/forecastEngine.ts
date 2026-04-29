// lib/ml/forecastEngine.ts

import { SymptomLog } from "./riskEngine"

export interface ForecastPoint {
    day: number
    predictedSeverity: number
}

function mean(values: number[]) {
    return values.reduce((a, b) => a + b, 0) / values.length
}

function linearRegression(values: number[]) {
    const n = values.length
    const x = Array.from({ length: n }, (_, i) => i)
    const xMean = mean(x)
    const yMean = mean(values)

    const numerator = x.reduce((sum, xi, i) => sum + (xi - xMean) * (values[i] - yMean), 0)
    const denominator = x.reduce((sum, xi) => sum + (xi - xMean) ** 2, 0)

    const slope = denominator === 0 ? 0 : numerator / denominator
    const intercept = yMean - slope * xMean

    return { slope, intercept }
}

export function calculateForecast(logs: SymptomLog[], horizon = 7): ForecastPoint[] {
    if (logs.length < 3) return []

    const severities = logs.map(l => l.severity ?? l.symptom_severity ?? 0)
    const { slope, intercept } = linearRegression(severities)

    const startIndex = severities.length

    const forecast: ForecastPoint[] = []

    for (let i = 0; i < horizon; i++) {
        const predicted = intercept + slope * (startIndex + i)
        forecast.push({
            day: i + 1,
            predictedSeverity: Math.max(0, Math.min(10, predicted))
        })
    }

    return forecast
}