// lib/ml/riskEngine.ts

export interface SymptomLog {
    symptom_type?: string
    severity?: number
    symptom_severity?: number
    mood?: string
    mood_score?: number
    sleep?: number
    sleep_hours?: number
    notes?: string
    date?: string
    created_at?: string
}

export interface RiskResult {
    riskScore: number
    riskLevel: "Low" | "Moderate" | "High"
    trend: "Improving" | "Stable" | "Worsening"
    volatility: number
}

function mean(values: number[]) {
    return values.reduce((a, b) => a + b, 0) / values.length
}

function std(values: number[]) {
    const m = mean(values)
    return Math.sqrt(mean(values.map(v => (v - m) ** 2)))
}

function linearSlope(values: number[]) {
    const n = values.length
    const x = Array.from({ length: n }, (_, i) => i)
    const xMean = mean(x)
    const yMean = mean(values)

    const numerator = x.reduce((sum, xi, i) => sum + (xi - xMean) * (values[i] - yMean), 0)
    const denominator = x.reduce((sum, xi) => sum + (xi - xMean) ** 2, 0)

    return denominator === 0 ? 0 : numerator / denominator
}

export function calculateRisk(logs: SymptomLog[]): RiskResult {
    if (!logs.length) {
        return {
            riskScore: 0,
            riskLevel: "Low",
            trend: "Stable",
            volatility: 0
        }
    }

    const severities = logs.map(l => l.severity ?? l.symptom_severity ?? 0)

    const avgSeverity = mean(severities)
    const volatility = std(severities)
    const slope = linearSlope(severities)

    const normalizedRisk = Math.min(100, avgSeverity * 10)

    let riskLevel: RiskResult["riskLevel"] = "Low"
    if (normalizedRisk >= 70) riskLevel = "High"
    else if (normalizedRisk >= 40) riskLevel = "Moderate"

    let trend: RiskResult["trend"] = "Stable"
    if (slope > 0.1) trend = "Worsening"
    if (slope < -0.1) trend = "Improving"

    return {
        riskScore: normalizedRisk,
        riskLevel,
        trend,
        volatility
    }
}