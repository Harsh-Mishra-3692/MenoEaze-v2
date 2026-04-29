// lib/ml/index.ts

import { calculateRisk } from "./riskEngine"
import { calculateForecast } from "./forecastEngine"
import { calculateCorrelations } from "./correlationEngine"
import { SymptomLog } from "./riskEngine"

export function analyzeUserData(logs: SymptomLog[]) {
  const risk = calculateRisk(logs)
  const forecast = calculateForecast(logs)
  const correlations = calculateCorrelations(logs)

  return {
    ...risk,
    forecast,
    correlations
  }
}