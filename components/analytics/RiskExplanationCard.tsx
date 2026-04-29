"use client"

interface RiskExplanationProps {
  riskScore: number
  riskLevel: "Low" | "Moderate" | "High"
  trend: string
  volatility: number
}

function getRiskColor(level: string) {
  if (level === "High") return "bg-rose-500"
  if (level === "Moderate") return "bg-amber-500"
  return "bg-emerald-500"
}

export default function RiskExplanationCard({
  riskScore,
  riskLevel,
  trend,
  volatility
}: RiskExplanationProps) {
  const contributions = [
    { label: "Severity Average", value: riskScore },
    { label: "Trend Impact", value: Math.abs(riskScore * 0.2) },
    { label: "Volatility Impact", value: volatility * 5 }
  ]

  const maxContribution = Math.max(
    ...contributions.map(c => c.value),
    1
  )

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-6">
      <h2 className="text-lg font-semibold text-gray-800">
        Risk Analysis & Explainability
      </h2>

      {/* Risk Badge */}
      <div className="flex items-center justify-between">
        <div className="text-3xl font-bold text-gray-800">
          {riskScore.toFixed(1)}%
        </div>
        <span
          className={`text-xs text-white px-3 py-1 rounded-full ${getRiskColor(
            riskLevel
          )}`}
        >
          {riskLevel}
        </span>
      </div>

      <div className="text-sm text-gray-500">
        Trend: {trend} | Volatility: {volatility.toFixed(2)}
      </div>

      {/* SHAP-style contributions */}
      <div className="space-y-3">
        {contributions.map((c, i) => (
          <div key={i} className="space-y-1">
            <div className="flex justify-between text-xs text-gray-600">
              <span>{c.label}</span>
              <span>{c.value.toFixed(1)}</span>
            </div>
            <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full bg-indigo-500"
                style={{
                  width: `${(c.value / maxContribution) * 100}%`
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}