"use client"

interface CorrelationProps {
  sleepVsSeverity: number
  moodVsSeverity: number
}

function getStrengthLabel(value: number) {
  const abs = Math.abs(value)

  if (abs > 0.7) return "Strong"
  if (abs > 0.4) return "Moderate"
  if (abs > 0.2) return "Weak"
  return "Minimal"
}

function getDirection(value: number) {
  if (value > 0) return "Positive"
  if (value < 0) return "Negative"
  return "Neutral"
}

export default function CorrelationCard({
  sleepVsSeverity,
  moodVsSeverity
}: CorrelationProps) {
  const correlations = [
    {
      label: "Sleep ↔ Severity",
      value: sleepVsSeverity
    },
    {
      label: "Mood ↔ Severity",
      value: moodVsSeverity
    }
  ]

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-6">
      <h2 className="text-lg font-semibold text-gray-800">
        Correlation Insights
      </h2>

      {correlations.map((item, idx) => (
        <div key={idx} className="space-y-2">
          <div className="flex justify-between text-sm text-gray-600">
            <span>{item.label}</span>
            <span className="font-medium text-gray-800">
              {item.value.toFixed(2)}
            </span>
          </div>

          <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
            <div
              className={`h-full ${
                item.value > 0
                  ? "bg-indigo-500"
                  : item.value < 0
                  ? "bg-rose-500"
                  : "bg-gray-300"
              }`}
              style={{
                width: `${Math.min(Math.abs(item.value) * 100, 100)}%`
              }}
            />
          </div>

          <div className="text-xs text-gray-500 flex justify-between">
            <span>{getDirection(item.value)} correlation</span>
            <span>{getStrengthLabel(item.value)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}