"use client"

interface ForecastPoint {
  day: number
  predictedSeverity: number
}

interface ForecastChartProps {
  forecast: ForecastPoint[]
}

export default function ForecastChart({ forecast }: ForecastChartProps) {
  if (!forecast || forecast.length === 0) {
    return (
      <div className="bg-white rounded-2xl shadow p-6">
        <h2 className="text-lg font-semibold text-gray-800">
          7-Day Severity Forecast
        </h2>
        <p className="text-sm text-gray-500 mt-4">
          Not enough data to generate forecast.
        </p>
      </div>
    )
  }

  const width = 500
  const height = 200
  const padding = 40

  const maxSeverity = 10
  const minSeverity = 0

  const xStep = (width - padding * 2) / (forecast.length - 1)

  const points = forecast.map((point, i) => {
    const x = padding + i * xStep
    const y =
      height -
      padding -
      ((point.predictedSeverity - minSeverity) /
        (maxSeverity - minSeverity)) *
        (height - padding * 2)
    return `${x},${y}`
  })

  return (
    <div className="bg-white rounded-2xl shadow p-6 space-y-4">
      <h2 className="text-lg font-semibold text-gray-800">
        7-Day Severity Forecast
      </h2>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto"
      >
        {/* Axes */}
        <line
          x1={padding}
          y1={height - padding}
          x2={width - padding}
          y2={height - padding}
          stroke="#e5e7eb"
        />
        <line
          x1={padding}
          y1={padding}
          x2={padding}
          y2={height - padding}
          stroke="#e5e7eb"
        />

        {/* Line */}
        <polyline
          fill="none"
          stroke="#6366f1"
          strokeWidth="3"
          points={points.join(" ")}
        />

        {/* Dots */}
        {forecast.map((point, i) => {
          const x = padding + i * xStep
          const y =
            height -
            padding -
            (point.predictedSeverity / 10) *
              (height - padding * 2)

          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r="4"
              fill="#6366f1"
            />
          )
        })}
      </svg>

      <div className="flex justify-between text-xs text-gray-500">
        {forecast.map(point => (
          <span key={point.day}>Day {point.day}</span>
        ))}
      </div>
    </div>
  )
}