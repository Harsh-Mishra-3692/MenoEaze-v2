"use client"

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Area
} from "recharts"

interface ForecastPoint {
  dayIndex: number
  predictedSeverity: number
  lowerBound: number
  upperBound: number
}

export default function ForecastChart({
  forecast
}: {
  forecast: ForecastPoint[]
}) {
  return (
    <div className="bg-white rounded-2xl shadow-lg p-6">
      <h2 className="text-lg font-semibold mb-4 text-purple-700">
        7-Day Severity Forecast
      </h2>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={forecast}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="dayIndex" label={{ value: "Days Ahead", position: "insideBottom", offset: -5 }} />
          <YAxis domain={[0, 10]} />
          <Tooltip />

          <Area
            dataKey="upperBound"
            stroke="none"
            fillOpacity={0.2}
          />
          <Area
            dataKey="lowerBound"
            stroke="none"
            fillOpacity={0.2}
          />

          <Line
            type="monotone"
            dataKey="predictedSeverity"
            strokeWidth={3}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}