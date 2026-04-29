'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  ReferenceDot
} from 'recharts'

type ChartPoint = {
  date: string
  severity: number
  mood: number
  smoothed: number
}

// interface ChartProps {
//   data: ChartPoint[]
//   forecast?: number
//   view: 'weekly' | 'monthly'
// }
interface ChartProps {
  data: {
    date: string
    severity: number
    mood: number
    smoothed: number
  }[]
  forecast?: number
  view: 'weekly' | 'monthly'
}

export default function Chart({
  data,
  forecast,
  view
}: ChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white p-6 rounded-2xl shadow-lg">
        No data available for visualization.
      </div>
    )
  }

  const lastPoint = data[data.length - 1]

  return (
    <div className="bg-white p-6 rounded-2xl shadow-lg space-y-6">

      <div>
        <h3 className="text-xl font-bold text-gray-900">
          📊 Symptom Trend ({view})
        </h3>
        <p className="text-sm text-gray-500">
          Dual-axis visualization of severity and mood patterns.
        </p>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" hide />
            <YAxis
              yAxisId="left"
              domain={[0, 10]}
              label={{ value: 'Severity', angle: -90, position: 'insideLeft' }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0, 10]}
              label={{ value: 'Mood', angle: 90, position: 'insideRight' }}
            />
            <Tooltip />
            <Legend />

            {/* Severity */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="severity"
              stroke="#9333ea"
              strokeWidth={2}
              dot={{ r: 3 }}
              name="Severity"
            />

            {/* Mood */}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="mood"
              stroke="#10b981"
              strokeWidth={2}
              dot={{ r: 3 }}
              name="Mood"
            />

            {/* Moving Average */}
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="smoothed"
              stroke="#ec4899"
              strokeWidth={2}
              dot={false}
              name="7-Day Moving Avg"
            />

            {/* Forecast Dot */}
            {forecast !== undefined && (
              <ReferenceDot
                x={lastPoint.date}
                y={forecast}
                yAxisId="left"
                r={6}
                fill="#2563eb"
                stroke="white"
                strokeWidth={2}
                label="Forecast"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

    </div>
  )
}