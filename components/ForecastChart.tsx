"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Area,
} from "recharts";

// ---- TYPES ----
interface ForecastPoint {
  dayIndex: number;
  predictedSeverity: number;
  lowerBound: number;
  upperBound: number;
}

// ---- HELPERS ----
function clamp(value: unknown): number {
  const num = typeof value === "number" ? value : Number(value);
  if (isNaN(num)) return 0;
  return Math.max(0, Math.min(10, num));
}

function sanitizeForecast(data?: ForecastPoint[]): ForecastPoint[] {
  if (!Array.isArray(data)) return [];

  return data
    .filter((d) => d && typeof d.dayIndex === "number")
    .map((d) => ({
      dayIndex: d.dayIndex,
      predictedSeverity: clamp(d.predictedSeverity),
      lowerBound: clamp(d.lowerBound),
      upperBound: clamp(d.upperBound),
    }));
}

// ---- COMPONENT ----
export default function ForecastChart({
  forecast,
}: {
  forecast?: ForecastPoint[];
}) {
  const safeData = sanitizeForecast(forecast);

  // ---- EMPTY STATE ----
  if (!safeData.length) {
    return (
      <div className="bg-white rounded-2xl shadow-lg p-6 text-center text-gray-500">
        No forecast data available
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-lg p-6">
      <h2 className="text-lg font-semibold mb-4 text-purple-700">
        7-Day Severity Forecast
      </h2>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={safeData}>
          <CartesianGrid strokeDasharray="3 3" />

          <XAxis
            dataKey="dayIndex"
            label={{
              value: "Days Ahead",
              position: "insideBottom",
              offset: -5,
            }}
          />

          <YAxis domain={[0, 10]} />

          <Tooltip
            formatter={(value: unknown) => {
              const num =
                typeof value === "number" ? value : Number(value);
              return isNaN(num) ? "-" : num.toFixed(2);
            }}
          />

          {/* Confidence band */}
          <Area
            type="monotone"
            dataKey="upperBound"
            stroke="none"
            fillOpacity={0.1}
          />
          <Area
            type="monotone"
            dataKey="lowerBound"
            stroke="none"
            fillOpacity={0.1}
          />

          {/* Prediction line */}
          <Line
            type="monotone"
            dataKey="predictedSeverity"
            strokeWidth={3}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}