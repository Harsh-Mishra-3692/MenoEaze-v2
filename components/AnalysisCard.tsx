"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

import { runPrediction } from "@/lib/mlClient";

/* ================= TYPES ================= */

interface Props {
  userId: string;
}

type ChartPoint = {
  day: number;
  value: number;
};

/* ================= COMPONENT ================= */

export default function AnalysisCard({ userId }: Props) {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{
    severity: number;
    confidence: number | null;
    trend: number[];
    trendSummary?: string;
    answer: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mounted = useRef<boolean>(true);
  const busy = useRef<boolean>(false);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  /* ================= SAFE SEQUENCE ================= */

  const buildSafeSequence = (): number[][] => {
    const base: number[] = [5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0];
    return Array.from({ length: 5 }, () => [...base]);
  };

  /* ================= RUN ================= */

  const handleRun = async () => {
    if (busy.current) return;
    busy.current = true;

    try {
      if (mounted.current) {
        setLoading(true);
        setError(null);
      }

      // Run with actual user ID instead of safe dummy sequence if possible
      // Assuming runPrediction can take userId
      const res = await runPrediction(userId || "demo_user");

      if (!mounted.current) return;

      setResult({
        severity: safe01(res.severity, 0.5),
        confidence: res.confidence, // null allowed
        trend: Array.isArray(res.raw && (res.raw as any).trend) ? (res.raw as any).trend : [res.severity],
        trendSummary: (res.raw as any)?.trend_summary || res.trend_summary,
        answer: typeof res.answer === "string" ? res.answer : "",
      });
    } catch (e: unknown) {
      if (!mounted.current) return;

      if (e instanceof Error) setError(e.message);
      else setError("Analysis failed");
    } finally {
      busy.current = false;
      if (mounted.current) setLoading(false);
    }
  };

  /* ================= CHART ================= */

  const rawTrend = result?.trend || [];
  const chartData: ChartPoint[] = rawTrend.length > 0
    ? rawTrend.map((v, i) => ({ day: i + 1, value: safe10(v * 10) }))
    : [{ day: 1, value: safe10((result?.severity ?? 0.5) * 10) }];

  /* ================= UI ================= */

  return (
    <div className="space-y-4">
      {/* HEADER */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">Health Analysis</h3>

        <button
          onClick={handleRun}
          disabled={loading}
          className="bg-teal-600 text-white px-4 py-2 rounded-lg text-sm disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Run Analysis"}
        </button>
      </div>

      {/* ERROR */}
      {error && (
        <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm">
          {error}
        </div>
      )}

      {/* EMPTY */}
      {!result && !loading && !error && (
        <div className="text-center text-gray-400 py-6 border rounded-lg">
          Run analysis to see results
        </div>
      )}

      {/* RESULTS */}
      {result && (
        <div className="space-y-4">
          {/* METRICS */}
          <div className="grid grid-cols-2 gap-4">
            <Metric
              label="Severity"
              value={`${(result.severity * 100).toFixed(0)}%`}
            />
            <Metric
              label="Confidence"
              value={result.confidence !== null ? `${(result.confidence * 100).toFixed(0)}%` : "Low confidence (insufficient data)"}
            />
          </div>

          {/* CHART */}
          <div className="bg-white p-4 rounded-xl border">
            <p className="text-sm mb-2">Trend Analysis</p>

            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis domain={[0, 10]} />
                  <Tooltip /> {/* ← safest fix */}
                  <Line
                    type="monotone"
                    dataKey="value"
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {result.trendSummary && (
              <p className="text-sm text-gray-600 mt-4 border-t pt-3">
                {result.trendSummary}
              </p>
            )}
          </div>

          {/* ANSWER */}
          {result.answer && (
            <div className="bg-teal-50 p-4 rounded-lg text-sm">
              {result.answer}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ================= SUB ================= */

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 p-3 rounded-lg">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-semibold">{value}</p>
    </div>
  );
}

/* ================= HELPERS ================= */

function safe01(v: unknown, d: number): number {
  const n = Number(v);
  if (Number.isNaN(n)) return d;
  return Math.max(0, Math.min(1, n));
}

function safe10(v: number): number {
  if (Number.isNaN(v)) return 5;
  return Math.max(0, Math.min(10, v));
}