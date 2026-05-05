import { NextResponse } from "next/server"
import { createClient } from "@supabase/supabase-js"

/* ================= CONFIG ================= */

function getSupabaseAdmin() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
  if (!url || !key) return null
  return createClient(url, key)
}

/* ================= HANDLER ================= */

export async function POST(req: Request) {
  try {
    const body = await req.json().catch(() => ({}));
    const userId = body.userId || "demo_user";

    // First try: backend /run stats action
    const ML_API_URL = (process.env.NEXT_PUBLIC_ML_API_URL || "http://localhost:8000").replace(/\/+$/, "");

    let backendStats: any = null

    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 10000)

      const res = await fetch(`${ML_API_URL}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: "stats",
          user_id: userId
        }),
        signal: controller.signal
      });

      clearTimeout(timeout)

      if (res.ok) {
        const data = await res.json();
        backendStats = data.stats || null;
      }
    } catch {
      // Backend unavailable — fall through to Supabase direct
    }

    // If backend returned stats, use them
    if (backendStats) {
      return NextResponse.json({
        total_logs: backendStats.total_logs || 0,
        avg_severity: typeof backendStats.avg_severity === "number" 
          ? parseFloat(backendStats.avg_severity.toFixed(2)) 
          : 0,
        last_logged: backendStats.last_log_at || '—',
        weekly_count: backendStats.recent_count || 0,
        trend: backendStats.history || []
      });
    }

    // Fallback: compute stats directly from Supabase
    const supabase = getSupabaseAdmin()
    if (!supabase) {
      return NextResponse.json({
        total_logs: 0,
        avg_severity: 0,
        last_logged: '—',
        weekly_count: 0,
        trend: []
      });
    }

    // Total logs
    const { count: totalLogs } = await supabase
      .from("symptom_logs")
      .select("*", { count: "exact", head: true })
      .eq("user_id", userId);

    // Recent logs for avg severity and last logged
    const { data: recentLogs } = await supabase
      .from("symptom_logs")
      .select("feature_vector, created_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false })
      .limit(50);

    let avgSeverity = 0;
    let lastLogged = '—';
    let weeklyCount = 0;

    if (recentLogs && recentLogs.length > 0) {
      // Average severity — use column if available, fallback to feature_vector
      const severities = recentLogs
        .map((l: any) => {
          const fv = l.feature_vector;
          if (Array.isArray(fv) && fv.length === 11) {
            return fv.reduce((a: number, b: number) => a + Number(b || 0), 0) / (fv.length * 10);
          }
          return null;
        })
        .filter((n: any): n is number => n !== null && !isNaN(n));
      
      if (severities.length > 0) {
        avgSeverity = parseFloat(
          (severities.reduce((a: number, b: number) => a + b, 0) / severities.length * 10).toFixed(1)
        );
      }

      // Last logged
      lastLogged = recentLogs[0]?.created_at || '—';

      // Weekly count
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
      weeklyCount = recentLogs.filter((l: any) => {
        try {
          return new Date(l.created_at) >= oneWeekAgo;
        } catch {
          return false;
        }
      }).length;
    }

    // Build trend from last 14 logs — must match Chart contract: { date, severity, mood, smoothed }
    const rawTrend = (recentLogs || [])
      .slice(0, 14)
      .reverse()
      .map((l: any) => {
        let sev = 0;
        let mood = 5; // default neutral
        const fv = l.feature_vector;
        if (Array.isArray(fv) && fv.length === 11) {
          sev = fv.reduce((a: number, b: number) => a + Number(b || 0), 0) / (fv.length * 10);
          mood = Number(fv[4] || 5); // mood is index 4 in 11-feature schema
        }
        return {
          severity: parseFloat((sev * 10).toFixed(1)),
          mood: Math.max(0, Math.min(10, mood)),
          date: l.created_at || ''
        };
      });

    // Compute smoothed (3-point moving average)
    const trend = rawTrend.map((point: any, i: number) => {
      const window = rawTrend.slice(Math.max(0, i - 1), i + 2);
      const avg = window.reduce((sum: number, p: any) => sum + p.severity, 0) / window.length;
      return {
        ...point,
        smoothed: parseFloat(avg.toFixed(1))
      };
    });

    return NextResponse.json({
      total_logs: totalLogs || 0,
      avg_severity: avgSeverity,
      last_logged: lastLogged,
      weekly_count: weeklyCount,
      trend
    });

  } catch (err) {
    console.error("[insight-summary] Error:", err);
    return NextResponse.json({
      total_logs: 0,
      avg_severity: 0,
      last_logged: '—',
      weekly_count: 0,
      trend: []
    });
  }
}
