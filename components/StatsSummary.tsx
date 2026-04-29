'use client'

import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'

type Log = {
  severity: number
  created_at: string
}

export default function StatsSummary({ userId }: { userId: string }) {
  const [logs, setLogs] = useState<Log[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchLogs = async () => {
      const { data, error } = await supabase
        .from('symptom_logs')
        .select('severity, created_at')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })

      if (!error && data) setLogs(data)
      setLoading(false)
    }

    fetchLogs()
  }, [userId])

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-white/10 backdrop-blur-md p-6 rounded-xl animate-pulse h-24" />
        ))}
      </div>
    )
  }

  const totalLogs = logs.length

  const avgSeverity =
    logs.length > 0
      ? logs.reduce((sum, l) => sum + l.severity, 0) / logs.length
      : 0

  const lastLogged =
    logs.length > 0
      ? new Date(logs[0].created_at).toLocaleDateString()
      : '—'

  const weeklyCount = logs.filter(l => {
    const d = new Date(l.created_at)
    const sevenDaysAgo = new Date()
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7)
    return d >= sevenDaysAgo
  }).length

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">

      {/* Total Logs */}
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Total Logs</p>
        <h3 className="text-2xl font-bold text-white mt-1">
          {totalLogs}
        </h3>
      </div>

      {/* Average Severity */}
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Average Severity</p>
        <h3 className="text-2xl font-bold text-white mt-1">
          {avgSeverity.toFixed(1)} / 10
        </h3>
      </div>

      {/* Last Logged */}
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Last Logged</p>
        <h3 className="text-lg font-semibold text-white mt-1">
          {lastLogged}
        </h3>
      </div>

      {/* Weekly Activity */}
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Logs This Week</p>
        <h3 className="text-2xl font-bold text-white mt-1">
          {weeklyCount}
        </h3>
      </div>

    </div>
  )
}