'use client'

// DISABLED: Frontend MUST NOT query Supabase directly.
// Stats will be provided by backend in a future /run action.

export default function StatsSummary({ userId }: { userId: string }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Total Logs</p>
        <h3 className="text-2xl font-bold text-white mt-1">—</h3>
      </div>
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Average Severity</p>
        <h3 className="text-2xl font-bold text-white mt-1">—</h3>
      </div>
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Last Logged</p>
        <h3 className="text-lg font-semibold text-white mt-1">—</h3>
      </div>
      <div className="bg-white/10 backdrop-blur-md p-6 rounded-2xl border border-white/10">
        <p className="text-sm text-gray-300">Logs This Week</p>
        <h3 className="text-2xl font-bold text-white mt-1">—</h3>
      </div>
    </div>
  )
}