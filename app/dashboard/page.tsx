'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { supabase } from '@/lib/supabase'
import SymptomForm from '@/components/SymptomForm'
import AnalysisCard from '@/components/AnalysisCard'
import {
    Activity,
    TrendingUp,
    Calendar,
    BarChart3,
    Heart,
    Sun,
    Sparkles
} from 'lucide-react'

const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    visible: (i: number) => ({
        opacity: 1,
        y: 0,
        transition: { delay: i * 0.1, duration: 0.5, ease: 'easeOut' as const }
    })
}

export default function DashboardPage() {
    const [userId, setUserId] = useState<string | null>(null)
    const [userName, setUserName] = useState('')
    const [showUsernameModal, setShowUsernameModal] = useState(false)
    const [usernameInput, setUsernameInput] = useState('')
    const [savingUsername, setSavingUsername] = useState(false)
    const [stats, setStats] = useState({
        total: 0,
        avgSeverity: 0,
        lastLogged: '',
        weekly: 0
    })

    useEffect(() => {
        supabase.auth.getUser().then(async ({ data }) => {
            if (data.user) {
                setUserId(data.user.id)

                // Fetch username from profiles
                const { data: profile } = await supabase
                    .from('profiles')
                    .select('username')
                    .eq('id', data.user.id)
                    .single()

                if (profile && profile.username && profile.username.trim() !== '') {
                    setUserName(profile.username)
                } else {
                    // No username set — show prompt
                    setShowUsernameModal(true)
                }
            }
        })
        fetchStats()
    }, [])

    const saveUsername = async () => {
        if (!usernameInput.trim() || !userId) return
        setSavingUsername(true)

        await supabase.from('profiles').upsert({
            id: userId,
            username: usernameInput.trim()
        })

        setUserName(usernameInput.trim())
        setShowUsernameModal(false)
        setSavingUsername(false)
    }

    async function fetchStats() {
        const { data } = await supabase
            .from('symptom_logs')
            .select('severity, created_at')
            .eq('is_deleted', false)

        if (!data) return

        const total = data.length
        const avgSeverity =
            total > 0
                ? parseFloat(
                    (data.reduce((a, b) => a + b.severity, 0) / total).toFixed(1)
                )
                : 0

        const lastLogged =
            total > 0
                ? new Date(
                    Math.max(...data.map(d => new Date(d.created_at).getTime()))
                ).toLocaleDateString('en-US', {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric'
                })
                : '—'

        const weekAgo = new Date()
        weekAgo.setDate(weekAgo.getDate() - 7)
        const weekly = data.filter(d => new Date(d.created_at) > weekAgo).length

        setStats({ total, avgSeverity, lastLogged, weekly })
    }

    const greeting = () => {
        const hour = new Date().getHours()
        if (hour < 12) return 'Good morning'
        if (hour < 17) return 'Good afternoon'
        return 'Good evening'
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-rose-50/80 via-purple-50/50 to-pink-50/60 relative overflow-hidden">
            {/* Ambient blobs */}
            <div className="fixed inset-0 pointer-events-none overflow-hidden">
                <div className="absolute top-[-15%] right-[-5%] w-[500px] h-[500px] rounded-full bg-purple-200/30 blur-[100px] animate-[drift_22s_ease-in-out_infinite_alternate]" />
                <div className="absolute bottom-[-10%] left-[-5%] w-[400px] h-[400px] rounded-full bg-pink-200/30 blur-[100px] animate-[drift_28s_ease-in-out_infinite_alternate-reverse]" />
                <div className="absolute top-[50%] left-[40%] w-[350px] h-[350px] rounded-full bg-rose-200/20 blur-[100px] animate-[drift_30s_ease-in-out_infinite_alternate]" />
            </div>

            {/* Username Prompt Modal */}
            {showUsernameModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 backdrop-blur-sm">
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="bg-white rounded-2xl shadow-2xl p-8 max-w-sm w-full mx-4"
                    >
                        <div className="text-center mb-6">
                            <div className="w-14 h-14 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center mx-auto mb-4">
                                <Heart size={24} className="text-white" />
                            </div>
                            <h2 className="text-xl font-semibold text-gray-800">Welcome to MenoEaze!</h2>
                            <p className="text-sm text-gray-500 mt-2">What should we call you?</p>
                        </div>

                        <input
                            value={usernameInput}
                            onChange={(e) => setUsernameInput(e.target.value)}
                            placeholder="Enter your name"
                            autoFocus
                            className="w-full border border-gray-300 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-purple-500 outline-none transition mb-4"
                            onKeyDown={(e) => e.key === 'Enter' && saveUsername()}
                        />

                        <button
                            onClick={saveUsername}
                            disabled={savingUsername || !usernameInput.trim()}
                            className="w-full bg-gradient-to-r from-purple-600 to-pink-500 text-white py-3 rounded-xl font-semibold shadow-sm hover:shadow-md transition disabled:opacity-50"
                        >
                            {savingUsername ? 'Saving...' : 'Continue'}
                        </button>
                    </motion.div>
                </div>
            )}

            <div className="relative z-10 max-w-6xl mx-auto px-6 py-10">

                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="mb-10"
                >
                    <div className="flex items-center gap-2 mb-1">
                        <Sun size={18} className="text-amber-400" />
                        <span className="text-sm text-gray-500 font-medium">
                            {greeting()}, {userName || 'there'}
                        </span>
                    </div>
                    <h1 className="text-3xl font-semibold text-gray-800 tracking-tight">
                        Your Wellness Dashboard
                    </h1>
                    <p className="text-gray-500 mt-1 text-sm">
                        Track your journey. Understand your body. Embrace the change.
                    </p>
                </motion.div>

                {/* Stat Cards */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
                    <StatCard
                        i={0}
                        icon={<Activity size={18} />}
                        label="Total Logs"
                        value={stats.total.toString()}
                        accent="text-purple-600"
                        bg="bg-purple-50"
                    />
                    <StatCard
                        i={1}
                        icon={<TrendingUp size={18} />}
                        label="Avg Severity"
                        value={`${stats.avgSeverity}/10`}
                        accent="text-rose-600"
                        bg="bg-rose-50"
                    />
                    <StatCard
                        i={2}
                        icon={<Calendar size={18} />}
                        label="Last Logged"
                        value={stats.lastLogged}
                        accent="text-indigo-600"
                        bg="bg-indigo-50"
                    />
                    <StatCard
                        i={3}
                        icon={<BarChart3 size={18} />}
                        label="This Week"
                        value={stats.weekly.toString()}
                        accent="text-emerald-600"
                        bg="bg-emerald-50"
                    />
                </div>

                {/* Main Content */}
                <div className="grid lg:grid-cols-5 gap-6">

                    {/* Left — Symptom Form */}
                    <motion.div
                        custom={4}
                        variants={fadeUp}
                        initial="hidden"
                        animate="visible"
                        className="lg:col-span-2"
                    >
                        <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 border border-purple-100/50 shadow-sm hover:shadow-md transition-shadow duration-300">
                            <div className="flex items-center gap-2.5 mb-5">
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                                    <Heart size={15} className="text-white" />
                                </div>
                                <h2 className="text-lg font-semibold text-gray-800">
                                    How are you feeling?
                                </h2>
                            </div>
                            <SymptomForm onSuccess={() => fetchStats()} />
                        </div>
                    </motion.div>

                    {/* Right — Analytics */}
                    <motion.div
                        custom={5}
                        variants={fadeUp}
                        initial="hidden"
                        animate="visible"
                        className="lg:col-span-3"
                    >
                        <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 border border-purple-100/50 shadow-sm hover:shadow-md transition-shadow duration-300">
                            <div className="flex items-center gap-2.5 mb-5">
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                                    <Sparkles size={15} className="text-white" />
                                </div>
                                <h2 className="text-lg font-semibold text-gray-800">
                                    Your Insights
                                </h2>
                            </div>
                            {userId ? (
                                <AnalysisCard userId={userId} />
                            ) : (
                                <div className="text-gray-400 text-sm py-12 text-center">
                                    Loading your analytics…
                                </div>
                            )}
                        </div>
                    </motion.div>
                </div>

                {/* Wellness tip */}
                <motion.div
                    custom={6}
                    variants={fadeUp}
                    initial="hidden"
                    animate="visible"
                    className="mt-8 bg-gradient-to-r from-purple-50/80 to-pink-50/80 backdrop-blur-sm rounded-2xl p-5 border border-purple-100/40 flex items-start gap-4"
                >
                    <div className="w-10 h-10 rounded-xl bg-purple-100 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <Heart size={18} className="text-purple-500" />
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-gray-700 mb-1">
                            Daily Wellness Reminder
                        </h3>
                        <p className="text-sm text-gray-500 leading-relaxed">
                            Consistent tracking helps you understand your body&apos;s patterns. Even on days
                            when symptoms feel mild, logging helps build a clearer picture of your journey.
                            You&apos;re doing great by being here. 💜
                        </p>
                    </div>
                </motion.div>
            </div>
        </div>
    )
}

function StatCard({
    i,
    icon,
    label,
    value,
    accent,
    bg
}: {
    i: number
    icon: React.ReactNode
    label: string
    value: string
    accent: string
    bg: string
}) {
    return (
        <motion.div
            custom={i}
            variants={fadeUp}
            initial="hidden"
            animate="visible"
            className="bg-white/70 backdrop-blur-sm rounded-2xl p-5 border border-purple-100/40 shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-300 group"
        >
            <div className={`w-9 h-9 rounded-xl ${bg} flex items-center justify-center ${accent} mb-3 group-hover:scale-110 transition-transform duration-300`}>
                {icon}
            </div>
            <p className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-1">
                {label}
            </p>
            <p className="text-2xl font-semibold text-gray-800 tracking-tight">
                {value}
            </p>
        </motion.div>
    )
}