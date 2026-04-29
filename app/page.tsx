'use client'

import {
  ShieldCheck,
  Users,
  Lock,
  BarChart3,
  MessageCircle,
  Clock,
  Hospital
} from 'lucide-react'
import Link from 'next/link'
import Image from 'next/image'
import { useEffect, useRef, useState, ReactNode } from 'react'
import { motion } from 'framer-motion'

/* ================= HERO TEXT ANIMATION ================= */

const sentence = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.05
    }
  }
}

const letter = {
  hidden: { opacity: 0, y: 40 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1] as const
    }
  }
}

/* ================= MAIN PAGE ================= */

export default function HomePage() {
  const heroText = "MenoEaze"

  return (
    <div className="overflow-hidden">

      {/* ================= HERO ================= */}
      <section className="relative bg-gradient-to-r from-purple-600 via-pink-500 to-rose-500 text-white text-center pt-32 pb-40 px-6 overflow-hidden">

        <AnimatedBackground />

        <div className="max-w-4xl mx-auto relative z-10">

          {/* Animated Logo */}
          <div className="flex justify-center mb-8">
            <div className="relative">
              {/* Spinning halo ring */}
              <div className="absolute -inset-4 rounded-full animate-halo-spin" style={{ background: 'conic-gradient(from 0deg, transparent, rgba(168,85,247,0.3), rgba(236,72,153,0.3), transparent)' }} />
              {/* Glow container */}
              <div className="relative w-28 h-28 md:w-32 md:h-32 rounded-full animate-glow-pulse bg-white/20 backdrop-blur-md border border-white/30 flex items-center justify-center">
                <div className="animate-gentle-float">
                  <Image
                    src="/image1.png"
                    alt="MenoEaze — Your Menopause Wellness Companion"
                    width={100}
                    height={100}
                    className="rounded-full drop-shadow-lg w-20 h-20 md:w-24 md:h-24 object-contain"
                    priority
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Animated Title */}
          <motion.h1
            variants={sentence}
            initial="hidden"
            animate="visible"
            className="text-5xl md:text-6xl font-bold mb-6"
          >
            {heroText.split('').map((char, i) => (
              <motion.span key={i} variants={letter}>
                {char}
              </motion.span>
            ))}
          </motion.h1>

          <p className="text-xl md:text-2xl font-medium mb-4">
            Your trusted companion for navigating menopause with confidence
          </p>

          <p className="text-lg opacity-90 mb-6">
            Track symptoms, gain insights, and connect with a supportive community
          </p>

          <div className="flex justify-center items-center gap-3 text-sm mb-10 opacity-90">
            <Lock className="w-4 h-4" />
            HIPAA Compliant • Privacy First • Trusted by 1,000+ Women
          </div>

          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              href="/auth"
              className="bg-white text-purple-600 px-8 py-3 rounded-xl font-semibold shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300"
            >
              Get Started Free
            </Link>

            <Link
              href="/community"
              className="border border-white px-8 py-3 rounded-xl font-semibold hover:bg-white/10 transition"
            >
              Explore Community
            </Link>
          </div>
        </div>

        {/* Wave */}
        <div className="absolute bottom-0 left-0 w-full">
          <svg viewBox="0 0 1440 120" className="w-full">
            <path
              fill="#f9fafb"
              d="M0,64L80,64C160,64,320,64,480,80C640,96,800,128,960,122.7C1120,117,1280,75,1360,53.3L1440,32V120H0Z"
            />
          </svg>
        </div>
      </section>

      {/* ================= FEATURES ================= */}
      <section className="bg-gray-50 py-28 px-6">
        <div className="max-w-7xl mx-auto text-center">

          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
            Your Complete Wellness Companion
          </h2>

          <p className="text-gray-600 text-lg max-w-2xl mx-auto mb-20">
            Everything you need to navigate your menopause journey with confidence and support
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-12">
            <Reveal>
              <PremiumFeatureCard
                title="Symptom Tracking"
                description="Comprehensive daily tracking with detailed insights and pattern recognition."
                icon={<BarChart3 className="w-6 h-6 text-white" />}
                gradient="from-purple-500 to-pink-500"
                bg="bg-purple-50"
              />
            </Reveal>

            <Reveal>
              <PremiumFeatureCard
                title="Personalized AI Chat"
                description="Instant AI-powered guidance available 24/7."
                icon={<MessageCircle className="w-6 h-6 text-white" />}
                gradient="from-pink-500 to-rose-500"
                bg="bg-rose-50"
              />
            </Reveal>

            <Reveal>
              <PremiumFeatureCard
                title="Community Support"
                description="Connect with women sharing similar experiences."
                icon={<Users className="w-6 h-6 text-white" />}
                gradient="from-indigo-500 to-purple-500"
                bg="bg-indigo-50"
              />
            </Reveal>

            <Reveal>
              <PremiumFeatureCard
                title="Privacy First"
                description="Encrypted, HIPAA compliant and secure."
                icon={<ShieldCheck className="w-6 h-6 text-white" />}
                gradient="from-green-500 to-emerald-500"
                bg="bg-green-50"
              />
            </Reveal>

            <Reveal>
              <PremiumFeatureCard
                title="24/7 Access"
                description="Anytime access to dashboard and assistant."
                icon={<Clock className="w-6 h-6 text-white" />}
                gradient="from-blue-500 to-indigo-500"
                bg="bg-blue-50"
              />
            </Reveal>

            <Reveal>
              <PremiumFeatureCard
                title="Healthcare Integration"
                description="Share insights with your provider seamlessly."
                icon={<Hospital className="w-6 h-6 text-white" />}
                gradient="from-fuchsia-500 to-purple-500"
                bg="bg-fuchsia-50"
              />
            </Reveal>
          </div>
        </div>
      </section>

      {/* ================= TRUST & CREDIBILITY ================= */}
      <section className="relative bg-white py-24 px-6 overflow-hidden">

        {/* Decorative floating lotus petals */}
        <div className="absolute inset-0 pointer-events-none overflow-hidden">
          <div className="absolute top-[10%] left-[8%] w-16 h-16 animate-petal-1">
            <svg viewBox="0 0 60 60" fill="none">
              <ellipse cx="30" cy="30" rx="22" ry="12" transform="rotate(-30 30 30)" fill="url(#p1)" opacity="0.5" />
              <defs><linearGradient id="p1" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#c084fc" /><stop offset="1" stopColor="#f9a8d4" /></linearGradient></defs>
            </svg>
          </div>
          <div className="absolute top-[60%] right-[5%] w-20 h-20 animate-petal-2">
            <svg viewBox="0 0 60 60" fill="none">
              <ellipse cx="30" cy="30" rx="20" ry="10" transform="rotate(25 30 30)" fill="url(#p2)" opacity="0.4" />
              <defs><linearGradient id="p2" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#f472b6" /><stop offset="1" stopColor="#a78bfa" /></linearGradient></defs>
            </svg>
          </div>
          <div className="absolute bottom-[15%] left-[45%] w-14 h-14 animate-petal-3">
            <svg viewBox="0 0 60 60" fill="none">
              <ellipse cx="30" cy="30" rx="18" ry="9" transform="rotate(50 30 30)" fill="url(#p3)" opacity="0.35" />
              <defs><linearGradient id="p3" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#e879f9" /><stop offset="1" stopColor="#fb7185" /></linearGradient></defs>
            </svg>
          </div>
          <div className="absolute top-[35%] right-[30%] w-12 h-12 animate-petal-1" style={{ animationDelay: '5s' }}>
            <svg viewBox="0 0 60 60" fill="none">
              <ellipse cx="30" cy="30" rx="16" ry="8" transform="rotate(-15 30 30)" fill="url(#p4)" opacity="0.3" />
              <defs><linearGradient id="p4" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#c084fc" /><stop offset="1" stopColor="#fda4af" /></linearGradient></defs>
            </svg>
          </div>
        </div>

        {/* Background watermark logo */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <Image
            src="/image1.png"
            alt=""
            width={400}
            height={400}
            className="opacity-[0.04] select-none"
            aria-hidden="true"
          />
        </div>

        <div className="max-w-5xl mx-auto relative z-10">

          {/* Section header */}
          <Reveal>
            <div className="text-center mb-16">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-purple-50 border border-purple-100 text-purple-600 text-xs font-semibold uppercase tracking-widest mb-5">
                <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.286 3.957a1 1 0 00.95.69h4.162c.969 0 1.371 1.24.588 1.81l-3.37 2.448a1 1 0 00-.364 1.118l1.287 3.957c.3.921-.755 1.688-1.54 1.118l-3.37-2.448a1 1 0 00-1.175 0l-3.37 2.448c-.784.57-1.838-.197-1.54-1.118l1.287-3.957a1 1 0 00-.364-1.118L2.063 9.384c-.783-.57-.38-1.81.588-1.81h4.162a1 1 0 00.95-.69l1.286-3.957z" /></svg>
                Why Women Trust Us
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
                Built With Care, Backed By Science
              </h2>
              <p className="text-gray-500 text-base max-w-xl mx-auto">
                Every feature is designed with empathy and informed by medical research to support your unique journey.
              </p>
            </div>
          </Reveal>

          {/* Trust pillar cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

            <Reveal>
              <TrustCard
                icon={
                  <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25" />
                  </svg>
                }
                title="Medically Informed"
                description="Our insights are grounded in peer-reviewed menopause research and clinical guidelines — not guesswork."
                gradient="from-purple-500 to-indigo-500"
              />
            </Reveal>

            <Reveal>
              <TrustCard
                icon={
                  <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
                  </svg>
                }
                title="Privacy First"
                description="HIPAA-compliant encryption protects every detail. Your health data stays yours — always private, always secure."
                gradient="from-pink-500 to-rose-500"
              />
            </Reveal>

            <Reveal>
              <TrustCard
                icon={
                  <svg className="w-7 h-7" fill="none" stroke="currentColor" strokeWidth={1.5} viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z" />
                  </svg>
                }
                title="Women-Led Community"
                description="Join thousands of women sharing experiences, supporting each other, and navigating menopause together."
                gradient="from-fuchsia-500 to-purple-500"
              />
            </Reveal>
          </div>

          {/* Elegant divider */}
          <div className="flex items-center justify-center mt-16 gap-3">
            <div className="h-px w-16 bg-gradient-to-r from-transparent to-purple-200" />
            <svg className="w-5 h-5 text-purple-300" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
            </svg>
            <div className="h-px w-16 bg-gradient-to-l from-transparent to-pink-200" />
          </div>

        </div>
      </section>

      {/* ================= CTA ================= */}
      <section className="bg-gradient-to-r from-purple-600 to-pink-500 text-white text-center py-28 px-6">
        <h2 className="text-4xl font-bold mb-6">
          Ready to take control?
        </h2>

        <p className="mb-8 text-lg opacity-90">
          Join thousands managing their menopause journey with confidence.
        </p>

        <Link
          href="/auth"
          className="bg-white text-purple-600 px-10 py-4 rounded-xl font-semibold shadow-lg hover:shadow-2xl hover:-translate-y-1 transition-all duration-300"
        >
          Start Your Journey Today
        </Link>
      </section>
    </div>
  )
}

/* ================= ANIMATED BACKGROUND ================= */

function AnimatedBackground() {
  return (
    <div className="absolute inset-0 -z-10 overflow-hidden">
      <div className="absolute w-[600px] h-[600px] bg-purple-400/40 rounded-full blur-[120px] animate-float1 top-[-150px] left-[-150px]" />
      <div className="absolute w-[600px] h-[600px] bg-pink-400/40 rounded-full blur-[120px] animate-float2 bottom-[-150px] right-[-150px]" />
      <div className="absolute w-[600px] h-[600px] bg-rose-400/30 rounded-full blur-[120px] animate-float3 top-[40%] left-[40%]" />
    </div>
  )
}

/* Add these keyframes to globals.css:

@keyframes float1 {
  from { transform: translate(0,0); }
  to { transform: translate(80px,-60px); }
}
@keyframes float2 {
  from { transform: translate(0,0); }
  to { transform: translate(-60px,60px); }
}
@keyframes float3 {
  from { transform: translate(0,0); }
  to { transform: translate(40px,-40px); }
}

.animate-float1 { animation: float1 20s ease-in-out infinite alternate; }
.animate-float2 { animation: float2 25s ease-in-out infinite alternate; }
.animate-float3 { animation: float3 30s ease-in-out infinite alternate; }

*/

/* ================= FEATURE CARD ================= */

interface FeatureProps {
  title: string
  description: string
  icon: ReactNode
  gradient: string
  bg: string
}

function PremiumFeatureCard({
  title,
  description,
  icon,
  gradient,
  bg
}: FeatureProps) {
  return (
    <div className="group relative">

      <div className={`absolute -inset-0.5 rounded-2xl bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-100 blur transition duration-500`} />

      <div className={`${bg} relative rounded-2xl p-8 shadow-sm group-hover:shadow-2xl transition-all duration-500 group-hover:-translate-y-3`}>
        <div className={`w-12 h-12 rounded-xl flex items-center justify-center bg-gradient-to-br ${gradient} mb-6`}>
          {icon}
        </div>

        <h3 className="text-xl font-semibold text-gray-900 mb-3">
          {title}
        </h3>

        <p className="text-gray-600 leading-relaxed">
          {description}
        </p>
      </div>
    </div>
  )
}

/* ================= TRUST CARD ================= */

interface TrustCardProps {
  icon: ReactNode
  title: string
  description: string
  gradient: string
}

function TrustCard({ icon, title, description, gradient }: TrustCardProps) {
  return (
    <div className="group relative">
      {/* Hover glow */}
      <div className={`absolute -inset-0.5 rounded-2xl bg-gradient-to-br ${gradient} opacity-0 group-hover:opacity-100 blur transition duration-500`} />

      <div className="relative bg-white rounded-2xl p-8 shadow-sm border border-gray-100 group-hover:shadow-xl group-hover:-translate-y-2 transition-all duration-500 text-center">
        {/* Icon container */}
        <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${gradient} flex items-center justify-center mx-auto mb-5 text-white shadow-lg group-hover:scale-110 transition-transform duration-300`}>
          {icon}
        </div>

        <h3 className="text-lg font-semibold text-gray-900 mb-2">{title}</h3>
        <p className="text-gray-500 text-sm leading-relaxed">{description}</p>

        {/* Shimmer accent line */}
        <div className={`mt-5 h-0.5 w-12 mx-auto rounded-full bg-gradient-to-r ${gradient} opacity-40 group-hover:w-20 group-hover:opacity-80 transition-all duration-500`} />
      </div>
    </div>
  )
}

/* ================= SCROLL REVEAL ================= */

function Reveal({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          observer.unobserve(entry.target)
        }
      },
      { threshold: 0.15 }
    )

    if (ref.current) observer.observe(ref.current)

    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${visible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
        }`}
    >
      {children}
    </div>
  )
}