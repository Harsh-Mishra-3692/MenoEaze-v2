'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'
import { supabase } from '@/lib/supabase'

export default function Footer() {
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null)

  useEffect(() => {
    const checkSession = async () => {
      const {
        data: { session }
      } = await supabase.auth.getSession()
      setIsLoggedIn(!!session)
    }

    checkSession()

    const {
      data: { subscription }
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsLoggedIn(!!session)
    })

    return () => subscription.unsubscribe()
  }, [])

  if (isLoggedIn === null) return null

  return (
    <footer
      className="bg-gray-950 text-gray-300 pt-14 pb-10 mt-16 relative"
      role="contentinfo"
    >
      {/* Gradient Accent Line */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-600 via-pink-500 to-purple-600" />

      <div className="max-w-7xl mx-auto px-6 pt-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-10 mb-10">
          
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <span className="text-2xl">❤️</span>
              <h3 className="text-xl font-bold text-white">MenoEaze</h3>
            </div>
            <p className="text-sm text-gray-400 leading-relaxed">
              AI-powered menopause tracking, symptom analytics, and personalized evidence-based guidance.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="text-white font-semibold mb-4 tracking-wide">
              Quick Links
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/" className="hover:text-white transition">
                  Home
                </Link>
              </li>

              {isLoggedIn && (
                <>
                  <li>
                    <Link href="/dashboard" className="hover:text-white transition">
                      Dashboard
                    </Link>
                  </li>
                  <li>
                    <Link href="/assistant" className="hover:text-white transition">
                      AI Assistant
                    </Link>
                  </li>
                </>
              )}

              <li>
                <Link href="/community" className="hover:text-white transition">
                  Community
                </Link>
              </li>

              {!isLoggedIn && (
                <li>
                  <Link href="/login" className="hover:text-white transition">
                    Login
                  </Link>
                </li>
              )}
            </ul>
          </div>

          {/* Support */}
          <div>
            <h4 className="text-white font-semibold mb-4 tracking-wide">
              Support
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <Link href="/privacy" className="hover:text-white transition">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link href="/terms" className="hover:text-white transition">
                  Terms
                </Link>
              </li>
              <li className="text-gray-500 text-xs leading-relaxed">
                For educational purposes only. Not a substitute for professional medical advice.
              </li>
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h4 className="text-white font-semibold mb-4 tracking-wide">
              Contact
            </h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a
                  href="mailto:support@menoeaze.com"
                  className="hover:text-white transition"
                  aria-label="Email support"
                >
                  support@menoeaze.com
                </a>
              </li>
              <li className="text-gray-400">
                AI Assistant available 24/7
              </li>
              <li>
                <button
                  onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                  className="text-xs text-gray-500 hover:text-white transition"
                >
                  Back to top ↑
                </button>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom */}
        <div className="border-t border-gray-800 pt-8 text-center text-sm text-gray-400">
          <p>© {new Date().getFullYear()} MenoEaze. All rights reserved.</p>
          <p className="mt-2 text-xs text-gray-500">
            Built for women’s menopause health analytics and AI-supported care.
          </p>
        </div>
      </div>
    </footer>
  )
}