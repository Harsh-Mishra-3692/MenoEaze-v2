import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'
import LayoutClient from '@/components/LayoutClient'

const geistSans = Geist({
  subsets: ['latin'],
  display: 'swap'
})

const geistMono = Geist_Mono({
  subsets: ['latin'],
  display: 'swap'
})

export const metadata: Metadata = {
  title: {
    default: 'MenoEaze – AI Menopause Health Companion',
    template: '%s | MenoEaze'
  },
  description:
    'AI-powered menopause tracking, symptom analytics, and personalized health insights.'
}

export default function RootLayout({
  children
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.className} ${geistMono.className} bg-gray-50 text-gray-900 antialiased min-h-screen flex flex-col`}
      >
        <LayoutClient>
          {children}
        </LayoutClient>
      </body>
    </html>
  )
}