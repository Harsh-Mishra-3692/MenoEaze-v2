'use client'

import { usePathname } from 'next/navigation'
import Header from '@/components/Header'
import Footer from '@/components/Footer'
import FloatingAssistant from '@/components/FloatingAssistant'
import { useMemo } from 'react'

interface LayoutClientProps {
  children: React.ReactNode
}

export default function LayoutClient({ children }: LayoutClientProps) {
  const pathname = usePathname()

  const { hideAssistant, hideHeaderFooter } = useMemo(() => {
    const isAuth = pathname.startsWith('/auth')
    const isAssistant = pathname === '/assistant'
    return {
      hideAssistant: isAuth || isAssistant,
      hideHeaderFooter: isAssistant
    }
  }, [pathname])

  return (
    <>
      {!hideHeaderFooter && <Header />}

      <main className="flex-1 w-full">
        {children}
      </main>

      {!hideHeaderFooter && <Footer />}

      {!hideAssistant && <FloatingAssistant />}
    </>
  )
}