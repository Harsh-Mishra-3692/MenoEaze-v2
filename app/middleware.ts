import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'
import { createServerClient } from '@supabase/ssr'

export async function middleware(request: NextRequest) {
  const response = NextResponse.next()

  const supabase = createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get: (name) => request.cookies.get(name)?.value,
        set: (name, value, options) => {
          response.cookies.set({
            name,
            value,
            ...options
          })
        },
        remove: (name, options) => {
          response.cookies.set({
            name,
            value: '',
            ...options
          })
        }
      }
    }
  )

  const {
    data: { session }
  } = await supabase.auth.getSession()

  const { pathname } = request.nextUrl

  /* ---------------- PROTECT DASHBOARD ---------------- */
  if (pathname.startsWith('/dashboard')) {
    if (!session) {
      return NextResponse.redirect(new URL('/auth', request.url))
    }
  }

  /* ---------------- PREVENT AUTH PAGE IF LOGGED IN ---------------- */
  if (pathname.startsWith('/auth')) {
    if (session) {
      return NextResponse.redirect(new URL('/dashboard', request.url))
    }
  }

  return response
}

/* ---------------- APPLY TO THESE ROUTES ---------------- */
export const config = {
  matcher: ['/dashboard/:path*', '/auth']
}