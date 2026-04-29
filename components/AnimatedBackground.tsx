'use client'

export default function AnimatedBackground() {
  return (
    <div className="absolute inset-0 -z-10 overflow-hidden pointer-events-none">
      <div className="blob blob1" />
      <div className="blob blob2" />
      <div className="blob blob3" />
    </div>
  )
}