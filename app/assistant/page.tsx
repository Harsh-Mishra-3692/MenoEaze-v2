"use client"

import { useAuth } from "@/lib/auth"
import AssistantChat from "@/components/assistant/AssistantChat"

export default function AssistantPage() {
  const { user, isAuthenticated } = useAuth()

  if (!isAuthenticated || !user) {
    return (
      <div className="flex items-center justify-center h-screen bg-[#0a0a0f]">
        <div className="animate-pulse text-gray-500 text-sm tracking-wide">
          Loading your assistant…
        </div>
      </div>
    )
  }

  return <AssistantChat userId={user.id} userEmail={user.email || ""} />
}