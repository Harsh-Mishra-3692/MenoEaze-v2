"use client";

import { useState, useRef, useEffect } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { usePathname } from "next/navigation";
import AuthModal from "./AuthModal";
import SymptomForm from "./SymptomForm";
import { Menu, X, Bot } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Header() {
  const { isAuthenticated, logout, user } = useAuth();
  const pathname = usePathname();

  const [authOpen, setAuthOpen] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "signup">("login");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [logOpen, setLogOpen] = useState(false);

  const dropdownRef = useRef<HTMLDivElement | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);

  const isActive = (path: string) =>
    pathname === path || pathname?.startsWith(path + "/");

  const navClass = (path: string) =>
    `px-3 py-2 rounded-lg text-sm font-medium transition ${
      isActive(path)
        ? "text-teal-600 bg-teal-50"
        : "text-gray-700 hover:text-teal-600"
    }`;

  // ---- SAFE USER INITIAL ----
  const userInitial =
    typeof user?.email === "string" && user.email.length > 0
      ? user.email[0].toUpperCase()
      : "U";

  // ---- CLICK OUTSIDE + ESC ----
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      const target = e.target as Node;

      if (dropdownRef.current && !dropdownRef.current.contains(target)) {
        setDropdownOpen(false);
      }

      if (modalRef.current && !modalRef.current.contains(target)) {
        setLogOpen(false);
      }
    };

    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setDropdownOpen(false);
        setLogOpen(false);
        setAuthOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleEsc);

    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleEsc);
    };
  }, []);

  // ---- CLOSE MOBILE ON ROUTE CHANGE ----
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  return (
    <>
      <header className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b shadow-sm">
        <nav className="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
          {/* LOGO */}
          <Link href="/" className="flex items-center gap-2">
            🌿
            <span className="text-2xl font-bold bg-gradient-to-r from-teal-600 to-rose-500 bg-clip-text text-transparent">
              MenoEaze
            </span>
          </Link>

          {/* DESKTOP NAV */}
          <div className="hidden md:flex items-center gap-6">
            <Link href="/" className={navClass("/")}>
              Home
            </Link>

            {isAuthenticated && (
              <>
                <Link href="/dashboard" className={navClass("/dashboard")}>
                  Dashboard
                </Link>

                <Link
                  href="/assistant"
                  className={`${navClass(
                    "/assistant"
                  )} flex items-center gap-1`}
                >
                  <Bot size={16} className="text-teal-500 animate-pulse" />
                  AI Assistant
                </Link>

                <button
                  onClick={() => setLogOpen(true)}
                  className="bg-gradient-to-r from-teal-600 to-rose-500 text-white px-4 py-2 min-h-[44px] rounded-lg text-sm font-semibold hover:shadow-md transition"
                >
                  + Log Symptom
                </button>
              </>
            )}

            {/* AUTH */}
            {isAuthenticated ? (
              <div className="relative ml-4" ref={dropdownRef}>
                <button
                  onClick={() => setDropdownOpen((p) => !p)}
                  className="w-[44px] h-[44px] rounded-full bg-gradient-to-r from-teal-600 to-rose-500 text-white flex items-center justify-center font-semibold text-sm"
                >
                  {userInitial}
                </button>

                {dropdownOpen && (
                  <div className="absolute right-0 mt-3 w-56 bg-white rounded-xl shadow-lg border py-2">
                    <div className="px-4 py-2 text-xs text-gray-500 truncate">
                      {user?.email ?? "No email"}
                    </div>

                    <div className="border-t my-1" />

                    <button
                      onClick={() => {
                        logout();
                        setDropdownOpen(false);
                      }}
                      className="block w-full text-left px-4 py-2 text-sm hover:bg-gray-50 text-red-500"
                    >
                      Logout
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <>
                <button
                  onClick={() => {
                    setAuthMode("login");
                    setAuthOpen(true);
                  }}
                  className="text-sm text-gray-700 hover:text-purple-600 min-h-[44px]"
                >
                  Sign In
                </button>

                <button
                  onClick={() => {
                    setAuthMode("signup");
                    setAuthOpen(true);
                  }}
                  className="bg-gradient-to-r from-teal-600 to-rose-500 text-white px-4 py-2 min-h-[44px] rounded-lg text-sm font-semibold"
                >
                  Get Started
                </button>
              </>
            )}
          </div>

          {/* MOBILE BUTTON */}
          <button
            onClick={() => setMobileOpen((p) => !p)}
            className="md:hidden min-h-[44px] min-w-[44px]"
          >
            {mobileOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </nav>

        {/* MOBILE MENU */}
        <AnimatePresence>
          {mobileOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="md:hidden border-t bg-white overflow-hidden"
            >
              <div className="flex flex-col px-6 py-4 gap-4">
                <Link href="/" className={navClass("/")}>
                  Home
                </Link>

                {isAuthenticated && (
                  <>
                    <Link href="/dashboard" className={navClass("/dashboard")}>
                      Dashboard
                    </Link>

                    <Link
                      href="/assistant"
                      className={`${navClass(
                        "/assistant"
                      )} flex items-center gap-1`}
                    >
                      <Bot size={16} className="text-teal-500" />
                      AI Assistant
                    </Link>

                    <button
                      onClick={() => {
                        setMobileOpen(false);
                        setLogOpen(true);
                      }}
                      className="bg-gradient-to-r from-teal-600 to-rose-500 text-white px-4 py-2 rounded-lg text-sm font-semibold"
                    >
                      + Log Symptom
                    </button>
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* LOG MODAL */}
      {logOpen && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50">
          <div
            ref={modalRef}
            className="bg-white w-full max-w-xl rounded-2xl shadow-2xl p-8 relative"
          >
            <button
              onClick={() => setLogOpen(false)}
              className="absolute top-2 right-2 min-h-[44px] min-w-[44px]"
            >
              ✕
            </button>

            {/* ✅ FIXED PROP NAME */}
            <SymptomForm onComplete={() => setLogOpen(false)} />
          </div>
        </div>
      )}

      <AuthModal
        isOpen={authOpen}
        onClose={() => setAuthOpen(false)}
        initialMode={authMode}
      />
    </>
  );
}