"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { useState } from "react"
import { Menu, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import clsx from "clsx"

const components = [
  { name: "Button", slug: "button" },
  { name: "Card", slug: "card" },
  { name: "Badge", slug: "badge" },
]

export function NavBar() {
  const pathname = usePathname()
  const [isOpen, setIsOpen] = useState(false)
  const [query, setQuery] = useState("")

  const navLinks = [
    { label: "Home", href: "/" },
    { label: "Try It", href: "/chat" },
    { label: "Interviewer", href: "/interviewer" },
    // Removed Models link
    { label: "Documentation", href: "/docs/introduction" },
    { label: "About Us", href: "/about" },
    {
      label: "GitHub",
      href: "https://github.com/msnabiel/Swipe",
      external: true,
    },
  ]

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    const match = components.find(
      (comp) => comp.name.toLowerCase() === query.trim().toLowerCase()
    )
    if (match) {
      window.location.href = `/components/${match.slug}`
    }
  }

  return (
    <nav className="border-b bg-background/80 backdrop-blur-sm px-4 py-3 sticky top-0 z-50 shadow-sm">
      <div className="flex items-center justify-between">
        <Link href="/" className="text-xl font-bold tracking-tight">
          ▨ Swipe
        </Link>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-6 flex-1 justify-center">
          {navLinks.map(({ label, href, external }) => {
            const isActive = pathname === href
            const baseClass =
              "relative text-sm font-medium transition-colors hover:text-primary"
            return external ? (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className={baseClass}
              >
                {label}
              </a>
            ) : (
              <Link
                key={label}
                href={href}
                className={clsx(baseClass, {
                  "text-primary": isActive,
                })}
              >
                {label}
                {isActive && (
                  <span className="absolute left-0 -bottom-1 w-full h-[2px] bg-primary transition-all rounded-full" />
                )}
              </Link>
            )
          })}
        </div>

        {/* Desktop Search */}
        <form onSubmit={handleSearch} className="hidden md:flex items-center space-x-2">
          <Input
            placeholder="Search components..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-[200px]"
          />
          <Button type="submit" variant="ghost">
            Go
          </Button>
        </form>

        {/* Mobile Toggle */}
        <Button
          variant="ghost"
          className="md:hidden"
          onClick={() => setIsOpen(!isOpen)}
        >
          {isOpen ? <X size={20} /> : <Menu size={20} />}
        </Button>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="mt-4 space-y-4 md:hidden">
          <form onSubmit={handleSearch} className="flex items-center gap-2 px-1">
            <Input
              placeholder="Search components..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <Button type="submit" variant="ghost">
              Go
            </Button>
          </form>
          {navLinks.map(({ label, href, external }) =>
            external ? (
              <a
                key={label}
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="block text-sm font-medium px-1"
              >
                {label}
              </a>
            ) : (
              <Link
                key={label}
                href={href}
                className={clsx(
                  "block text-sm font-medium px-1",
                  pathname === href && "text-primary"
                )}
                onClick={() => setIsOpen(false)}
              >
                {label}
              </Link>
            )
          )}
        </div>
      )}
    </nav>
  )
}
