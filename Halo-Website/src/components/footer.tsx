"use client"

import Link from "next/link"
import { Mail, Github, Linkedin } from "lucide-react"


export interface FooterLink {
  label: string
  href: string
}

export interface FooterProps {
  logo?: React.ReactNode
  links?: FooterLink[]
  contactEmail?: string
  social?: {
    facebook?: string
    twitter?: string
    instagram?: string
    github?: string
    linkedin?: string
  }
}

export const Footer: React.FC<FooterProps> = ({
  logo = <span className="font-bold text-xl">Halo</span>,
  links = [
    { label: "Guide", href: "/guide" },
  { label: "Download", href: "/download" },
  { label: "Pricing", href: "/pricing" },
    //{ label: "GitHub", href: "https://github.com/msnabiel/Swipe-AI", external: true },
  ],
  contactEmail = "msyednabiel@gmail.com",
  social = {
    instagram: "https://instagram.com/docura_ai",
    twitter: "https://twitter.com/docura_ai",
    facebook: "https://facebook.com/docura.ai",
    github: "https://github.com/msnabiel/Swipe-AI",
    linkedin: "https://linkedin.com/in/msnabiel",
  },
}) => {
  return (
    <footer className="bg-muted border-t text-muted-foreground">
      <div className="container px-4 py-10 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8">
        {/* Branding */}
        <div className="space-y-2">
          <Link href="/" className="text-foreground">
            {logo}
          </Link>
          <p className="text-sm">
Halo AI is your stealthy, always-on intelligence — helping you stay one step ahead in meetings, chats, and tasks, all while remaining completely unobtrusive.
          </p>
        </div>

        {/* Quick Links */}
        <div>
          <h3 className="text-sm font-semibold mb-2 text-foreground">Quick Links</h3>
          <ul className="space-y-1">
            {links.map((link) => (
              <li key={link.href}>
                <Link href={link.href} className="hover:text-foreground transition">
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        {/* Contact + Social */}
        <div>
          <h3 className="text-sm font-semibold mb-2 text-foreground">Contact</h3>
          <ul className="space-y-1">
            <li>
              <a
                href={`mailto:${contactEmail}`}
                className="hover:text-foreground transition flex items-center gap-1"
              >
                <Mail className="w-4 h-4" />
                {contactEmail}
              </a>
            </li>
          </ul>

          <div className="flex items-center gap-4 mt-4">
            {social?.github && (
              <a href={social.github} target="_blank" rel="noreferrer" className="hover:text-foreground">
                <Github className="h-5 w-5" />
              </a>
            )}
            {social?.linkedin && (
              <a href={social.linkedin} target="_blank" rel="noreferrer" className="hover:text-foreground">
                <Linkedin className="h-5 w-5" />
              </a>
            )}
          </div>
        </div>
      </div>

      <div className="border-t text-center py-4 text-xs">
        &copy; {new Date().getFullYear()} Halo AI. All rights reserved.
      </div>
    </footer>
  )
}
