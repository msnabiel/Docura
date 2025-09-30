// app/layout.tsx (Server Component)
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import ClientProviders from "./ClientProviders"; // separate client wrapper
import { NavBar } from "@/components/navbar";
import { Footer } from "@/components/footer";
const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Halo - Smarter Finance",
  description: "A modern fintech platform for smarter trading, investing, and digital payments.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable}`}>
      <body className="antialiased flex flex-col min-h-screen">
        {/* Navbar at top */}
        <NavBar />

        {/* Main content grows to fill available space */}
        <ClientProviders>
          <main className="flex-1">
            {children}
          </main>
        </ClientProviders>

        {/* Footer at bottom */}
        <Footer />
      </body>
    </html>
  );
}

