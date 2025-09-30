"use client"

import { useState } from "react"
import { Check } from "lucide-react"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

type Solution = {
  name: string
  description: string
  features: string[]
  bg: string
  badge?: string
}

const solutionSets: Record<"dev" | "prod", Solution[]> = {
  dev: [
    {
      name: "FAISS + Gemini API",
      description: "Lightweight RAG setup ideal for fast prototyping. Uses FAISS local vector DB and Gemini 2.5 API.",
      features: [
        "FAISS for local embeddings",
        "HuggingFace MiniLM model",
        "Gemini Flash 2.5 API for LLM",
        "Great for demos, quick iteration",
      ],
      bg: "bg-[#E3F2FD]",
      badge: "For Fast Prototyping",
    },
    {
      name: "Mini LLM + FAISS (Basic)",
      description: "Cloud-ready deployment with FAISS vector DB and a mini LLM. No lazy loading or GPU dependency.",
      features: [
        "FAISS for scalable retrieval",
        "Self-hosted mini LLM",
        "Clean separation of vector & model layers",
        "No GPU or infra dependency",
      ],
      bg: "bg-[#E8F5E9]",
      badge: "Lightweight Deployment",
    },
    {
      name: "Optimized GPU Setup",
      description: "Best for production-grade use with GPU inference, cold-start control, and optimized vector search.",
      features: [
        "Mini LLM w/ NVIDIA Inference",
        "FAISS w/ lazy loading",
        "Render-ready Docker setup",
        "Cost-efficient cold-starts",
      ],
      bg: "bg-[#FFF3E0]",
      badge: "Production Optimized",
    },
  ],
  prod: [
    // Optional: Mirror the same content or adapt per environment
  ],
}

export function DeploymentOptionsSection() {
  const [env, setEnv] = useState<"dev" | "prod">("dev")

  return (
    <section className="py-12 w-full bg-background">
      <div className="max-w-6xl mx-auto px-6 text-center">
        <h2 className="text-3xl font-bold mb-2">AI Deployment Options</h2>
        <p className="text-muted-foreground mb-8 text-sm sm:text-base">
          Three tailored setups depending on your use-case: dev-stage, scale-ready, or full production.
        </p>

        <div className="flex justify-center mb-12">
          <ToggleGroup
            type="single"
            value={env}
            onValueChange={(val: string) => {
              if (val === "dev" || val === "prod") setEnv(val)
            }}
            className="border rounded-lg bg-muted/40"
          >
            <ToggleGroupItem value="dev">Overview</ToggleGroupItem>
            <ToggleGroupItem value="prod" disabled>Prod (Coming Soon)</ToggleGroupItem>
          </ToggleGroup>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 text-left">
          {solutionSets[env].map((solution, idx) => (
            <div
              key={idx}
              className={cn(
                "rounded-2xl p-6 border transition-all hover:shadow-lg group relative",
                solution.bg
              )}
            >
              <h3 className="text-xl font-semibold text-foreground">{solution.name}</h3>

              {solution.badge && (
                <Badge className="mt-2 inline-block text-xs" variant="secondary">
                  {solution.badge}
                </Badge>
              )}

              <p className="text-sm text-black mt-3 mb-4">{solution.description}</p>

              <ul className="space-y-3 text-sm mt-6">
                {solution.features.map((feature, i) => (
                  <li key={i} className="flex items-start gap-2 text-black">
                    <Check className="h-4 w-4 text-primary mt-0.5" />
                    <span>{feature}</span>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
