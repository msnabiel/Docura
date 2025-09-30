"use client";
import React from "react";
import { motion } from "framer-motion";
import { Keyboard, MousePointerClick, Cpu } from "lucide-react";

export default function GuidePage() {
  const features = [
    {
      title: "Always-On Overlay",
      description:
        "A frameless, transparent overlay that stays on top of all apps. Works seamlessly across macOS, Windows, and Linux.",
      icon: Cpu,
    },
    {
      title: "Click-Through Mode",
      description:
        "Toggle whether clicks pass through the overlay to underlying apps.",
      icon: MousePointerClick,
    },
    {
      title: "AI Assistant (Gemini)",
      description:
        "Ask Gemini AI questions in real time using the input box. Supports unlimited queries.",
      icon: Keyboard,
    },
    {
      title: "Dynamic Resizing",
      description:
        "Resize the overlay dynamically to fit your workflow.",
      icon: MousePointerClick,
    },
  ];

  const shortcuts = [
    { keys: ["Cmd/Ctrl", "Shift", "O"], action: "Toggle overlay show/hide" },
    { keys: ["Cmd/Ctrl", "Shift", "I"], action: "Toggle click-through mode" },
    { keys: ["Cmd/Ctrl", "Shift", "H"], action: "Quick hide/show overlay" },
    { keys: ["Cmd/Ctrl", "Shift", "M"], action: "Trigger Gemini ask" },
    { keys: ["Cmd/Ctrl", "Shift", "G"], action: "Open input box" },
  ];

  return (
    <motion.main
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="min-h-screen bg-gradient-to-br from-[#E0F7FA] via-white to-[#FFF3E0] dark:from-[#0f172a] dark:via-[#1e293b] dark:to-[#0f172a] text-gray-900 dark:text-gray-100 py-16 px-6"
    >
      <div className="max-w-4xl mx-auto space-y-16">
        {/* Short Intro */}
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center text-lg sm:text-xl text-gray-700 dark:text-gray-300"
        >
          Halo AI is an always-on overlay that gives you AI assistance while you work. Learn features, shortcuts, and tips below.
        </motion.p>

        {/* Features */}
<div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
  {features.map((f, i) => (
    <motion.div
      key={i}
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      whileHover={{ y: -2, boxShadow: "0 4px 15px rgba(0,0,0,0.15)" }}
      transition={{ delay: i * 0.1, type: "spring", stiffness: 100 }}
      className="flex items-start gap-4 bg-white dark:bg-gray-800 p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200"
    >
      <f.icon className="h-6 w-6 text-indigo-500 mt-1 flex-shrink-0" />
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{f.title}</h3>
        <p className="text-gray-600 dark:text-gray-300 mt-1">{f.description}</p>
      </div>
    </motion.div>
  ))}
</div>


        {/* Shortcuts */}
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold mb-4">Available Shortcuts</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full border border-gray-300 dark:border-gray-700 rounded-lg overflow-hidden">
              <thead className="bg-gray-100 dark:bg-gray-900">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">Keys</th>
                  <th className="text-left px-4 py-2 font-medium">Action</th>
                </tr>
              </thead>
              <tbody>
                {shortcuts.map((s, i) => (
                  <tr
                    key={i}
                    className={`border-t border-gray-200 dark:border-gray-700 ${
                      i % 2 === 0 ? "bg-gray-50 dark:bg-gray-900/50" : ""
                    }`}
                  >
<td className="px-4 py-2">
  <div className="flex gap-1 flex-wrap items-center">
    {s.keys.map((key, j) => (
      <React.Fragment key={j}>
        <span className="bg-gray-200 dark:bg-gray-700 px-2 py-1 rounded text-sm font-mono text-gray-800 dark:text-gray-100">
          {key}
        </span>
        {j < s.keys.length - 1 && <span className="text-gray-600 dark:text-gray-300 px-1">+</span>}
      </React.Fragment>
    ))}
  </div>
</td>

                    <td className="px-4 py-2">{s.action}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Tips Section */}
        <div>
          <h2 className="text-2xl sm:text-3xl font-bold mb-4">Tips & Tricks</h2>
          <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
            <li>Use click-through mode to interact with apps behind the overlay.</li>
            <li>Resize the overlay to fit your workflow.</li>
            <li>Use shortcuts to access Gemini input quickly without leaving your workflow.</li>
            <li>Unlimited queries mean you can ask anything during meetings or work sessions.</li>
          </ul>
        </div>
      </div>
    </motion.main>
  );
}
