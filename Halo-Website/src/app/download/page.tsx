"use client";
import { motion } from "framer-motion";

const downloads = [
  { name: "Halo AI Installer (macOS)", link: "/downloads/Halo-mac.dmg", size: "120 MB" },
  { name: "Halo AI Installer (Windows)", link: "/downloads/Halo-win.exe", size: "130 MB" },
  { name: "User Guide PDF", link: "/downloads/Halo-Guide.pdf", size: "5 MB" },
];

export default function DownloadPage() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col items-center p-6">
      <motion.h1
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-4xl sm:text-5xl font-extrabold text-gray-900 dark:text-gray-100 mb-6 text-center"
      >
        Download Halo AI
      </motion.h1>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="text-gray-700 dark:text-gray-300 mb-10 text-center max-w-xl"
      >
        Choose your installer or resources below and get started with Halo AI. Available for macOS and Windows.
      </motion.p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 w-full max-w-4xl">
        {downloads.map((d, i) => (
          <motion.a
            key={i}
            href={d.link}
            download
            whileHover={{ y: -2, boxShadow: "0 6px 20px rgba(0,0,0,0.08)" }}
            className="flex flex-col justify-between p-6 bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md transition-shadow duration-200 cursor-pointer"
          >
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{d.name}</h3>
              <p className="text-gray-500 dark:text-gray-400 mt-2">{d.size}</p>
            </div>
            <button className="mt-4 px-4 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg transition-colors duration-200 text-sm font-medium">
              Download
            </button>
          </motion.a>
        ))}
      </div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="text-gray-500 dark:text-gray-400 mt-12 text-sm text-center"
      >
        By downloading, you agree to our terms and privacy policy.
      </motion.p>
    </div>
  );
}
