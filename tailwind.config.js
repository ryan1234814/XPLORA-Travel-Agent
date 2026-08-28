/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#38bdf8",
        secondary: "#818cf8",
        accent: "#67e8f9",
        dark: "#07090d",
        card: "rgba(22, 26, 33, 0.7)",
        teal: "#2dd4bf",
        amber: "#fbbf24",
        rose: "#fb7185",
        emerald: "#34d399",
        fuchsia: "#e879f9",
        indigo: "#818cf8",
      },
      fontFamily: {
        outfit: ["Outfit", "sans-serif"],
        inter: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
}
