/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    // Design system: one accent (sky) + neutrals (white opacities, slate, zinc).
    // red is kept only for error/danger states. Everything else is unavailable on purpose.
    colors: {
      transparent: "transparent",
      current: "currentColor",
      white: "#ffffff",
      black: "#000000",
      slate: require("tailwindcss/colors").slate,
      zinc: require("tailwindcss/colors").zinc,
      sky: require("tailwindcss/colors").sky,
      red: require("tailwindcss/colors").red,
    },
    extend: {
      colors: {
        accent: {
          DEFAULT: "#38bdf8",
          dark: "#0284c7",
          light: "#7dd3fc",
        },
        surface: {
          deep: "#05070a",
          base: "#0c0e12",
          card: "#14171c",
        },
      },
    },
  },
  plugins: [],
}
