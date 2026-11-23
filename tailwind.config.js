/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    // ===== Design system =====
    // A deliberate blend of two references:
    //  • Apple  — near-black cinematic surfaces, hairline borders, tight display
    //             tracking, generous whitespace, a single translucent nav.
    //  • Google Developers — one confident blue accent, crisp rounded cards,
    //             numbered steps and compact utility labels.
    //
    // The class names stay `sky` / `slate` / `red` so every existing screen
    // inherits the new palette without touching logic. Values are what change.
    colors: {
      transparent: "transparent",
      current: "currentColor",
      white: "#ffffff",
      black: "#000000",

      // Accent — Google blue family. 300 is the dark-surface text accent
      // (high contrast on black), 400/500 are solid fills for CTAs.
      sky: {
        50: "#e8f0fe",
        100: "#d2e3fc",
        200: "#aecbfa",
        300: "#8ab4f8", // links, eyebrows, labels on dark
        400: "#669df6", // solid fills, indicators
        500: "#1a73e8", // primary CTA fill
        600: "#1765cc",
        700: "#174ea6",
        800: "#0f3d82",
        900: "#0a2b5c",
        950: "#061a38",
      },

      // Neutrals — Apple's neutral grey ramp (warm-neutral, not blue-grey).
      slate: {
        50: "#f5f5f7",
        100: "#e8e8ed",
        200: "#d2d2d7",
        300: "#a1a1a6",
        400: "#86868b",
        500: "#6e6e73",
        600: "#48484a",
        700: "#333336",
        800: "#232326",
        900: "#1d1d1f",
        950: "#0a0a0b",
      },

      // Semantic only — kept narrow on purpose.
      red: {
        300: "#f28b82",
        400: "#ea4335",
        500: "#d93025",
        600: "#b3261e",
      },
      green: {
        300: "#81c995",
        400: "#34a853",
        500: "#1e8e3e",
      },
      orange: {
        300: "#fdd663",
        400: "#f9ab00",
        500: "#e37400",
      },
      // A cool semantic tone for weather/thermometer glyphs.
      blue: {
        300: "#aecbfa",
        400: "#669df6",
        500: "#1a73e8",
      },
    },
    extend: {
      colors: {
        accent: {
          DEFAULT: "#1a73e8",
          light: "#8ab4f8",
          dark: "#174ea6",
        },
        surface: {
          deep: "#000000",
          base: "#0a0a0b",
          card: "#161617",
          raised: "#1d1d1f",
        },
        hairline: "rgba(255,255,255,0.10)",
      },
      fontFamily: {
        sans: [
          "-apple-system", "BlinkMacSystemFont", "SF Pro Display", "SF Pro Text",
          "Inter", "Outfit", "Segoe UI", "Helvetica Neue", "sans-serif",
        ],
        display: [
          "Outfit", "-apple-system", "BlinkMacSystemFont", "SF Pro Display",
          "Inter", "Segoe UI", "Helvetica Neue", "sans-serif",
        ],
        mono: ["SF Mono", "ui-monospace", "Menlo", "Consolas", "monospace"],
      },
      fontSize: {
        // Apple-style display steps with matching tight leading.
        "display-xl": ["clamp(3rem, 8vw, 6.5rem)", { lineHeight: "1.02", letterSpacing: "-0.035em" }],
        "display-lg": ["clamp(2.5rem, 5.5vw, 4.25rem)", { lineHeight: "1.05", letterSpacing: "-0.03em" }],
        "display-md": ["clamp(2rem, 3.6vw, 2.75rem)", { lineHeight: "1.1", letterSpacing: "-0.025em" }],
      },
      borderRadius: {
        "4xl": "2rem",
      },
      maxWidth: {
        content: "1100px",
        wide: "1280px",
      },
      boxShadow: {
        // Apple product-card elevation: soft, low, never glowy.
        product: "0 2px 12px rgba(0,0,0,0.35), 0 18px 48px -12px rgba(0,0,0,0.55)",
        hairline: "inset 0 0 0 1px rgba(255,255,255,0.08)",
        cta: "0 1px 2px rgba(0,0,0,0.4), 0 8px 24px -8px rgba(26,115,232,0.55)",
      },
      transitionTimingFunction: {
        // Apple's restrained easing for large surface transitions.
        apple: "cubic-bezier(0.28, 0.11, 0.32, 1)",
      },
    },
  },
  plugins: [],
}
