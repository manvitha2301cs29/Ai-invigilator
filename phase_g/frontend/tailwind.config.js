/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // A quiet, focused workspace palette -- deep ink/slate rather
        // than pure black, with ONE warm signal color (amber) reserved
        // for the away-stopwatch/warning states, since that's the one
        // thing in this product that should read as urgent. Verdict
        // colors (verdict-on-track/shorter/breaks) are semantic, not
        // decorative -- they're the same three colors everywhere a
        // verdict appears (badge, history row, live banner).
        ink: {
          950: "#0f1417",
          900: "#161d21",
          800: "#1f282d",
          700: "#2b363c",
          600: "#3d4a51",
          400: "#6b7a82",
          200: "#b7c2c7",
          100: "#dfe5e7",
          50: "#f5f7f7",
        },
        signal: {
          amber: "#d98b3f",
        },
        verdict: {
          ontrack: "#3f8f6b",
          breaks: "#c99a3a",
          shorter: "#c15b4a",
        },
      },
      fontFamily: {
        display: ["'Fraunces'", "serif"],
        body: ["'Inter'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "monospace"],
      },
    },
  },
  plugins: [],
};
