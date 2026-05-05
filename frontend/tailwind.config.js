/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Outfit', 'Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        'dark-bg': '#050811',
        'dark-panel': '#0a0f18',
        'q-primary': '#10b981',
        'q-secondary': '#8b5cf6',
        'q-accent': '#06b6d4',
        'c-primary': '#3b82f6',
      },
      boxShadow: {
        'neon-cyan': '0 0 12px rgba(6,182,212,0.5), 0 0 24px rgba(6,182,212,0.2)',
        'neon-purple': '0 0 12px rgba(139,92,246,0.5), 0 0 24px rgba(139,92,246,0.2)',
        'neon-emerald': '0 0 12px rgba(16,185,129,0.5), 0 0 24px rgba(16,185,129,0.2)',
        'glass': '0 8px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05)',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'glass-gradient': 'linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01))',
      },
      animation: {
        'pulse-slow': 'pulse-slow 4s ease-in-out infinite',
        'spin-slow': 'spin-slow 3s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        'pulse-slow': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.6', transform: 'scale(1.05)' },
        },
        'spin-slow': {
          from: { transform: 'rotate(0deg)' },
          to: { transform: 'rotate(360deg)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(16,185,129,0.3)' },
          '100%': { boxShadow: '0 0 25px rgba(16,185,129,0.7)' },
        },
      },
    },
  },
  plugins: [],
}

