import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        base: {
          950: '#0a0a0f',
          900: '#101018',
          800: '#161620',
          700: '#1e1e2b',
        },
        healthy: {
          DEFAULT: '#22d3ee',
          dim: '#0e7490',
        },
        warning: {
          DEFAULT: '#f59e0b',
          dim: '#92400e',
        },
        failure: {
          DEFAULT: '#ef4444',
          dim: '#7f1d1d',
        },
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Space Mono"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        glass: '0 8px 32px 0 rgba(0, 0, 0, 0.45)',
        glow: '0 0 40px rgba(34, 211, 238, 0.25)',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '0.35', transform: 'scale(1)' },
          '50%': { opacity: '0.6', transform: 'scale(1.08)' },
        },
        drift: {
          '0%': { transform: 'translate(0, 0)' },
          '50%': { transform: 'translate(12px, -16px)' },
          '100%': { transform: 'translate(0, 0)' },
        },
      },
      animation: {
        'pulse-glow': 'pulse-glow 6s ease-in-out infinite',
        drift: 'drift 14s ease-in-out infinite',
      },
    },
  },
  plugins: [],
} satisfies Config;
