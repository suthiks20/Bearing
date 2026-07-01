import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: [
      'react',
      'react/jsx-runtime',
      'react/jsx-dev-runtime',
      'react-dom',
      'react-dom/client',
      'framer-motion',
      '@react-three/fiber',
      '@react-three/drei',
      'three',
      'recharts',
      'lucide-react',
    ],
  },
})
