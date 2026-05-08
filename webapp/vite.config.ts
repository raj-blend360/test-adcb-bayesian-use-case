import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/upload': 'http://localhost:8000',
      '/transform': 'http://localhost:8000',
      '/model': 'http://localhost:8000',
      '/results': 'http://localhost:8000',
      '/tune': 'http://localhost:8000',
      '/visualize': 'http://localhost:8000',
      '/optimize': 'http://localhost:8000',
    },
  },
})
