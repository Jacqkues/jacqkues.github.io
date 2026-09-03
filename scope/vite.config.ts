import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served at https://jacqkues.github.io/scope/ from the jacques-blog repo.
export default defineConfig({
  plugins: [react()],
  base: '/scope/',
  server: { port: 5173, host: true },
})
