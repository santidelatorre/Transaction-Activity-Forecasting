import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const sourceDir = fileURLToPath(new URL('./src', import.meta.url));

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': sourceDir },
  },
  server: {
    proxy: { '/api': process.env.RECURRING_API_TARGET || 'http://127.0.0.1:8000' },
  },
  build: { outDir: 'dist', emptyOutDir: true },
});
