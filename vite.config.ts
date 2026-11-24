import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(process.cwd(), './src'),
    },
  },
  optimizeDeps: {
    exclude: ['@/wasm/nn_engine.js'],
  },
  build: {
    chunkSizeWarningLimit: 1200,
  },
  server: {
    port: 3000,
  },
});
