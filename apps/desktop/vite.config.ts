import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Tauri 开发模式固定端口，端口被占用直接报错
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**"] },
  },
  build: {
    target: "es2021",
    chunkSizeWarningLimit: 2000,
  },
});
