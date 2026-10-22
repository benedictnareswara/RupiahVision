import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [
    react(),
    // Installable and offline: after the first visit, the app, the model and the
    // ONNX Runtime WebAssembly files are all served from the phone's cache.
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["icons/icon-192.png"],
      manifest: {
        name: "RupiahVision",
        short_name: "RupiahVision",
        description: "Pengenal uang kertas Rupiah untuk tunanetra",
        lang: "id",
        start_url: "./",
        scope: "./",
        display: "standalone",
        orientation: "portrait",
        background_color: "#000000",
        theme_color: "#000000",
        icons: [
          { src: "icons/icon-192.png", sizes: "192x192", type: "image/png", purpose: "any" },
          { src: "icons/icon-512.png", sizes: "512x512", type: "image/png", purpose: "any maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,png,json,onnx,wasm,mjs}"],
        // The model (~17 MB) and the WebAssembly runtime (~14 MB) exceed the 2 MB default.
        maximumFileSizeToCacheInBytes: 40 * 1024 * 1024,
      },
    }),
  ],
  // Relative asset paths so the build works under a GitHub Pages sub-path.
  base: "./",
  server: {
    // The shared label schema lives in ../data/labels.json.
    fs: { allow: [".."] },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
  },
});
