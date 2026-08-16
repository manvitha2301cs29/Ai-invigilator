import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// docs/03_DEPLOYMENT_GUIDE.txt Part 1 Step 4: `npm run build` produces
// dist/, deployed as a static site (Railway/Render/Vercel/Netlify).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
  },
});
