import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";
export default defineConfig({
  test: { include: ["test/**/*.test.ts"] },
  resolve: {
    alias: {
      "@galaxyproject/galaxy-ops": fileURLToPath(new URL("../galaxy-ops/src/index.ts", import.meta.url)),
    },
  },
});
