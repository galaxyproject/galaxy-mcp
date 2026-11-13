import { defineConfig } from "tsup";
export default defineConfig({
  entry: ["src/index.ts", "src/index.browser.ts"],
  format: ["esm"],
  dts: false,
  clean: true,
  target: "es2022",
  sourcemap: true,
});
