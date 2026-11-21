import { defineConfig } from "tsup";
export default defineConfig([
  // The command.
  {
    entry: ["src/index.ts"],
    format: ["esm"],
    dts: false,
    clean: true,
    target: "es2022",
    banner: { js: "#!/usr/bin/env node" },
  },
  // The library, for a program that builds the server itself.
  {
    entry: ["src/library.ts"],
    format: ["esm"],
    dts: false,
    target: "es2022",
    sourcemap: true,
  },
]);
