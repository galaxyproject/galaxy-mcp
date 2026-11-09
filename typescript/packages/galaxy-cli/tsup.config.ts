import { defineConfig } from "tsup";
export default defineConfig({
  entry: {
    index: "src/index.ts",
    gxy: "src/gxy.ts",
    "keyring-child": "src/auth/keyring-child.ts",
  },
  format: ["esm"],
  dts: false,
  clean: true,
  target: "es2022",
  banner: { js: "#!/usr/bin/env node" },
});
