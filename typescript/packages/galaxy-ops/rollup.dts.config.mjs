// Declarations are bundled here rather than by tsup: tsup calls the TypeScript
// compiler API directly and TypeScript 7 no longer ships one. rollup-plugin-dts
// falls back to @typescript/typescript6 for that API, while tsc 7 still does the
// type checking. Options mirror what tsup's own dts build passed.
import { readFileSync } from "node:fs";
import { dts } from "rollup-plugin-dts";

const pkg = JSON.parse(readFileSync(new URL("./package.json", import.meta.url), "utf8"));
const deps = [...Object.keys(pkg.dependencies ?? {}), ...Object.keys(pkg.peerDependencies ?? {})];

export default {
  input: {
    index: "src/index.ts",
    "index.browser": "src/index.browser.ts",
  },
  external: deps.map((dep) => new RegExp(`^${dep}($|\\/|\\\\)`)),
  plugins: [
    dts({
      compilerOptions: {
        declaration: true,
        noEmit: false,
        emitDeclarationOnly: true,
        noEmitOnError: true,
        checkJs: false,
        declarationMap: false,
        skipLibCheck: true,
        preserveSymlinks: false,
      },
    }),
  ],
  output: {
    dir: "dist",
    format: "esm",
    exports: "named",
    entryFileNames: "[name].d.ts",
    chunkFileNames: "[name]-[hash].d.ts",
  },
};
