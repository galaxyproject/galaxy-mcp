/** @type {import('dependency-cruiser').IConfiguration} */
module.exports = {
  forbidden: [
    {
      name: "surface-no-raw-http",
      comment:
        "MCP/CLI surfaces must go through @galaxyproject/galaxy-ops' public API -- " +
        "never the raw client, the choreography, the wait loop, or the bindings.",
      severity: "error",
      from: { path: "^packages/galaxy-(mcp|cli)/src" },
      to: {
        path:
          "(^|/)openapi-fetch|" +
          "(^|/)@galaxyproject/galaxy-api-client|" +
          "packages/galaxy-ops/src/(client|execute-tool-request|wait|context)",
      },
    },
  ],
  options: {
    doNotFollow: { path: "node_modules" },
    tsPreCompilationDeps: true,
    // dependency-cruiser 16 declares TypeScript support as <6 and we are on 6, so
    // it leaves .ts out of its default extension list and an extensionless
    // "./program" is never tried as program.ts -- which left the cruise stopping at
    // each package's entry file and reporting a clean run.
    enhancedResolveOptions: {
      extensions: [".ts", ".tsx", ".d.ts", ".js", ".mjs", ".cjs", ".json"],
    },
  },
};
