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
  options: { doNotFollow: { path: "node_modules" }, tsPreCompilationDeps: true },
};
