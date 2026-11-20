import type { AddressInfo } from "node:net";
import { parseArgs } from "node:util";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import { envFlag, httpConfig, startHttpServer } from "./http.js";
import { APPS_ENV, buildServer } from "./server.js";

// The Python server's flags and environment variables, with the same defaults.
const USAGE =
  "usage: galaxy-mcp [--transport stdio|streamable-http] [--host HOST] [--port PORT] [--path PATH] [--allow-unauthenticated]";

function fail(message: string): never {
  console.error(`${USAGE}\ngalaxy-mcp: ${message}`);
  process.exit(2);
}

function readArgs() {
  try {
    return parseArgs({
      options: {
        transport: { type: "string" },
        host: { type: "string" },
        port: { type: "string" },
        path: { type: "string" },
        "allow-unauthenticated": { type: "boolean" },
      },
    }).values;
  } catch (error) {
    return fail((error as Error).message);
  }
}

const args = readArgs();
const baseUrl = process.env.GALAXY_URL;
const apiKey = process.env.GALAXY_API_KEY;
if (!baseUrl || !apiKey) {
  console.error("galaxy-mcp: set GALAXY_URL and GALAXY_API_KEY");
  process.exit(1);
}

const apps = envFlag(process.env[APPS_ENV]);
const transport = (args.transport ?? process.env.GALAXY_MCP_TRANSPORT ?? "stdio").toLowerCase();
if (transport === "stdio") {
  await buildServer(createGalaxyContext({ baseUrl, apiKey }), { apps }).connect(new StdioServerTransport());
  console.error("galaxy-mcp: connected on stdio");
} else if (transport === "streamable-http") {
  let config;
  try {
    config = httpConfig({ ...args, allowUnauthenticated: args["allow-unauthenticated"] }, process.env);
  } catch (error) {
    fail((error as Error).message);
  }
  // One context for every request, so the Galaxy version it probes is fetched once.
  const ctx = createGalaxyContext({ baseUrl, apiKey });
  let httpServer;
  try {
    httpServer = await startHttpServer({ ...config, newServer: () => buildServer(ctx, { allowLocalFiles: config.allowLocalFiles, apps }) });
  } catch (error) {
    // A port in use, or an address this machine does not have: say so in one line.
    console.error(`galaxy-mcp: ${(error as Error).message}`);
    process.exit(1);
  }
  // The bound port, not the configured one, so --port 0 reports where it actually listens.
  const { port } = httpServer.address() as AddressInfo;
  const { host, path } = config;
  console.error(`galaxy-mcp: listening on http://${host.includes(":") ? `[${host}]` : host}:${port}${path}`);
} else {
  fail(`unsupported transport '${transport}'; choose stdio or streamable-http`);
}
