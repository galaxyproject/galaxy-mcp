import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { buildServer } from "./server.js";

const baseUrl = process.env.GALAXY_URL;
const apiKey = process.env.GALAXY_API_KEY;
if (!baseUrl || !apiKey) {
  console.error("galaxy-mcp: set GALAXY_URL and GALAXY_API_KEY");
  process.exit(1);
}

const server = buildServer({ baseUrl, apiKey });
await server.connect(new StdioServerTransport());
console.error("galaxy-mcp: connected on stdio");
