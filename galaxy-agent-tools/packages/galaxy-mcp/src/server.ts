import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { allOperations, runWithEnvelope, createGalaxyContext } from "@galaxyproject/galaxy-ops";

export function toolNames(): string[] {
  return allOperations.map((op) => op.name);
}

export function buildServer(conn: { baseUrl: string; apiKey: string }): McpServer {
  const server = new McpServer({ name: "galaxy", version: "0.0.0" });
  const ctx = createGalaxyContext(conn);
  for (const op of allOperations) {
    server.registerTool(
      op.name,
      { description: op.summary, inputSchema: op.input }, // input IS the raw Zod shape
      async (args: unknown) => {
        const result = await runWithEnvelope(op as never, args as never, ctx);
        return { content: [{ type: "text" as const, text: JSON.stringify(result) }] };
      },
    );
  }
  return server;
}
