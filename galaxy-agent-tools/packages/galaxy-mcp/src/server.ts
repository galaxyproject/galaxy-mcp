import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { allOperations, runWithEnvelope, createGalaxyContext } from "@galaxyproject/galaxy-ops";

export function toolNames(): string[] {
  return allOperations.map((op) => op.name);
}

/** Per-tool MCP annotations derived from the op's readOnly hint (default read-only). */
export function toolAnnotations(): Record<string, { readOnlyHint: boolean; destructiveHint: boolean }> {
  const out: Record<string, { readOnlyHint: boolean; destructiveHint: boolean }> = {};
  for (const op of allOperations) {
    const readOnly = op.readOnly !== false;
    out[op.name] = { readOnlyHint: readOnly, destructiveHint: false };
  }
  return out;
}

export function buildServer(conn: { baseUrl: string; apiKey: string }): McpServer {
  const server = new McpServer({ name: "galaxy", version: "0.0.0" });
  const ctx = createGalaxyContext(conn);
  const annotations = toolAnnotations();
  for (const op of allOperations) {
    server.registerTool(
      op.name,
      { description: op.summary, inputSchema: op.input, annotations: annotations[op.name] },
      async (args: unknown) => {
        const result = await runWithEnvelope(op as never, args as never, ctx);
        return { content: [{ type: "text" as const, text: JSON.stringify(result) }], isError: !result.success };
      },
    );
  }
  return server;
}
