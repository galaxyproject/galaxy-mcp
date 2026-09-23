import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import {
  allOperations,
  createGalaxyContext,
  describeOperation,
  runWithEnvelope,
} from "@galaxyproject/galaxy-ops";

export function toolNames(): string[] {
  return allOperations.map((op) => op.name);
}

/** MCP annotations for a single op: read-only by default, destructive only when flagged. */
export function annotationsFor(op: { readOnly?: boolean; destructive?: boolean }): {
  readOnlyHint: boolean;
  destructiveHint: boolean;
} {
  return { readOnlyHint: op.readOnly !== false, destructiveHint: op.destructive === true };
}

/** Per-tool MCP annotations derived from each op's readOnly/destructive hints. */
export function toolAnnotations(): Record<string, { readOnlyHint: boolean; destructiveHint: boolean }> {
  const out: Record<string, { readOnlyHint: boolean; destructiveHint: boolean }> = {};
  for (const op of allOperations) {
    out[op.name] = annotationsFor(op);
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
      {
        description: describeOperation(op),
        inputSchema: op.input,
        annotations: annotations[op.name],
      },
      async (args: unknown) => {
        const result = await runWithEnvelope(op as never, args as never, ctx);
        return { content: [{ type: "text" as const, text: JSON.stringify(result) }], isError: !result.success };
      },
    );
  }
  return server;
}
