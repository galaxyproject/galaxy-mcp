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

/**
 * The whole envelope as one text block, which is what a client decodes and a
 * model reads. Named rather than inlined because the ops measure their page caps
 * against exactly this shape and cannot import it; see the test that pins it.
 */
export function toolResult(result: { success: boolean }): {
  content: [{ type: "text"; text: string }];
  isError: boolean;
} {
  return { content: [{ type: "text", text: JSON.stringify(result) }], isError: !result.success };
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
      async (args: unknown) => toolResult(await runWithEnvelope(op as never, args as never, ctx)),
    );
  }
  return server;
}
