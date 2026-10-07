import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { ZodRawShape } from "zod";
import {
  allOperations,
  createGalaxyContext,
  describeOperation,
  GALAXY_MCP_SURFACE,
  runWithEnvelope,
} from "@galaxyproject/galaxy-ops";
import { LaxArgumentsTransport } from "./lax-transport.js";
import { inWireNames, toOperationInput, wireShape } from "./wire-names.js";
// The version a client is told, read from the package rather than written twice. tsup
// inlines it into dist/, so nothing looks for a package.json at run time.
import { version } from "../package.json";

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
 * What FastMCP puts in front of a tool's own sentence on the way out.
 *
 * Measured, not guessed (fastmcp 3.4.2, `mask_error_details` false by default): a tool that
 * raises anything but a ToolError reaches the client as
 * `ToolError(f"Error calling tool {name!r}: {e}")`, so the text block carries this prefix and
 * then the message. `{name!r}` is a Python repr of a str, and every tool name here is
 * lowercase ASCII with underscores, so a single-quoted name is the whole of that rule.
 */
const toolErrorPrefix = (tool: string): string => `Error calling tool '${tool}': `;

/**
 * One tool call's result, as the wire carries it.
 *
 * A success is the whole envelope as one text block, which is what a client decodes and a
 * model reads -- named rather than inlined because the ops measure their page caps against
 * exactly this shape and cannot import it; see the test that pins it.
 *
 * A failure is not an envelope at all. The Python server raises, FastMCP turns the exception
 * into an error result, and what a client gets is `isError` with one text block carrying the
 * prefix above and the tool's sentence -- no JSON, no structured content, nothing to parse.
 * This says the same thing: an agent that reads a failure here reads it the way it reads one
 * from that server, rather than a success-shaped object with `success: false` inside it.
 */
export function toolResult(
  tool: string,
  result: { success: boolean; message?: string },
): {
  content: [{ type: "text"; text: string }];
  isError: boolean;
} {
  if (!result.success) {
    const text = `${toolErrorPrefix(tool)}${result.message ?? ""}`;
    return { content: [{ type: "text", text }], isError: true };
  }
  return { content: [{ type: "text", text: JSON.stringify(result) }], isError: false };
}

export function buildServer(conn: { baseUrl: string; apiKey: string }): McpServer {
  const server = new McpServer({ name: "galaxy", version });
  const ctx = createGalaxyContext(conn);
  const annotations = toolAnnotations();
  const shapes = new Map<string, ZodRawShape>();
  for (const op of allOperations) {
    // Advertised, validated and decoded under the Python parameter names; the op is handed
    // back its own. See wire-names.ts for why the rename stops at the top level. A tool the
    // Python server also serves is described as that server describes it, so a client is told
    // the same thing by either.
    const advertised = GALAXY_MCP_SURFACE[op.name];
    const { shape, object, toInput } = wireShape(op, advertised?.parameters);
    shapes.set(op.name, shape);
    server.registerTool(
      op.name,
      {
        description: advertised?.description ?? inWireNames(describeOperation(op), op.input),
        inputSchema: object,
        annotations: annotations[op.name],
      },
      async (args: unknown) => {
        const input = toOperationInput(args as Record<string, unknown>, toInput);
        return toolResult(op.name, await runWithEnvelope(op as never, input as never, ctx));
      },
    );
  }
  // Arguments are decoded as a call arrives, under the same wire names they are validated
  // against; see LaxArgumentsTransport for why it is here and not in the handlers above.
  const connect = server.connect.bind(server);
  server.connect = (transport: Transport) => connect(new LaxArgumentsTransport(transport, shapes));
  return server;
}
