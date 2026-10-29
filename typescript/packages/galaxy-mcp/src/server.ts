import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import { z, type ZodRawShape } from "zod";
import {
  allOperations,
  describeOperation,
  getSelectedDatasets,
  MAX_SELECTED_DATASETS,
  runWithEnvelope,
  type GalaxyContext,
} from "@galaxyproject/galaxy-ops";
import { registerCard, type CardSpec } from "./apps/card-page.js";
import { datasetsCard, historiesCard, historyCard } from "./apps/cards.js";
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

/**
 * Tools that take a path on the machine running this server. Over stdio that machine is the
 * caller's own; over HTTP it is not, so they are refused there with the Python server's
 * sentence (its `require_local_files`), unless the operator turns them back on.
 */
const localFileTools: Record<string, { uses: (input: Record<string, unknown>) => boolean; action: string; alternative: string }> = {
  upload_file: {
    uses: () => true,
    action: "upload_file",
    alternative: "Use upload_file_from_url to load data the Galaxy server can fetch itself.",
  },
  download_dataset: {
    uses: (input) => Boolean(input.filePath),
    action: "download_dataset(file_path=...)",
    alternative: "Omit file_path to receive the content in the response instead.",
  },
};

export const ALLOW_LOCAL_FILES_ENV = "GALAXY_MCP_ALLOW_LOCAL_FILES";
export const APPS_ENV = "GALAXY_MCP_APPS";

function localFilesRefusal({ action, alternative }: { action: string; alternative: string }): string {
  return (
    `${action} is disabled over HTTP transports, because the path would refer to the MCP ` +
    `server's filesystem rather than yours. ${alternative} Operators of a trusted single-user ` +
    `deployment can re-enable it with ${ALLOW_LOCAL_FILES_ENV}=1.`
  );
}

const MAX_CAPTION_LENGTH = 280;

/** show_datasets: a chosen set, not a read, so it has its own shape and answer. */
const showDatasetsShape = {
  history_id: z.string().describe("Encoded id of the history the datasets are in"),
  dataset_ids: z
    .array(z.string())
    .min(1)
    .max(MAX_SELECTED_DATASETS)
    .describe(
      `The datasets to show, in this order: their id from get_history_contents (not hid), 1 to ${MAX_SELECTED_DATASETS}`,
    ),
  caption: z
    .string()
    .max(MAX_CAPTION_LENGTH)
    .optional()
    .describe("A short note on why these datasets, shown as yours rather than Galaxy's"),
};

const showDatasetsDescription =
  "Show a chosen set of datasets from one history as a card: only these, in the order given. First read " +
  "with get_history_contents and pick the entries the user asked for, datasets only (history_content_type " +
  '"dataset"), passing their id, not their hid. For "the latest successful ones", for example, read with ' +
  'order="create_time-dsc" and take the first entries whose state is "ok", paging on until you have enough ' +
  "or saying how far you read. Galaxy supplies every name and state; an id that is not in this history, or " +
  "that the user cannot access, is shown as such. The card opens on five rows. Use a show_* tool only when " +
  "the user wants to see something, and show at most one card per reply across all of them.";

export interface ServerOptions {
  /** Whether tools may read or write paths on this machine. True over stdio. */
  allowLocalFiles?: boolean;
  /** Whether to add the MCP Apps cards and the show_* tools that open them. Off by default. */
  apps?: boolean;
}

/**
 * Cards over a read. Each show_* tool runs an existing read and returns exactly what it returns;
 * the difference is the card the host renders beside it. Reads stay card-free, so an agent can
 * look things up without putting a card in front of the user each time.
 */
const readCards: { name: string; operation: string; card: CardSpec; guidance: string }[] = [
  {
    name: "show_histories",
    operation: "get_histories",
    card: historiesCard,
    guidance:
      "Show the user's histories as a card they can page through, with a small limit such as 5 to 20: " +
      "without one, every history comes back at once. The card opens on five rows. Use a show_* tool only " +
      "when the user wants to see something, and show at most one card per reply across all of them. For " +
      "your own lookups and polling use get_histories, which shows nothing.",
  },
  {
    name: "show_history",
    operation: "get_history_contents",
    card: historyCard,
    guidance:
      "Show one history's datasets and collections as a card the user can page through and open, with a " +
      "small limit such as 5 to 20. The card opens on five rows. Use a show_* tool only when the user wants " +
      "to see something, and show at most one card per reply across all of them. For your own lookups and " +
      "polling use get_history_contents, which shows nothing. To show only some datasets, use show_datasets.",
  },
];

/**
 * One MCP server over one Galaxy context. The context is passed in rather than built here so
 * servers can share it, along with the Galaxy version it caches.
 */
export function buildServer(ctx: GalaxyContext, { allowLocalFiles = true, apps = false }: ServerOptions = {}): McpServer {
  const server = new McpServer({ name: "galaxy", version });
  const annotations = toolAnnotations();
  const shapes = new Map<string, ZodRawShape>();
  const register = (name: string, op: (typeof allOperations)[number], extra: { guidance?: string; _meta?: Record<string, unknown> } = {}) => {
    // Advertised, validated and decoded under the Python parameter names; the op is handed
    // back its own. See wire-names.ts for why the rename stops at the top level.
    const { shape, object, toInput } = wireShape(op);
    shapes.set(name, shape);
    const description = inWireNames(describeOperation(op), op.input);
    server.registerTool(
      name,
      {
        description: extra.guidance ? `${extra.guidance}\n\n${description}` : description,
        inputSchema: object,
        annotations: annotations[op.name],
        ...(extra._meta && { _meta: extra._meta }),
      },
      async (args: unknown) => {
        const input = toOperationInput(args as Record<string, unknown>, toInput);
        const localFiles = localFileTools[op.name];
        if (!allowLocalFiles && localFiles?.uses(input as Record<string, unknown>)) {
          return toolResult(name, { success: false, message: localFilesRefusal(localFiles) });
        }
        return toolResult(name, await runWithEnvelope(op as never, input as never, ctx));
      },
    );
  };
  for (const op of allOperations) register(op.name, op);
  if (apps) {
    for (const { name, operation, card, guidance } of readCards) {
      registerCard(server, name, card);
      const op = allOperations.find((candidate) => candidate.name === operation)!;
      register(name, op, { guidance, _meta: { ui: { resourceUri: card.uri } } });
    }
    registerCard(server, "show_datasets", datasetsCard);
    shapes.set("show_datasets", showDatasetsShape);
    server.registerTool(
      "show_datasets",
      {
        description: showDatasetsDescription,
        inputSchema: z.object(showDatasetsShape).strict(),
        annotations: annotationsFor({}),
        _meta: { ui: { resourceUri: datasetsCard.uri } },
      },
      async ({ history_id, dataset_ids, caption }) => {
        try {
          const shown = await getSelectedDatasets({ historyId: history_id, datasetIds: dataset_ids }, ctx);
          const count = shown.selected.length;
          const answer = {
            success: true,
            data: { ...shown, ...(caption && { caption }) },
            message: `Showing ${shown.available} of ${count} selected datasets`,
            count,
          };
          return toolResult("show_datasets", answer);
        } catch (error) {
          return toolResult("show_datasets", { success: false, message: `Failed to show datasets: ${(error as Error).message}` });
        }
      },
    );
  }
  // Arguments are decoded as a call arrives, under the same wire names they are validated
  // against; see LaxArgumentsTransport for why it is here and not in the handlers above.
  const connect = server.connect.bind(server);
  server.connect = (transport: Transport) => connect(new LaxArgumentsTransport(transport, shapes));
  return server;
}
