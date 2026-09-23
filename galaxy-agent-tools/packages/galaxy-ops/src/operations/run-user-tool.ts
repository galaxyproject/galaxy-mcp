import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { legacyGet, legacyPost } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: result from POST /api/tools for a user-defined tool run. */
export interface UserToolRun {
  outputs?: unknown[];
  jobs?: unknown[];
  [k: string]: unknown;
}

const input = {
  historyId: z.string().describe("Galaxy history id where outputs will be placed"),
  toolUuid: z.string().describe("The UUID of the user-defined tool"),
  inputs: z
    .record(z.string(), z.unknown())
    .describe("tool inputs; dataset refs as {src:'hda',id}"),
};
type In = { historyId: string; toolUuid: string; inputs: Record<string, unknown> };

/** Minimal shape we read back from GET /api/unprivileged_tools/{tool_uuid}. */
interface ToolLookup {
  tool_id?: string;
  representation?: { version?: string };
  [k: string]: unknown;
}

async function run(i: In, ctx: GalaxyContext): Promise<UserToolRun> {
  // Step 1: look up tool_id and version from the UDT record.
  const toolInfo = await legacyGet<ToolLookup>(ctx, "/api/unprivileged_tools/{tool_uuid}", {
    params: { path: { tool_uuid: i.toolUuid } },
  });

  if (!toolInfo.tool_id) {
    throw new GalaxyNotFoundError(`No user-defined tool found with UUID '${i.toolUuid}'`);
  }

  const toolVersion = toolInfo.representation?.version ?? "0.1.0";

  // Step 2: run via POST /api/tools (the synchronous UDT path, off-schema -> legacyPost).
  return legacyPost<UserToolRun>(ctx, "/api/tools", {
    body: {
      history_id: i.historyId,
      tool_uuid: i.toolUuid,
      tool_version: toolVersion,
      inputs: i.inputs,
      input_format: "legacy",
    },
  });
}

export const runUserToolOp: Operation<typeof input, UserToolRun> = {
  name: "run_user_tool",
  domain: "userTools",
  summary: "Run a user-defined tool via the Galaxy tools API (two-step: lookup then POST /api/tools).",
  input,
  readOnly: false,
  run,
  project: (_out, i) => ({ message: `Submitted user tool ${i.toolUuid} to history ${i.historyId}` }),
};

register(runUserToolOp as AnyOperation);

export const runUserTool = (i: In, ctx: GalaxyContext) => runOperation(runUserToolOp, i, ctx);
