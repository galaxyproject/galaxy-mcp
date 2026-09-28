import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError } from "../errors";
import { jsonObject } from "../json-object";
import { legacyGet, legacyPost } from "../legacy";
import { enrichedRunFailure, USER_TOOL_SHAPE_HINT } from "../tool-input-error";
import { preflightToolInputs } from "../tool-preflight";
import { envelopeFact, readFact, recordFact } from "./envelope-facts";
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
  inputs: jsonObject().describe("tool inputs; dataset refs as {src:'hda',id}"),
};
type In = { historyId: string; toolUuid: string; inputs: Record<string, unknown> };

/** Minimal shape we read back from GET /api/unprivileged_tools/{tool_uuid}. */
interface ToolLookup {
  tool_id?: string;
  representation?: { version?: string };
  [k: string]: unknown;
}

/**
 * The Galaxy tool_id behind the uuid, for the summary line.
 *
 * The other server's sentence names it, and it comes from the lookup rather than the
 * run's reply, so it travels beside the call.
 */
const userToolId = envelopeFact<string>("run_user_tool.tool_id");

/**
 * Why the inputs were not checked against the representation, when they were not.
 *
 * This tool holds the definition already, so nothing is fetched: either the representation
 * carries a parameter list or it does not, and the second answer is the clause.
 */
const uncheckedInputs = envelopeFact<string>("run_user_tool.unchecked_inputs");

async function run(i: In, ctx: GalaxyContext): Promise<UserToolRun> {
  // Step 1: look up tool_id and version from the UDT record.
  const toolInfo = await legacyGet<ToolLookup>(ctx, "/api/unprivileged_tools/{tool_uuid}", {
    params: { path: { tool_uuid: i.toolUuid } },
  });

  if (!toolInfo.tool_id) {
    throw new GalaxyNotFoundError(`No user-defined tool found with UUID '${i.toolUuid}'`);
  }

  recordFact(ctx, userToolId, toolInfo.tool_id);

  const toolVersion = toolInfo.representation?.version ?? "0.1.0";

  // The definition is in hand, so the check costs nothing and no request goes out for it.
  // An empty representation is a schema with no parameter list, which is the clause rather
  // than a reason to go looking for one in the toolbox -- a user tool is not in there.
  const unchecked = await preflightToolInputs(ctx, toolInfo.tool_id, i.inputs, {
    schema: (toolInfo.representation ?? {}) as Record<string, unknown>,
  });
  if (unchecked !== null) recordFact(ctx, uncheckedInputs, unchecked);

  // Step 2: run via POST /api/tools (the synchronous UDT path, off-schema -> legacyPost).
  try {
    return await legacyPost<UserToolRun>(ctx, "/api/tools", {
      body: {
        history_id: i.historyId,
        tool_uuid: i.toolUuid,
        tool_version: toolVersion,
        inputs: i.inputs,
        input_format: "legacy",
      },
    });
  } catch (err) {
    // The representation is the schema to explain the refusal with -- this tool is not in
    // the toolbox, so a lookup by id would 404 and the message would come back empty. No
    // credentials branch: the other server's user-tool run has no credentials handling.
    throw await enrichedRunFailure(ctx, err, {
      action: "Run user tool",
      toolId: toolInfo.tool_id,
      historyId: i.historyId,
      inputs: i.inputs,
      usedCredentials: false,
      schema: (toolInfo.representation ?? null) as Record<string, unknown> | null,
      shapeHint: USER_TOOL_SHAPE_HINT,
    });
  }
}

export const runUserToolOp: Operation<typeof input, UserToolRun> = {
  name: "run_user_tool",
  domain: "userTools",
  summary: "Run a user-defined tool via the Galaxy tools API (two-step: lookup then POST /api/tools).",
  input,
  readOnly: false,
  run,
  // server.py, run_user_tool. The tool_id is the one thing in it that the arguments
  // do not carry, and run() refuses a record without one -- so an empty name here
  // means nobody collected the fact, not that Galaxy sent no tool_id.
  project: (_out, i, facts) => {
    const unchecked = readFact(facts, uncheckedInputs);
    return {
      message:
        `Started user tool '${readFact(facts, userToolId) ?? ""}' (UUID: ${i.toolUuid}) ` +
        `in history '${i.historyId}'` +
        (unchecked ? ` (inputs not pre-checked: ${unchecked})` : ""),
    };
  },
  // server.py, run_user_tool: the lookup is a raw GET it checks itself and the run is a
  // bioblend write, so which sentence a caller reads depends on which of the two failed.
  failure: {
    shape: (facts) => (facts.method === "GET" ? "raise-for-status" : "bioblend-write"),
    action: "Run user tool",
    context: (i) => ({ history_id: i.historyId, tool_uuid: i.toolUuid }),
  },
};

register(runUserToolOp as AnyOperation);

export const runUserTool = (i: In, ctx: GalaxyContext) => runOperation(runUserToolOp, i, ctx);
