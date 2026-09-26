import { z } from "zod";
import type { components, GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyConnectionError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

export type InvocationDetail = components["schemas"]["WorkflowInvocationElementView"];
export type InvocationSummary = GetJson<"/api/invocations">[number];

const DEFAULT_VIEW = "collection";
const DEFAULT_STEP_DETAILS = false;

// Python spells every one of these `T | None = None` and treats the null as "not supplied", so
// null is accepted here and means the same thing. The two parameters Python declares without a
// null branch -- view and step_details -- reject it on both surfaces.
const input = {
  invocationId: z
    .string()
    .nullish()
    .describe("Encoded workflow invocation id. When given, returns that one invocation and ignores the filters"),
  workflowId: z.string().nullish().describe("List mode: only invocations of this stored workflow"),
  historyId: z.string().nullish().describe("List mode: only invocations in this history"),
  limit: z
    .number()
    .int()
    .nullish()
    .describe(
      "List mode: most invocations to return. Left out, the parameter is not sent and Galaxy's own " +
        "index default applies -- which is a page, not everything (26.1 returns 20). Pass a limit to " +
        "know what you are getting.",
    ),
  view: z
    .string()
    .default(DEFAULT_VIEW)
    .describe("List mode detail level: 'collection' for a summary per invocation, 'element' for the full record"),
  stepDetails: z
    .boolean()
    .default(DEFAULT_STEP_DETAILS)
    .describe("List mode: include per-step details (only applies when view is 'element')"),
};

type In = {
  invocationId?: string | null;
  workflowId?: string | null;
  historyId?: string | null;
  limit?: number | null;
  view?: string;
  stepDetails?: boolean;
};

/** One invocation when an id was given, the filtered listing when it was not. */
export type GetInvocationsResult = InvocationDetail | InvocationSummary[];

async function run(i: In, ctx: GalaxyContext): Promise<GetInvocationsResult> {
  if (i.invocationId) {
    const { data, error, response } = await ctx.client.GET("/api/invocations/{invocation_id}", {
      params: { path: { invocation_id: i.invocationId } },
    });
    if (error || !data) throw classifyHttp(response.status, error);
    return data as InvocationDetail;
  }

  const { data, error, response } = await ctx.client.GET("/api/invocations", {
    params: {
      query: {
        // Truthiness, not ?? -- bioblend builds the same request with `if workflow_id:`, so a
        // blank filter is no filter. Sent as "" it would reach Galaxy's encoded-id validator
        // and fail the whole listing over a value the caller plainly did not mean.
        workflow_id: i.workflowId || null,
        history_id: i.historyId || null,
        limit: i.limit ?? null,
        view: i.view ?? DEFAULT_VIEW,
        step_details: i.stepDetails ?? DEFAULT_STEP_DETAILS,
        // Sent because the Python surface sends it: bioblend pins include_terminal=true on
        // every listing, and leaving it to the index's own default would be one surface
        // silently hiding finished invocations the other one shows.
        include_terminal: true,
      },
    },
  });
  if (error || !data) throw classifyHttp(response.status, error);

  if (!Array.isArray(data)) {
    // A 200 whose body is not a list is Galaxy reporting a problem inside the response -- the
    // err_msg shape the other ops surface. Coercing it to [] would hand the caller a failure
    // dressed up as "Retrieved 0 workflow invocations".
    const errMsg =
      data && typeof data === "object" && "err_msg" in data
        ? String((data as { err_msg: unknown }).err_msg)
        : "the invocation index did not return a list";
    throw new GalaxyConnectionError(errMsg, response.status);
  }

  // Whatever the index sends back for the window it was asked for. Trimming the page here
  // would be a bound Python does not apply, and an agent comparing the two surfaces would
  // have no way to see where the missing rows went.
  return data as InvocationSummary[];
}

export const getInvocationsOp: Operation<typeof input, GetInvocationsResult> = {
  name: "get_invocations", // parity: mcp-server-galaxy-py get_invocations
  domain: "invocations",
  summary:
    "View one workflow invocation by id, or list invocations, optionally filtered by workflow or history.",
  input,
  run,
  project: (result, i) => {
    if (!Array.isArray(result)) {
      const inv = result as { id?: string; state?: string };
      return { message: `Invocation ${inv.id} state=${inv.state}` };
    }
    return {
      message: `Retrieved ${result.length} workflow invocation${result.length === 1 ? "" : "s"}`,
      ...(i.limit == null ? {} : { pagination: { limit: i.limit } }),
    };
  },
};

register(getInvocationsOp as AnyOperation);

// The declared defaults reach run() only through whatever parsed the input -- MCP's
// registerTool, the CLI's safeParse -- and a programmatic caller supplies none of them, so
// they are filled in here rather than asserted away with a cast. The return type is the
// check: add a .default() above without one here and this stops compiling.
const withDefaults = (i: In): InputOf<typeof input> => ({
  ...i,
  view: i.view ?? DEFAULT_VIEW,
  stepDetails: i.stepDetails ?? DEFAULT_STEP_DETAILS,
});

export const getInvocations = (i: In, ctx: GalaxyContext) =>
  runOperation(getInvocationsOp, withDefaults(i), ctx);
