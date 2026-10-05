import { z } from "zod";
import type { components, GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError, GalaxyConnectionError } from "../errors";
import { pyFormatError } from "../python-failure";
import { pyStr } from "../python-values";
import { withOutcome, type JobStates } from "../invocation-outcome";
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
    .describe(
      "Include each step's jobs. Applies to one invocation by id, and in list mode only when view is 'element'",
    ),
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

/**
 * How many of an invocation's jobs are in each state, or undefined when the summary cannot be
 * read: the invocation is still worth answering with, but not with an outcome guessed from no
 * jobs.
 */
async function jobStates(ctx: GalaxyContext, invocationId: string): Promise<JobStates | undefined> {
  let states: unknown;
  try {
    const { data } = await ctx.client.GET("/api/invocations/{invocation_id}/jobs_summary", {
      params: { path: { invocation_id: invocationId } },
    });
    states = (data as { states?: unknown } | undefined)?.states;
  } catch {
    return undefined;
  }
  if (!states || typeof states !== "object" || Array.isArray(states)) return undefined;
  return Object.fromEntries(
    Object.entries(states).filter(([, n]) => typeof n === "number" && Number.isInteger(n)),
  ) as JobStates;
}

async function run(i: In, ctx: GalaxyContext): Promise<GetInvocationsResult> {
  if (i.invocationId) {
    // Galaxy answers a single invocation with every step's jobs list empty unless
    // step_details is set. Sent only when asked for, as the other server does, so the
    // default request is unchanged.
    const { data, error, response } = await ctx.client.GET("/api/invocations/{invocation_id}", {
      params: {
        path: { invocation_id: i.invocationId },
        ...(i.stepDetails ? { query: { step_details: true } } : {}),
      },
    });
    if (error || !data) throw httpError(response, error);
    return withOutcome(data as InvocationDetail, await jobStates(ctx, i.invocationId));
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
  if (error || !data) throw httpError(response, error);

  if (!Array.isArray(data)) {
    // A 200 whose body is not a list is Galaxy reporting a problem inside the response -- the
    // err_msg shape the other ops surface. Coercing it to [] would hand the caller a failure
    // dressed up as "Retrieved 0 workflow invocations".
    //
    // server.py, _refuse_error_body: this one is worded here rather than through the op's
    // failure contract, because it is not a failed request -- it is a reply that succeeded
    // and said no. Python builds it from an exception it made out of err_msg, which carries
    // no status field, so the hint falls to the text search that exception text gets.
    const said = data && typeof data === "object" && "err_msg" in data;
    if (said) {
      const body = data as { err_msg: unknown; err_code?: unknown };
      throw new GalaxyConnectionError(
        pyFormatError(
          "Get workflow invocations",
          pyStr(body.err_msg),
          null,
          { err_code: "err_code" in body ? body.err_code : undefined },
          { statusIsKnown: false },
        ),
        response.status,
      );
    }
    throw new GalaxyConnectionError("the invocation index did not return a list", response.status);
  }

  // Whatever the index sends back for the window it was asked for. Trimming the page here
  // would be a bound Python does not apply, and an agent comparing the two surfaces would
  // have no way to see where the missing rows went. A listing carries Galaxy's scheduling
  // state only: an outcome needs each invocation's jobs, one request each, so it is read for
  // one invocation by id rather than for every row of a page.
  return data as InvocationSummary[];
}

export const getInvocationsOp: Operation<typeof input, GetInvocationsResult> = {
  name: "get_invocations", // parity: mcp-server-galaxy-py get_invocations
  domain: "invocations",
  summary:
    "View one workflow invocation by id, or list invocations, optionally filtered by workflow or " +
    "history. Galaxy's `state` describes scheduling only, not what the jobs did. One invocation " +
    "by id also carries `job_states`, how many of its jobs are in each state, and `outcome` -- " +
    "failing, failed, cancelled, completed, or Galaxy's state while that is all the jobs say -- " +
    "for what the run amounted to; both are left out when Galaxy's jobs summary cannot be read. " +
    "Get an invocation by id to learn its outcome.",
  input,
  run,
  project: (result, i) => {
    // server.py, get_invocations: the id that was asked for on the detail branch --
    // the state is in data -- and a count with the plural written either way on the
    // listing branch.
    if (!Array.isArray(result)) return { message: `Retrieved invocation '${i.invocationId}'` };
    // A count of what came back, and no pagination: Galaxy windows this index
    // server-side and reports no total, so there is no window to describe. The
    // Python tool answers the same way, limit or no limit.
    return {
      message: `Retrieved ${result.length} workflow invocations`,
      count: result.length,
    };
  },
  // server.py, get_invocations: its own sentence. An error body under a 200 is refused at the
  // throw site instead, because that one is format_error's with a context of its own.
  failure: {
    shape: "bioblend-get",
    sentence: (text) => `Failed to get workflow invocations: ${text}`,
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
