import { z } from "zod";
import type { components, GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError, GalaxyConnectionError, GalaxyNotFoundError, GalaxyValidationError } from "../errors";
import { pyFormatError } from "../python-failure";
import { pyStr } from "../python-values";
import { isEncodedId, isThisRecord, notAGalaxyId, notThatRecord } from "./encoded-id";
import { validatePagination } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

export type InvocationDetail = components["schemas"]["WorkflowInvocationElementView"];
export type InvocationSummary = GetJson<"/api/invocations">[number];
// What the index's sort_by query parameter is typed as (InvocationSortByEnum). The op takes a
// plain string, as the Python tool does, and Galaxy refuses anything outside the enum itself.
type InvocationSortBy = components["schemas"]["InvocationSortByEnum"];

const DEFAULT_VIEW = "collection";
const DEFAULT_STEP_DETAILS = false;
const DEFAULT_OFFSET = 0;
const DEFAULT_INCLUDE_TERMINAL = true;
// Galaxy's own ceiling: /api/invocations declares limit le=100 and answers 400 above it.
// server.py, MAX_PAGE_SIZE["get_invocations"] -- the same number, so the refusal reads the same.
const MAX_LIMIT = 100;

// Python spells the id, the filters, the limit and the sort as `T | None = None` and treats the
// null as "not supplied", so null is accepted here and means the same thing. The parameters
// Python declares without a null branch -- view, step_details, offset and include_terminal --
// reject it on both surfaces.
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
      "List mode: most invocations to return, at most 100 -- Galaxy's invocation index serves no " +
        "larger page, so page with offset to see more. Left out, the parameter is not sent and " +
        "Galaxy's own index default applies -- which is a page, not everything (26.1 returns 20). " +
        "Pass a limit to know what you are getting.",
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
  offset: z
    .number()
    .int()
    .default(DEFAULT_OFFSET)
    .describe(
      "List mode: how many invocations to skip before the page starts, for walking a listing longer " +
        "than one page of limit",
    ),
  sortBy: z
    .string()
    .nullish()
    .describe(
      "List mode: order the listing by 'create_time' or 'update_time'. Left out, Galaxy's own order " +
        "applies, which is newest first",
    ),
  sortDesc: z
    .boolean()
    .nullish()
    .describe(
      "List mode, with sortBy: true orders newest first and false oldest first. Left out, Galaxy's " +
        "own direction applies",
    ),
  includeTerminal: z
    .boolean()
    .default(DEFAULT_INCLUDE_TERMINAL)
    .describe(
      "List mode: whether invocations that have finished -- scheduled, failed or cancelled -- are " +
        "listed. false keeps only the ones still in flight",
    ),
};

type In = {
  invocationId?: string | null;
  workflowId?: string | null;
  historyId?: string | null;
  limit?: number | null;
  view?: string;
  stepDetails?: boolean;
  offset?: number;
  sortBy?: string | null;
  sortDesc?: boolean | null;
  includeTerminal?: boolean;
};

/** One invocation when an id was given, the filtered listing when it was not. */
export type GetInvocationsResult = InvocationDetail | InvocationSummary[];

/**
 * server.py, _refuse_error_body: a 200 whose body is Galaxy reporting a problem -- the
 * err_msg shape -- is refused rather than read as the thing asked for. Worded here rather
 * than through the op's failure contract, because it is not a failed request: it is a reply
 * that succeeded and said no. Python builds it from an exception it made out of err_msg,
 * which carries no status field, so the hint falls to the text search that exception text
 * gets.
 */
function refuseErrorBody(data: unknown, status: number): void {
  if (data && typeof data === "object" && "err_msg" in data) {
    const body = data as { err_msg: unknown; err_code?: unknown };
    throw new GalaxyConnectionError(
      pyFormatError(
        "Get workflow invocations",
        pyStr(body.err_msg),
        null,
        { err_code: "err_code" in body ? body.err_code : undefined },
        { statusIsKnown: false },
      ),
      status,
    );
  }
}

async function run(i: In, ctx: GalaxyContext): Promise<GetInvocationsResult> {
  if (i.invocationId) {
    // Not an id, not a request: see encoded-id.ts for what a stray "." would turn into.
    // A refusal of our own, worded whole, so no facts for the failure contract.
    if (!isEncodedId(i.invocationId)) {
      throw new GalaxyNotFoundError(notAGalaxyId("Invocation", i.invocationId, "get_invocations()"));
    }
    // Galaxy answers a single invocation with every step's jobs list empty unless
    // step_details is set. Sent only when asked for, as the other server does, so the
    // default request is unchanged.
    const { data, error, response } = await ctx.client.GET("/api/invocations/{invocation_id}", {
      params: {
        path: { invocation_id: i.invocationId },
        ...(i.stepDetails ? { query: { step_details: true } } : {}),
      },
    });
    if (error || !data) {
      const failed = httpError(response, error);
      // An id Galaxy cannot decode is answered 400, not 404, and to a caller it is the same
      // thing: there is no invocation by that name, and retrying will not make one. Reclassified
      // here so the kind, and the CLI's exit code, say not-found either way; the reply's facts
      // travel along, because the sentence is still worded from them.
      if (response?.status === 400) {
        const notFound = new GalaxyNotFoundError(failed.message);
        notFound.http = failed.http;
        throw notFound;
      }
      throw failed;
    }
    refuseErrorBody(data, response.status);
    // The same check get_job_details makes: the reply has to be this invocation's record,
    // not merely a 200 from somewhere under /api/invocations.
    if (!isThisRecord(data, i.invocationId)) {
      throw new GalaxyConnectionError(notThatRecord("invocation", i.invocationId), response.status);
    }
    return data as InvocationDetail;
  }

  // server.py, get_invocations: the window is checked before anything is sent. Galaxy's
  // index refuses limit > 100 with a 400 about validation; this refusal names the cap and
  // the way past it. A negative offset is refused with or without a limit.
  const offset = i.offset ?? DEFAULT_OFFSET;
  if (i.limit != null) {
    validatePagination(i.limit, offset, { maxLimit: MAX_LIMIT });
  } else if (offset < 0) {
    throw new GalaxyValidationError(`offset must be 0 or greater (got ${offset})`);
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
        // Sent because the Python surface sends it: bioblend pins include_terminal on every
        // listing (true unless told otherwise), and leaving it to the index's own default
        // would be one surface silently hiding finished invocations the other one shows.
        include_terminal: i.includeTerminal ?? DEFAULT_INCLUDE_TERMINAL,
        // The rest only when asked for, as the other server sends them: a zero offset and an
        // unset sort are left off, so the default listing is the request it always was.
        offset: offset || null,
        sort_by: (i.sortBy || null) as InvocationSortBy | null,
        sort_desc: i.sortDesc ?? undefined,
      },
    },
  });
  if (error || !data) throw httpError(response, error);

  if (!Array.isArray(data)) {
    // A 200 whose body is not a list is Galaxy reporting a problem inside the response -- the
    // err_msg shape the other ops surface. Coercing it to [] would hand the caller a failure
    // dressed up as "Retrieved 0 workflow invocations".
    refuseErrorBody(data, response.status);
    throw new GalaxyConnectionError("the invocation index did not return a list", response.status);
  }

  // Whatever the index sends back for the window it was asked for. Trimming the page here
  // would be a bound Python does not apply, and an agent comparing the two surfaces would
  // have no way to see where the missing rows went.
  return data as InvocationSummary[];
}

export const getInvocationsOp: Operation<typeof input, GetInvocationsResult> = {
  name: "get_invocations", // parity: Python server get_invocations
  domain: "invocations",
  summary:
    "View one workflow invocation by id, or list invocations, optionally filtered by workflow or history.",
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
  offset: i.offset ?? DEFAULT_OFFSET,
  includeTerminal: i.includeTerminal ?? DEFAULT_INCLUDE_TERMINAL,
});

export const getInvocations = (i: In, ctx: GalaxyContext) =>
  runOperation(getInvocationsOp, withDefaults(i), ctx);
