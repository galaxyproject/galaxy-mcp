import { z } from "zod";
import type { components, GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError, GalaxyConnectionError } from "../errors";
import { pyFormatError } from "../python-failure";
import { pyStr } from "../python-values";
import { validatePagination } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

/** One row of GET /api/jobs: the 'collection' view's summary, or more for an admin view. */
export type JobSummary = GetJson<"/api/jobs">[number];
type JobOrderBy = components["schemas"]["JobIndexSortByEnum"];
type JobView = components["schemas"]["JobIndexViewEnum"];

const DEFAULT_ORDER_BY = "update_time";
const DEFAULT_VIEW = "collection";
const DEFAULT_LIMIT = 100;
const DEFAULT_OFFSET = 0;

// Python spells the four filters `T | None = None` and treats the null as "not supplied", so
// null is accepted here and means the same thing. The four it declares with a value -- order_by,
// view, limit, offset -- reject null on both surfaces and default the same way.
const input = {
  historyId: z
    .string()
    .nullish()
    .describe("Only jobs in this history (an encoded history id). Leave it unset for jobs from any history the user can see"),
  state: z
    .string()
    .nullish()
    .describe(
      "Only jobs in this state -- 'new', 'queued', 'running', 'ok', 'error', 'paused', 'deleted' and the rest " +
        "of Galaxy's job states. A comma-separated list matches any of them",
    ),
  dateRangeMin: z
    .string()
    .nullish()
    .describe(
      "Only jobs updated at or after this time, as an ISO 8601 date or datetime such as '2026-01-01' or " +
        "'2026-01-01T12:00:00'",
    ),
  dateRangeMax: z.string().nullish().describe("Only jobs updated at or before this time, in the same format"),
  orderBy: z
    .string()
    .default(DEFAULT_ORDER_BY)
    .describe("Sort by 'update_time' (default) or 'create_time', newest first"),
  view: z
    .string()
    .default(DEFAULT_VIEW)
    .describe(
      "'collection' (default) returns one small summary per job -- id, state, tool_id, exit_code, create_time, " +
        "update_time -- which is what a listing wants. 'admin_job_list' adds runner and handler detail and needs " +
        "an admin key",
    ),
  limit: z
    .number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(
      `Most jobs to return in one page (default ${DEFAULT_LIMIT}). Galaxy puts no upper cap on this, so this op ` +
        "does not either; keep it to what fits the output budget",
    ),
  offset: z
    .number()
    .int()
    .default(DEFAULT_OFFSET)
    .describe("Skip this many jobs (default 0). Page by raising it by limit until a page comes back shorter than limit"),
};

type In = {
  historyId?: string | null;
  state?: string | null;
  dateRangeMin?: string | null;
  dateRangeMax?: string | null;
  orderBy?: string;
  view?: string;
  limit?: number;
  offset?: number;
};

async function run(i: In, ctx: GalaxyContext): Promise<JobSummary[]> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? DEFAULT_OFFSET;
  // The floor every listing has and no ceiling: Galaxy's job index takes any limit, and the
  // Python tool passes one through unchanged, so this one does too.
  validatePagination(limit, offset);

  const { data, error, response } = await ctx.client.GET("/api/jobs", {
    params: {
      query: {
        limit,
        offset,
        // Truthiness, not ?? -- the other surface builds the request with `if history_id:`,
        // so a blank filter is no filter. Sent as "" it would reach Galaxy's encoded-id
        // validator and fail the whole listing over a value the caller plainly did not mean.
        history_id: i.historyId || null,
        // Galaxy takes a list or one comma-separated string. The Python surface sends the
        // string it was given, so this sends the same bytes rather than splitting it into
        // the repeated parameter the bindings type it as.
        state: (i.state || null) as unknown as string[] | null,
        date_range_min: i.dateRangeMin || null,
        date_range_max: i.dateRangeMax || null,
        order_by: (i.orderBy ?? DEFAULT_ORDER_BY) as JobOrderBy,
        view: (i.view ?? DEFAULT_VIEW) as JobView,
      },
    },
  });
  if (error || !data) throw httpError(response, error);

  if (!Array.isArray(data)) {
    // A 200 whose body is not a list is Galaxy reporting a problem inside the response -- the
    // err_msg shape the other ops surface. Coercing it to [] would hand the caller a failure
    // dressed up as "Retrieved 0 jobs". server.py, _refuse_error_body: worded here rather than
    // through the op's failure contract because it is not a failed request, and Python builds
    // it from an exception that carries no status, so the hint falls to the text search.
    const said = data && typeof data === "object" && "err_msg" in data;
    if (said) {
      const body = data as { err_msg: unknown; err_code?: unknown };
      throw new GalaxyConnectionError(
        pyFormatError(
          "List jobs",
          pyStr(body.err_msg),
          null,
          { err_code: "err_code" in body ? body.err_code : undefined },
          { statusIsKnown: false },
        ),
        response.status,
      );
    }
    throw new GalaxyConnectionError("the job index did not return a list", response.status);
  }

  // Whatever the index sends back for the window it was asked for. Galaxy windows this
  // listing itself, and trimming it here would be a bound Python does not apply.
  return data as JobSummary[];
}

export const listJobsOp: Operation<typeof input, JobSummary[]> = {
  name: "list_jobs", // parity: Python server list_jobs
  domain: "jobs",
  summary:
    "List jobs, optionally narrowed to one history, one state and a time window, a page at a time by limit and " +
    "offset. Galaxy reports no total for this index, so a page shorter than limit is the last one.",
  input,
  run,
  // server.py, list_jobs: a count of what came back, and no pagination -- the index reports
  // no total, so there is no window to describe, and the Python tool sends none either.
  project: (jobs) => ({
    message: `Retrieved ${jobs.length} jobs`,
    count: jobs.length,
  }),
  // server.py, list_jobs: bioblend's own GET, wrapped by format_error with the history in the
  // context. An error body under a 200 is refused at the throw site instead, because that one
  // carries a context of its own.
  failure: {
    shape: "bioblend-get",
    action: "List jobs",
    context: (i) => ({ history_id: i.historyId }),
  },
};

register(listJobsOp as AnyOperation);

// The declared defaults reach run() only through whatever parsed the input -- MCP's
// registerTool, the CLI's safeParse -- and a programmatic caller supplies none of them, so
// they are filled in here rather than asserted away with a cast. The return type is the
// check: add a .default() above without one here and this stops compiling.
const withDefaults = (i: In): InputOf<typeof input> => ({
  ...i,
  orderBy: i.orderBy ?? DEFAULT_ORDER_BY,
  view: i.view ?? DEFAULT_VIEW,
  limit: i.limit ?? DEFAULT_LIMIT,
  offset: i.offset ?? DEFAULT_OFFSET,
});

export const listJobs = (i: In, ctx: GalaxyContext) => runOperation(listJobsOp, withDefaults(i), ctx);
