import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
import { paginate, wirePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

type HistoryIndex = GetJson<"/api/histories">;
type History = HistoryIndex extends readonly (infer T)[] ? T : never;
export type Histories = Paged<History>;

const input = {
  // Null, not just absent: Python declares this one `int | None`, so a client that
  // sends null is asking for the default rather than making a mistake.
  limit: z.number()
    .int()
    .nullish()
    .describe("Max histories to return. Unset returns everything Galaxy sends, which on a busy account can be more than a client will accept."),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N histories. Pass pagination.nextOffset to walk to the following page."),
  name: z
    .string()
    .nullish()
    .describe(
      'Return only histories with exactly this name. The match is exact and case-sensitive, ' +
        'not a substring: name="RNA" does not match "RNA-seq analysis".',
    ),
};
type In = { limit?: number | null; offset?: number; name?: string | null };

async function run(i: In, ctx: GalaxyContext): Promise<Histories> {
  const offset = i.offset ?? 0;
  // Unwindowed, because the other surface ends up fetching everything anyway: it
  // asks Galaxy for the page and then asks again unpaged to count what matched.
  // One fetch, the same answer, and a total that is a count rather than a guess.
  const { data, error, response } = await ctx.client.GET("/api/histories", {
    params: { query: { limit: null, offset: null } },
  });
  if (error || !data) throw httpError(response, error);
  const all = data as History[];
  // Exact equality, because that is what bioblend does on the other surface:
  // `[h for h in histories if h["name"] == name]`, and it filters whenever the
  // caller supplied a name at all, an empty one included.
  const matching =
    i.name != null ? all.filter((h) => (h as { name?: string }).name === i.name) : all;
  // No limit means no window, and `||` rather than `??` on purpose: bioblend drops
  // a falsy limit with `if limit:`, so zero means unset there and means unset here.
  // One page then holds everything there is, which is what the other surface
  // answers when it has no window to report.
  return paginate(matching, {
    limit: i.limit || Math.max(matching.length, 1),
    offset,
    noun: "histories",
  }) as Histories;
}

export const getHistoriesOp: Operation<typeof input, Histories> = {
  name: "get_histories",
  domain: "histories",
  result: { kind: "list" },
  summary: "List the current user's histories (id, name, counts). Optional exact-name filter.",
  input,
  run,
  project: (out, i) => {
    const returned = out.items.length;
    return {
      data: out.items,
      // Two sentences, because server.py's get_histories has two branches: a limit
      // describes the window it returned out of what matched, and no limit has no
      // window to describe. Neither of them is singular for one row.
      message:
        i.limit == null
          ? `Retrieved ${returned} histories`
          : `Retrieved ${returned} of ${out.pagination.total} histories`,
      count: returned,
      // No limit, no window to describe: the Python tool returns no pagination
      // block at all on that branch, and neither does this one. run() still
      // windows, because a library caller wants a Paged either way.
      pagination: i.limit == null ? null : wirePagination(out.pagination),
    };
  },
  // server.py, get_histories: its own sentence, and its own advice after it.
  failure: {
    shape: "bioblend-get",
    sentence: (text) =>
      `Failed to get histories: ${text}. Check your connection to Galaxy and that you have ` +
      "permission to view histories.",
  },
};

register(getHistoriesOp as AnyOperation);

export const getHistories = (i: In, ctx: GalaxyContext) => runOperation(getHistoriesOp, i, ctx);
