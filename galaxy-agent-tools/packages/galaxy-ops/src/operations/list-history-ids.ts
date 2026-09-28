import { z } from "zod";
import type { GalaxyContext } from "../context";
import { paginate, shrinkPaged, validatePagination, wirePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import { getHistories } from "./get-histories";
import type { AnyOperation, InputOf, Operation } from "./types";

export interface HistoryRef { id: string; name: string; }

const DEFAULT_LIMIT = 100;
// Python's ceiling for this tool; a window one surface refuses the other refuses.
const MAX_LIMIT = 500;

const input = {
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Histories to return per page (default ${DEFAULT_LIMIT}, max ${MAX_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N histories. Pass pagination.nextOffset for the next page."),
};
type In = { limit?: number; offset?: number };

async function run(i: In, ctx: GalaxyContext): Promise<Paged<HistoryRef>> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  validatePagination(limit, offset, { maxLimit: MAX_LIMIT });
  // get_histories pages too, so take its whole first page: with no limit it puts
  // everything in one, which is what this listing needs to count and slice.
  const { items } = await getHistories({}, ctx);
  const histories = items as Array<{ id?: string; name?: string }>;
  // A history with no name is "Unnamed", not "": that is the word the other server
  // puts there, and `h.get("name", "Unnamed")` substitutes it for an ABSENT key
  // only -- a name Galaxy sent as null stays null, and an empty one stays empty.
  const rows = histories.map((h) => ({
    id: h.id ?? "",
    name: ("name" in h ? h.name : "Unnamed") as string,
  }));
  return paginate(rows, { limit, offset, noun: "histories" });
}

export const listHistoryIdsOp: Operation<typeof input, Paged<HistoryRef>> = {
  name: "list_history_ids",
  domain: "histories",
  summary: "List just the id and name of each history (compact picker for agents), one page at a time.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "histories"),
  },
  project: (out) => ({
    data: out.items,
    // The other server's sentence, word for word; the budget is measured on it too.
    message: out.pagination.total
      ? `Found ${out.items.length} of ${out.pagination.total} histories`
      : "No histories found",
    count: out.items.length,
    pagination: wirePagination(out.pagination),
  }),
  // server.py, list_history_ids: its own sentence over the same listing get_histories reads.
  failure: { shape: "bioblend-get", sentence: (text) => `Failed to list history IDs: ${text}` },
};

register(listHistoryIdsOp as AnyOperation);

// A library caller may leave the paged arguments out; run() applies the same
// defaults the schema declares for the parsed surface path.
export const listHistoryIds = (i: In, ctx: GalaxyContext) =>
  runOperation(listHistoryIdsOp, i as InputOf<typeof input>, ctx);
