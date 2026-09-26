import { z } from "zod";
import type { GalaxyContext } from "../context";
import { paginate, shrinkPaged, validatePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import { getHistories } from "./get-histories";
import type { AnyOperation, Operation } from "./types";

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
  const rows = histories.map((h) => ({ id: h.id ?? "", name: h.name ?? "" }));
  return paginate(rows, { limit, offset, noun: "histories" });
}

export const listHistoryIdsOp: Operation<typeof input, Paged<HistoryRef>> = {
  name: "list_history_ids",
  domain: "histories",
  result: { kind: "object", fields: ["items", "pagination"], paginated: true },
  summary: "List just the id and name of each history (compact picker for agents), one page at a time.",
  input,
  run,
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "histories"),
  },
  project: (out) => ({
    message: `${out.items.length} histor${out.items.length === 1 ? "y" : "ies"}`,
    pagination: out.pagination,
  }),
};

register(listHistoryIdsOp as AnyOperation);

export const listHistoryIds = (i: In, ctx: GalaxyContext) => runOperation(listHistoryIdsOp, i, ctx);
