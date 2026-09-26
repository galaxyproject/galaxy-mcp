import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { paginate, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

type ContentsIndex = GetJson<"/api/histories/{history_id}/contents">;
type ContentItem = ContentsIndex extends readonly (infer T)[] ? T : never;
export type HistoryContents = Paged<ContentItem>;

// Python's default. Neither surface has a ceiling here and neither budgets the
// result: this op is not in MAX_PAGE_SIZE and does not go through the budgeted
// page, so a history of long names can return a page too big for a client to pass
// through, on both surfaces alike.
const DEFAULT_LIMIT = 100;

const input = {
  historyId: z.string().describe("Encoded history id"),
  limit: z.number()
    .int()
    .default(DEFAULT_LIMIT)
    .describe(`Items to return per page (default ${DEFAULT_LIMIT})`),
  offset: z.number()
    .int()
    .default(0)
    .describe("Skip the first N items. Pass pagination.nextOffset to walk to the following page."),
  deleted: z
    .boolean()
    .default(false)
    .describe("Include deleted items alongside active items (default false)."),
  visible: z.boolean().default(true).describe("Only visible items by default; set false to include hidden items too."),
  order: z
    .string()
    .default("hid-asc")
    .describe(
      "Sort order. 'hid-asc' (default, oldest first), 'hid-dsc', 'create_time-asc', " +
        "'create_time-dsc', 'update_time-dsc', 'name-asc'.",
    ),
};
type In = {
  historyId: string;
  limit?: number;
  offset?: number;
  deleted?: boolean;
  visible?: boolean;
  order?: string;
};

type Sortable = {
  hid?: number;
  name?: string;
  create_time?: string;
  update_time?: string;
  deleted?: boolean;
  visible?: boolean;
  history_content_type?: string;
  collection_type?: string;
  type?: string;
};

/**
 * What Galaxy calls this item, filled in when it did not say.
 *
 * The Python tool adds the field rather than trusting it to be there, so an agent
 * can tell a dataset from a collection without knowing which serializer answered.
 */
const withContentType = (item: Sortable): Sortable => ({
  ...item,
  history_content_type:
    item.history_content_type ??
    (item.collection_type || item.type === "collection" ? "dataset_collection" : "dataset"),
});

/**
 * The sort key `order` selects, and which direction.
 *
 * Copied from the Python tool, prefix test and all: anything starting hid sorts by
 * hid, create_time by create_time, update_time by update_time, name by name, and
 * anything else by hid. Only the `-dsc` suffix reverses. So "hid" is ascending and
 * "nonsense-asc" is hid ascending, where Galaxy's own index would answer "hid"
 * descending and refuse the nonsense with a 400 -- which is the reason this op
 * windows the list itself instead of asking Galaxy to.
 */
function sortKey(order: string): (item: Sortable) => number | string {
  if (order.startsWith("hid")) return (item) => item.hid ?? 0;
  if (order.startsWith("create_time")) return (item) => item.create_time ?? "";
  if (order.startsWith("update_time")) return (item) => item.update_time ?? "";
  if (order.startsWith("name")) return (item) => item.name ?? "";
  return (item) => item.hid ?? 0;
}

async function run(i: In, ctx: GalaxyContext): Promise<HistoryContents> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  const order = i.order ?? "hid-asc";
  // The whole contents index, unfiltered and unwindowed, which is what
  // show_history(contents=True) fetches on the other surface: no query parameters
  // at all. Everything below happens here, so both surfaces answer the same way
  // and this one can report a real total.
  const { data, error, response } = await ctx.client.GET("/api/histories/{history_id}/contents", {
    params: { path: { history_id: i.historyId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);

  const all = (Array.isArray(data) ? (data as Sortable[]) : []).map(withContentType);
  let matching = all;
  if (!i.deleted) matching = matching.filter((item) => !(item.deleted ?? false));
  if (i.visible ?? true) matching = matching.filter((item) => item.visible ?? true);

  const key = sortKey(order);
  const reverse = order.endsWith("-dsc");
  const sorted = [...matching].sort((a, b) => {
    const [x, y] = [key(a), key(b)];
    const ordered = x < y ? -1 : x > y ? 1 : 0;
    return reverse ? -ordered : ordered;
  });

  return paginate(sorted, { limit, offset, noun: "items" }) as HistoryContents;
}

export const getHistoryContentsOp: Operation<typeof input, HistoryContents> = {
  name: "get_history_contents",
  domain: "histories",
  result: { kind: "object", fields: ["items", "pagination"], paginated: true },
  summary: "List the datasets and collections in a history, one page at a time.",
  input,
  run,
  project: (out) => ({
    message: `${out.items.length} of ${out.pagination.total} item(s)`,
    pagination: out.pagination,
  }),
};

register(getHistoryContentsOp as AnyOperation);

export const getHistoryContents = (i: In, ctx: GalaxyContext) => runOperation(getHistoryContentsOp, i, ctx);
