import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { GalaxyValidationError, httpError } from "../errors";
import { paginationInfo, shrinkPaged, validatePagination, wirePagination, type Paged } from "./pagination";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

type ContentsIndex = GetJson<"/api/histories/{history_id}/contents">;
type ContentItem = ContentsIndex extends readonly (infer T)[] ? T : never;
export type HistoryContents = Paged<ContentItem>;

// Python's default. Neither surface has a ceiling here; what bounds a page is the
// output budget both measure on the way out.
const DEFAULT_LIMIT = 100;

/**
 * The orders this tool takes: each key Galaxy sorts the contents of one history by, ascending
 * or descending (managers/history_contents.py, parse_order_by), less history_id, which is the
 * same for every row here. Galaxy's own parser reads a bare key as descending, so the direction
 * is always spelled out, and a value outside this list is refused on both surfaces before
 * anything is sent.
 */
export const CONTENTS_ORDERS = [
  "hid-asc",
  "hid-dsc",
  "create_time-asc",
  "create_time-dsc",
  "update_time-asc",
  "update_time-dsc",
  "name-asc",
  "name-dsc",
  "extension-asc",
  "extension-dsc",
  "size-asc",
  "size-dsc",
] as const;

/** Galaxy's contents index with the matching count beside the page (api/history_contents.py). */
const WITH_STATS = "application/vnd.galaxy.history.contents.stats+json";

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
    .enum(CONTENTS_ORDERS)
    .default("hid-asc")
    .describe(
      "Sort order: hid, create_time, update_time, name, extension or size, followed by " +
        "'-asc' or '-dsc'. 'hid-asc' (default) is oldest first.",
    ),
};
type In = {
  historyId: string;
  limit?: number;
  offset?: number;
  deleted?: boolean;
  visible?: boolean;
  order?: (typeof CONTENTS_ORDERS)[number];
};

type Row = {
  collection_type?: string;
  type?: string;
  history_content_type?: string;
  dataset_id?: unknown;
};

/**
 * What Galaxy calls this item, filled in when it did not say.
 *
 * The Python tool adds the field rather than trusting it to be there, so an agent
 * can tell a dataset from a collection without knowing which serializer answered.
 */
const withContentType = (item: Row): Row => ({
  ...item,
  history_content_type:
    item.history_content_type ??
    (item.collection_type || item.type === "collection" ? "dataset_collection" : "dataset"),
});

/**
 * A row as a caller reads it: everything Galaxy sent except `dataset_id`.
 *
 * `id` is what every dataset-taking tool accepts. `dataset_id` names a different
 * object -- the Dataset under the history item -- in the same encoded id space, so
 * passed where `id` belongs it resolves rather than failing, to an unrelated item in
 * an unrelated history. No tool on either surface takes it.
 */
export function contentRow(item: unknown): unknown {
  if (!item || typeof item !== "object" || Array.isArray(item)) return item;
  const { dataset_id: _, ...row } = item as Row;
  return row;
}

async function run(i: In, ctx: GalaxyContext): Promise<HistoryContents> {
  const limit = i.limit ?? DEFAULT_LIMIT;
  const offset = i.offset ?? 0;
  const order = i.order ?? "hid-asc";
  // A limit of zero asks for a page of nothing and gets a walk that reports more to
  // come and never moves; a negative offset is a window nothing can describe. Both
  // surfaces refuse them, in the same words.
  validatePagination(limit, offset);
  if (!(CONTENTS_ORDERS as readonly string[]).includes(order)) {
    throw new GalaxyValidationError(
      `order must be one of ${CONTENTS_ORDERS.join(", ")} (got '${order}')`,
    );
  }
  // Galaxy filters, sorts and windows, and counts what matched in the same request,
  // so a history of thousands is read a page at a time and the total is still real.
  // v=dev is the index that honours all of it; the filters are ORM filters, which the
  // stats media type requires.
  const filters: [string, string][] = [];
  if (!i.deleted) filters.push(["deleted", "False"]);
  if (i.visible ?? true) filters.push(["visible", "True"]);
  const { data, error, response } = await ctx.client.GET("/api/histories/{history_id}/contents", {
    params: {
      path: { history_id: i.historyId },
      query: {
        v: "dev",
        limit,
        offset,
        order,
        q: filters.map(([field]) => field),
        qv: filters.map(([, value]) => value),
      },
      header: { accept: WITH_STATS },
    },
  });
  if (error || !data) throw httpError(response, error);
  const body = data as { contents?: Row[]; stats?: { total_matches?: number } };
  const items = (body.contents ?? []).map(withContentType) as ContentItem[];
  return {
    items,
    pagination: paginationInfo({
      total: body.stats?.total_matches ?? offset + items.length,
      returned: items.length,
      limit,
      offset,
      noun: "items",
    }),
  };
}

export const getHistoryContentsOp: Operation<typeof input, HistoryContents> = {
  name: "get_history_contents",
  domain: "histories",
  result: { kind: "object", fields: ["history_id", "contents"], paginated: true },
  summary: "List the datasets and collections in a history, one page at a time.",
  input,
  run,
  // This tool's data is an object rather than the bare page: the Python tool names
  // the history the contents came from beside them, and a caller holding one page
  // of several should not have to remember which history it asked about.
  // Python's _budgeted_page: a page of long names is cut until the text a model reads
  // fits, and the next page starts at the first item this one did not return.
  budget: {
    rows: (out) => out.items.length,
    shrink: (out, keep) => shrinkPaged(out, keep, "items"),
  },
  // This tool's data is an object rather than the bare page: the Python tool names
  // the history the contents came from beside them, and a caller holding one page
  // of several should not have to remember which history it asked about.
  project: (out, i) => ({
    data: { history_id: i.historyId, contents: out.items.map(contentRow) },
    // server.py, get_history_contents: the page size and nothing else. The total is
    // in the pagination block beside it, which is where this sentence leaves it.
    message: `Retrieved ${out.items.length} items from history`,
    count: out.items.length,
    pagination: wirePagination(out.pagination),
  }),
  // server.py, get_history_contents: a requests GET through make_get_request's headers,
  // and the same pair of sentences as get_history_details.
  failure: {
    shape: "raise-for-status",
    action: "Get history contents",
    context: (i) => ({ history_id: i.historyId }),
    sentence: (_text, status, i) =>
      status === 404
        ? `History ID '${i.historyId}' not found. Make sure to pass a valid history ID string.`
        : undefined,
  },
};

register(getHistoryContentsOp as AnyOperation);

export const getHistoryContents = (i: In, ctx: GalaxyContext) => runOperation(getHistoryContentsOp, i, ctx);
