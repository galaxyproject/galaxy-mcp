import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { envelopeFact, readFact, recordFact } from "./envelope-facts";
import { paginationInfo, wirePagination } from "./pagination";
import type { PageSummary } from "./pages-common";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

const DEFAULT_LIMIT = 100;

const input = {
  historyId: z.string().min(1).optional().describe("Encoded history id; lists only that history's notebooks"),
  search: z.string().optional().describe("Freetext filter over title, slug, tag and owner"),
  limit: z.number().int().default(DEFAULT_LIMIT).describe(`Max pages to return (default ${DEFAULT_LIMIT})`),
  offset: z.number().int().default(0).describe("Skip the first N"),
  showPublished: z.boolean().default(false).describe("Also include pages published by other users (default false)"),
  showShared: z.boolean().default(false).describe("Also include pages shared with the user (default false)"),
};
type In = {
  historyId?: string;
  search?: string;
  limit?: number;
  offset?: number;
  showPublished?: boolean;
  showShared?: boolean;
};

/**
 * How many pages matched the filters, for the window run() just returned.
 *
 * Galaxy reports the total on a `total_matches` RESPONSE HEADER rather than in the
 * body, and run() returns the body -- an array of pages, which is what every
 * library caller has always got and what it goes on returning. So the header is
 * recorded beside the call and the projection of that same call reads it back.
 */
const totalMatches = envelopeFact<number>("list_pages.total_matches");

async function run(i: In, ctx: GalaxyContext): Promise<PageSummary[]> {
  const { data, error, response } = await ctx.client.GET("/api/pages", {
    params: {
      query: {
        limit: i.limit ?? DEFAULT_LIMIT,
        offset: i.offset ?? 0,
        // The index's own defaults (show_own and show_published both on) would hand an agent
        // every published page on the server, so send each visibility flag explicitly.
        show_own: true,
        show_published: i.showPublished ?? false,
        show_shared: i.showShared ?? false,
        search: i.search ?? null,
        // history_id is a 26.1 filter the pinned bindings do not carry, so it goes in through
        // its own cast. That is belt and braces rather than a guard: openapi-fetch infers the
        // init generically, so an unknown query key compiles either way. The VALUES above are
        // checked.
        ...(i.historyId === undefined ? {} : ({ history_id: i.historyId } as never)),
      },
    },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  const pages = data as PageSummary[];
  // Number(null) is 0, which would report an empty server rather than an absent
  // header, so the missing case is checked before the parse. No header means the
  // page is all we know about, which is the fallback the Python tool takes too.
  const header = response.headers.get("total_matches");
  const total = header === null ? Number.NaN : Number(header);
  recordFact(ctx, totalMatches, Number.isInteger(total) ? total : pages.length);
  return pages;
}

export const listPagesOp: Operation<typeof input, PageSummary[]> = {
  name: "list_pages",
  domain: "pages",
  summary:
    "List Galaxy pages (markdown notebooks and reports) the user can see. " +
    "Pass historyId to list only that history's notebooks. The history filter is what needs " +
    "26.1: an older server ignores it and answers with every page instead of that history's.",
  input,
  requires: { galaxy: ">=26.1" },
  run,
  project: (pages, i, facts) => ({
    // server.py, list_pages: the rows in hand. The total is in the pagination block.
    message: `Retrieved ${pages.length} pages`,
    count: pages.length,
    // The shared describer, like every other listing. This one windows server-side
    // and reads its total from a header, which the describer already has a line for:
    // rows in hand prove a floor, so a server that under-reports cannot make the page
    // it just sent disappear. It used to be built by hand here to match a hand-built
    // block on the other server -- no helper text at all, and an advance by the limit
    // asked for rather than by what came back. That block is gone now.
    pagination: wirePagination(
      paginationInfo({
        total: readFact(facts, totalMatches) ?? pages.length,
        returned: pages.length,
        limit: i.limit ?? DEFAULT_LIMIT,
        offset: i.offset ?? 0,
        noun: "pages",
      }),
    ),
  }),
};

register(listPagesOp as AnyOperation);

// A library caller may leave the defaulted arguments out; run() applies the same
// values the schema declares for the parsed surface path.
export const listPages = (i: In, ctx: GalaxyContext) =>
  runOperation(listPagesOp, i as InputOf<typeof input>, ctx);
