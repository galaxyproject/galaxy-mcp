import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import type { PageRevisionSummary } from "./pages-common";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  pageId: z.string().min(1).describe("Encoded page id"),
  sortDesc: z.boolean().optional().describe("Newest first when true (default oldest first)"),
};
type In = { pageId: string; sortDesc?: boolean };

async function run(i: In, ctx: GalaxyContext): Promise<PageRevisionSummary[]> {
  // Off-schema: the revisions endpoints arrive with the 26.1 bindings.
  return legacyGet<PageRevisionSummary[]>(ctx, "/api/pages/{id}/revisions", {
    params: { path: { id: i.pageId }, query: { sort_desc: i.sortDesc ?? false } },
  });
}

export const listPageRevisionsOp: Operation<typeof input, PageRevisionSummary[]> = {
  name: "list_page_revisions",
  domain: "pages",
  summary:
    "List a page's revision history. A revision carries an edit_source of " +
    '"user", "agent" or "restore" where Galaxy recorded one -- a page\'s first revision ' +
    "has none.",
  input,
  requires: { galaxy: ">=26.1" },
  run,
  project: (revs, i) => ({ message: `${revs.length} revision(s) for page ${i.pageId}` }),
};

register(listPageRevisionsOp as AnyOperation);

export const listPageRevisions = (i: In, ctx: GalaxyContext) => runOperation(listPageRevisionsOp, i, ctx);
