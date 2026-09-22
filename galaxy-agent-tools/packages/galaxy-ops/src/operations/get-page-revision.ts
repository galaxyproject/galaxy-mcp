import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import {
  withEditableContent,
  type PageRevisionDetails,
  type PageRevisionResponse,
} from "./pages-common";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  pageId: z.string().min(1).describe("Encoded page id"),
  revisionId: z.string().min(1).describe("Encoded revision id (from list_page_revisions)"),
};
type In = { pageId: string; revisionId: string };

async function run(i: In, ctx: GalaxyContext): Promise<PageRevisionDetails> {
  // Off-schema: the revisions endpoints arrive with the 26.1 bindings. Keep both forms so
  // callers can edit content_editor without losing the expanded content needed for export.
  const revision = await legacyGet<PageRevisionResponse>(
    ctx,
    "/api/pages/{id}/revisions/{revision_id}",
    { params: { path: { id: i.pageId, revision_id: i.revisionId } } },
  );
  return withEditableContent(revision);
}

export const getPageRevisionOp: Operation<typeof input, PageRevisionDetails> = {
  name: "get_page_revision",
  domain: "pages",
  summary:
    "Get one page revision. Edit `content_editor` and pass it to update_page; `content` is " +
    "the same document with its embeds expanded for export. Galaxy began sending " +
    "content_editor on a revision after 26.1; against an older server it is filled from " +
    "`content`, so both come back expanded.",
  input,
  run,
  project: (rev) => ({ message: `Revision ${rev.id} of page ${rev.page_id} (${rev.edit_source ?? "unknown"})` }),
};

register(getPageRevisionOp as AnyOperation);

export const getPageRevision = (i: In, ctx: GalaxyContext) => getPageRevisionOp.run(i, ctx);
