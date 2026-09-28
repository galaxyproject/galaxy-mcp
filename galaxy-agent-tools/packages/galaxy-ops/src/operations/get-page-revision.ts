import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import {
  withEditableContent,
  type PageRevisionDetails,
  type PageRevisionResponse,
} from "./pages-common";
import { register, runOperation } from "./registry";
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
    "content_editor on a revision after 26.1, so check `content_editor_source`: " +
    '"content" means the server sent none and the editable text is the expanded form.',
  input,
  requires: { galaxy: ">=26.1" },
  run,
  // server.py, get_page_revision: the two ids that were asked for. Where the
  // editable text came from is in data, under content_editor_source, which is what
  // the docstring tells a caller to read.
  project: (_rev, i) => ({ message: `Retrieved revision '${i.revisionId}' of page '${i.pageId}'` }),
  // server.py, get_page_revision: a raw GET plus raise_for_status.
  failure: {
    shape: "raise-for-status",
    action: "Get page revision",
    context: (i) => ({ page_id: i.pageId, revision_id: i.revisionId }),
  },
};

register(getPageRevisionOp as AnyOperation);

export const getPageRevision = (i: In, ctx: GalaxyContext) => runOperation(getPageRevisionOp, i, ctx);
