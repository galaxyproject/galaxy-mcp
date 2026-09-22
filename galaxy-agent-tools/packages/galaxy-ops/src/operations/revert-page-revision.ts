import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyPost } from "../legacy";
import {
  withEditableContent,
  type PageRevisionDetails,
  type PageRevisionResponse,
} from "./pages-common";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  pageId: z.string().min(1).describe("Encoded page id"),
  revisionId: z.string().min(1).describe("Encoded id of the revision to restore"),
};
type In = { pageId: string; revisionId: string };

async function run(i: In, ctx: GalaxyContext): Promise<PageRevisionDetails> {
  // Off-schema: the revisions endpoints arrive with the 26.1 bindings. Same content_editor
  // fallback as get_page_revision -- the restored revision comes back through the same model.
  const revision = await legacyPost<PageRevisionResponse>(
    ctx,
    "/api/pages/{id}/revisions/{revision_id}/revert",
    { params: { path: { id: i.pageId, revision_id: i.revisionId } } },
  );
  return withEditableContent(revision);
}

export const revertPageRevisionOp: Operation<typeof input, PageRevisionDetails> = {
  name: "revert_page_revision",
  domain: "pages",
  summary:
    "Roll a page back to an earlier revision. The history is append-only: this writes a NEW " +
    "revision from the old content, tagged edit_source=restore, and deletes nothing. Its " +
    "`content_editor` follows the same rule as get_page_revision -- filled from `content` " +
    "where the server sends none, and that form has its embeds expanded.",
  input,
  readOnly: false,
  run,
  project: (rev, i) => ({ message: `Reverted page ${i.pageId} to revision ${i.revisionId} as ${rev.id}` }),
};

register(revertPageRevisionOp as AnyOperation);

export const revertPageRevision = (i: In, ctx: GalaxyContext) => revertPageRevisionOp.run(i, ctx);
