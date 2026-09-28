import { z } from "zod";
import type { GalaxyContext } from "../context";
import { httpError, GalaxyConnectionError } from "../errors";
import { stripRendered, type PageDetail } from "./pages-common";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  pageId: z.string().min(1).describe("Encoded page id"),
  content: z
    .string()
    .optional()
    .describe("New Galaxy-flavored markdown, with ENCODED ids in directives. Omit to leave unchanged"),
  title: z.string().min(1).optional().describe("New title. Omit to leave unchanged"),
};
type In = { pageId: string; content?: string; title?: string };

async function run(i: In, ctx: GalaxyContext): Promise<PageDetail> {
  if (i.content === undefined && i.title === undefined) {
    throw new GalaxyConnectionError("nothing to update", 400);
  }

  // edit_source attributes the new revision to the agent rather than to a person; Galaxy only
  // writes a revision when content changes, so a title-only edit records nothing. The cast is
  // needed because the pinned UpdatePagePayload has neither edit_source nor content, and marks
  // slug and title required -- a content-only edit sends neither.
  const body: Record<string, unknown> = { edit_source: "agent" };
  if (i.content !== undefined) body["content"] = i.content;
  if (i.title !== undefined) body["title"] = i.title;

  const { data, error, response } = await ctx.client.PUT("/api/pages/{id}", {
    params: { path: { id: i.pageId } },
    body: body as never,
  });
  if (error || !data) throw httpError(response, error);
  return stripRendered(data as PageDetail, false);
}

export const updatePageOp: Operation<typeof input, PageDetail> = {
  name: "update_page",
  domain: "pages",
  summary:
    "Update a page's content or title. New content creates a revision tagged " +
    "edit_source=agent; a title-only change does not. The page keeps its content_format, so " +
    "sending markdown to a page authored as HTML stores it under the wrong format.",
  input,
  requires: { galaxy: ">=26.1" },
  readOnly: false,
  run,
  // server.py, update_page: the page that was updated, and not which fields moved.
  // A caller knows what it sent; the updated record is in data.
  project: (_p, i) => ({ message: `Updated page '${i.pageId}'` }),
  // server.py, update_page: a bioblend write.
  failure: {
    shape: "bioblend-write",
    action: "Update page",
    context: (i) => ({ page_id: i.pageId }),
  },
};

register(updatePageOp as AnyOperation);

export const updatePage = (i: In, ctx: GalaxyContext) => runOperation(updatePageOp, i, ctx);
