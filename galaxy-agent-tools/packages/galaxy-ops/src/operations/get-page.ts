import { z } from "zod";
import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
import { contentHash, stripRendered, type HashedPage, type PageDetail } from "./pages-common";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  pageId: z.string().min(1).describe("Encoded page id (from list_pages or create_page)"),
  includeRendered: z
    .boolean()
    .default(false)
    .describe("Also return `content`, the embed-expanded render. Can be large (default false)"),
};
type In = { pageId: string; includeRendered?: boolean };

async function run(i: In, ctx: GalaxyContext): Promise<HashedPage> {
  const { data, error, response } = await ctx.client.GET("/api/pages/{id}", {
    params: { path: { id: i.pageId } },
  });
  if (error || !data) throw httpError(response, error);
  // Hashed before the render is dropped: an HTML page's source is in `content`.
  const content_hash = contentHash(data as PageDetail);
  return { ...stripRendered(data as PageDetail, i.includeRendered ?? false), content_hash };
}

export const getPageOp: Operation<typeof input, HashedPage> = {
  name: "get_page",
  domain: "pages",
  summary:
    "Get a page and the latest revision's `content_editor` -- the editable Galaxy-flavored " +
    "markdown to pass back to update_page -- and `content_hash`, Galaxy's page hash of that " +
    "source, to tell on a later read whether anyone changed the page since.",
  input,
  run,
  // server.py, get_page: the id that was asked for. The title is in data.
  project: (_p, i) => ({ message: `Retrieved page '${i.pageId}'` }),
  // server.py, get_page: a raw GET plus raise_for_status.
  failure: { shape: "raise-for-status", action: "Get page", context: (i) => ({ page_id: i.pageId }) },
};

register(getPageOp as AnyOperation);

export const getPage = (i: In, ctx: GalaxyContext) => runOperation(getPageOp, i, ctx);
