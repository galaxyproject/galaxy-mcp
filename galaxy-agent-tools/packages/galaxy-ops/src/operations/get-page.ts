import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { stripRendered, type PageDetail } from "./pages-common";
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

async function run(i: In, ctx: GalaxyContext): Promise<PageDetail> {
  const { data, error, response } = await ctx.client.GET("/api/pages/{id}", {
    params: { path: { id: i.pageId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  return stripRendered(data as PageDetail, i.includeRendered ?? false);
}

export const getPageOp: Operation<typeof input, PageDetail> = {
  name: "get_page",
  domain: "pages",
  summary:
    "Get a page and the latest revision's `content_editor` -- the editable Galaxy-flavored " +
    "markdown to pass back to update_page.",
  input,
  run,
  project: (p) => ({ message: `Page ${p.id} (${p.title})` }),
};

register(getPageOp as AnyOperation);

export const getPage = (i: In, ctx: GalaxyContext) => runOperation(getPageOp, i, ctx);
