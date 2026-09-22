import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyConnectionError } from "../errors";
import { stripRendered, type PageDetail } from "./pages-common";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  historyId: z.string().min(1).optional().describe("Encoded history id to attach the page to (makes it a notebook)"),
  title: z.string().min(1).optional().describe('Page title. A notebook left untitled is named "Untitled Notebook"'),
  content: z.string().optional().describe("Initial Galaxy-flavored markdown, with ENCODED ids in directives"),
  annotation: z.string().optional().describe("Optional annotation attached to the page"),
  slug: z.string().min(1).optional().describe("URL slug, lowercase/digits/hyphens. Required for a standalone report"),
};
type In = {
  historyId?: string;
  title?: string;
  content?: string;
  annotation?: string;
  slug?: string;
};

async function run(i: In, ctx: GalaxyContext): Promise<PageDetail> {
  // An empty string is not a history id, so it makes a report rather than a notebook.
  const historyId = i.historyId ? i.historyId : undefined;

  // A notebook needs neither: Galaxy names an untitled one and leaves its slug unset. A report
  // has to carry both, so say which one is missing rather than spend a round trip on a 400.
  if (historyId === undefined) {
    const missing: string[] = [];
    if (!i.title) missing.push("a title");
    if (!i.slug) missing.push("a unique slug");
    if (missing.length > 0) {
      throw new GalaxyConnectionError(
        `a standalone report needs ${missing.join(" and ")}; pass a historyId to create a notebook instead`,
        400,
      );
    }
  }

  // The endpoint defaults content_format to html; these ops speak markdown. The cast is needed
  // because the pinned CreatePagePayload has no history_id at all and marks title, slug and
  // content required -- a history-attached notebook supplies none of the three.
  const body: Record<string, unknown> = { content_format: "markdown" };
  if (historyId !== undefined) body["history_id"] = historyId;
  if (i.title !== undefined) body["title"] = i.title;
  if (i.content !== undefined) body["content"] = i.content;
  if (i.annotation !== undefined) body["annotation"] = i.annotation;
  if (i.slug !== undefined) body["slug"] = i.slug;

  const { data, error, response } = await ctx.client.POST("/api/pages", { body: body as never });
  if (error || !data) throw classifyHttp(response.status, error);
  return stripRendered(data as PageDetail, false);
}

export const createPageOp: Operation<typeof input, PageDetail> = {
  name: "create_page",
  domain: "pages",
  summary:
    "Create a markdown page. With historyId it is a notebook attached to that history; " +
    "without one it is a standalone report, which requires a title and a unique slug.",
  input,
  readOnly: false,
  run,
  project: (p) => ({ message: `Created page ${p.id} (${p.title})` }),
};

register(createPageOp as AnyOperation);

export const createPage = (i: In, ctx: GalaxyContext) => createPageOp.run(i, ctx);
