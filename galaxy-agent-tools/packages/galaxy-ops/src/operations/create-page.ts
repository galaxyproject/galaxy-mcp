import { z } from "zod";
import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
import { stripRendered, type PageDetail } from "./pages-common";
import { pyGet, pyStr } from "../python-values";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  historyId: z.string().min(1).nullish().describe("Encoded history id to attach the page to (makes it a notebook)"),
  title: z.string().min(1).nullish().describe('Page title. A notebook left untitled is named "Untitled Notebook"'),
  content: z.string().nullish().describe("Initial Galaxy-flavored markdown, with ENCODED ids in directives"),
  annotation: z.string().nullish().describe("Optional annotation attached to the page"),
  slug: z.string().min(1).nullish().describe("URL slug, lowercase/digits/hyphens. Required for a standalone report"),
};
type In = {
  historyId?: string | null;
  title?: string | null;
  content?: string | null;
  annotation?: string | null;
  slug?: string | null;
};

async function run(i: In, ctx: GalaxyContext): Promise<PageDetail> {
  // An empty string is not a history id, so it makes a report rather than a notebook.
  const historyId = i.historyId ? i.historyId : undefined;

  // A standalone report does need a title and a unique slug, and this used to refuse one
  // without them rather than spend a round trip on a 400. The other server does not: it
  // sends whatever it was given and lets Galaxy answer, so a caller gets Galaxy's own
  // reason. Two surfaces refusing the same call in a different order is the difference that
  // costs an agent a retry it cannot reason about, and Python's order is the contract.

  // The endpoint defaults content_format to html; these ops speak markdown. The cast is needed
  // because the pinned CreatePagePayload has no history_id at all and marks title, slug and
  // content required -- a history-attached notebook supplies none of the three.
  const body: Record<string, unknown> = { content_format: "markdown" };
  if (historyId != null) body["history_id"] = historyId;
  if (i.title != null) body["title"] = i.title;
  if (i.content != null) body["content"] = i.content;
  if (i.annotation != null) body["annotation"] = i.annotation;
  if (i.slug != null) body["slug"] = i.slug;

  const { data, error, response } = await ctx.client.POST("/api/pages", { body: body as never });
  if (error || !data) throw httpError(response, error);
  return stripRendered(data as PageDetail, false);
}

export const createPageOp: Operation<typeof input, PageDetail> = {
  name: "create_page",
  domain: "pages",
  summary:
    "Create a markdown page. With historyId it is a notebook attached to that history; " +
    "without one it is a standalone report, which requires a title and a unique slug.",
  input,
  requires: { galaxy: ">=26.1" },
  readOnly: false,
  run,
  // server.py, create_page: the id of the page Galaxy made -- the one thing the
  // caller did not have before the call -- read with dict.get and defaulted to the
  // empty string, so a reply that carries no id still produces a sentence.
  project: (p) => ({
    message: `Created page '${pyStr(pyGet(p as unknown as Record<string, unknown>, "id", ""))}'`,
  }),
  // server.py, create_page: a bioblend write, and the context names the history rather than
  // the page that does not exist yet.
  failure: {
    shape: "bioblend-write",
    action: "Create page",
    context: (i) => ({ history_id: i.historyId }),
  },
};

register(createPageOp as AnyOperation);

export const createPage = (i: In, ctx: GalaxyContext) => runOperation(createPageOp, i, ctx);
