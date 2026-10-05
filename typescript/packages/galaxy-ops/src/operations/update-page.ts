import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyValidationError, httpError } from "../errors";
import {
  applySectionEdit,
  contentHash,
  malformedObjectIds,
  stripRendered,
  type HashedPage,
  type PageDetail,
} from "./pages-common";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const input = {
  pageId: z.string().min(1).describe("Encoded page id"),
  content: z
    .string()
    .nullish()
    .describe("New Galaxy-flavored markdown, with ENCODED ids in directives. Omit to leave unchanged"),
  title: z.string().min(1).nullish().describe("New title. Omit to leave unchanged"),
  sectionHeading: z
    .string()
    .min(1)
    .nullish()
    .describe(
      "The exact heading line of the section to replace, e.g. '## Methods'; every section with " +
        "that heading line is replaced, as Galaxy's page editor does. Markdown pages only",
    ),
  sectionContent: z
    .string()
    .nullish()
    .describe("That section's new text, heading line included. Appended when no section has the heading"),
  expectHash: z
    .string()
    .nullish()
    .describe("content_hash from when the page was read; the write is refused if the page changed since"),
};
type In = {
  pageId: string;
  content?: string | null;
  title?: string | null;
  sectionHeading?: string | null;
  sectionContent?: string | null;
  expectHash?: string | null;
};

/** The refusals that need no request, in the order the other server makes them. */
function refusal(i: In): string | undefined {
  const section = i.sectionHeading != null || i.sectionContent != null;
  if (section && (i.sectionHeading == null || i.sectionContent == null)) {
    return "section_heading and section_content go together: give both to replace one section.";
  }
  if (section && i.content != null) {
    return "Give either content or a section to replace, not both.";
  }
  const malformed = malformedObjectIds(section ? i.sectionContent : i.content);
  if (malformed.length) {
    return (
      "These directive arguments name a Galaxy object by something that is not its encoded id: " +
      `${malformed.join(", ")}. Galaxy cannot resolve them, so the embed would render nothing. ` +
      "Use the encoded id a tool returned, e.g. from get_history_contents."
    );
  }
  return undefined;
}

async function run(i: In, ctx: GalaxyContext): Promise<HashedPage> {
  const refused = refusal(i);
  if (refused) throw new GalaxyValidationError(refused);

  // No refusal for an edit with nothing in it: the other server sends the PUT with only its
  // edit_source in the body and lets Galaxy decide, which is a no-op revision rather than an
  // error. update_history refuses its empty update -- this tool does not, and copying the
  // sibling's refusal here would be inventing a rule one surface has.

  // edit_source attributes the new revision to the agent rather than to a person; Galaxy only
  // writes a revision when content changes, so a title-only edit records nothing. The cast is
  // needed because the pinned UpdatePagePayload has neither edit_source nor content, and marks
  // slug and title required -- a content-only edit sends neither.
  const body: Record<string, unknown> = { edit_source: "agent" };
  let content = i.content;

  // A section edit and an expected hash both need the page as it is now. Galaxy's PUT has no
  // precondition of its own, so this read and the write below are two requests, and a writer
  // that lands between them is not seen.
  if (i.sectionHeading != null || i.expectHash != null) {
    const { data, error, response } = await ctx.client.GET("/api/pages/{id}", {
      params: { path: { id: i.pageId } },
    });
    if (error || !data) throw httpError(response, error);
    const current = data as PageDetail;
    const actual = contentHash(current);
    if (i.expectHash != null && i.expectHash !== actual) {
      throw new GalaxyValidationError(
        `Page '${i.pageId}' changed since it was read: its content_hash is now ${actual}, not ` +
          `${i.expectHash}. Read it again with get_page and make the edit against what is there now.`,
      );
    }
    if (i.sectionHeading != null) {
      if (current.content_format !== "markdown") {
        throw new GalaxyValidationError(
          `Page '${i.pageId}' is authored as ${current.content_format}; a section can be ` +
            "replaced only in a markdown page. Send the whole content instead.",
        );
      }
      content = applySectionEdit(
        current.content_editor || current.content || "",
        i.sectionHeading,
        i.sectionContent!,
      );
    }
  }
  if (content != null) body["content"] = content;
  if (i.title != null) body["title"] = i.title;

  const { data, error, response } = await ctx.client.PUT("/api/pages/{id}", {
    params: { path: { id: i.pageId } },
    body: body as never,
  });
  if (error || !data) throw httpError(response, error);
  const written = data as PageDetail;
  return { ...stripRendered(written, false), content_hash: contentHash(written) };
}

export const updatePageOp: Operation<typeof input, HashedPage> = {
  name: "update_page",
  domain: "pages",
  summary:
    "Update a page's content or title, or replace one section by its heading line. Pass the " +
    "content_hash get_page returned as expectHash to refuse the write if the page changed since. " +
    "New content creates a revision tagged edit_source=agent; a title-only change does not. The " +
    "page keeps its content_format, so sending markdown to a page authored as HTML stores it " +
    "under the wrong format. Answers with the page and its new content_hash.",
  input,
  requires: { galaxy: ">=26.1" },
  readOnly: false,
  run,
  // server.py, update_page: the page that was updated, and not which fields moved.
  // A caller knows what it sent; the updated record is in data.
  project: (_p, i) => ({ message: `Updated page '${i.pageId}'` }),
  // server.py, update_page: the read is a raw GET, the write a bioblend PUT.
  failure: {
    shape: (facts) => (facts.method === "GET" ? "raise-for-status" : "bioblend-write"),
    action: "Update page",
    context: (i) => ({ page_id: i.pageId }),
  },
};

register(updatePageOp as AnyOperation);

export const updatePage = (i: In, ctx: GalaxyContext) => runOperation(updatePageOp, i, ctx);
