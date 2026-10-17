// Shared types and helpers for the Galaxy Pages ops.
//
// A Galaxy "Page" is a markdown document. Attached to a history it is a Notebook;
// standalone it is a Report. Content is Galaxy-flavored markdown whose directives
// carry ENCODED ids (e.g. history_dataset_display(history_dataset_id=<encoded-dataset-id>)),
// so there is no encode/decode step: content_editor is read, edited and posted back as-is.
//
// Most of these ops declare Galaxy 26.1, each for its own reason -- get_page does not, because
// 26.0 already answers it with a content_editor. list_pages does, not for the endpoint but for
// its history filter, which 26.0 ignores while answering with every page the user can see.
// The models below are hand-written from 26.1's schema because the pinned bindings are 26.0.x,
// which predate the notebook work: a page there has no history_id and no edit_source, its slug
// is required rather than optional, /api/pages/{id}/revisions* is absent entirely, the update
// payload has no content field at all, and create/update declare a response_model of
// PageSummary -- which carries no content_editor, so those two return nothing worth editing.
// One field runs the other way: content_editor on a REVISION postdates 26.1 and is not in its
// schema, so the revision model below describes both shapes rather than only the newer one.
// Deriving from the bindings is not an option either way: the 26.0 PageDetails ends in an
// `& { [key: string]: unknown }` index signature, and Omit over that erases every named field
// while quietly admitting misspellings. Delete this block once bindings.ts moves to 26.1.x and
// use GetJson/PostJson/PutJson directly.

/** GET /api/pages -- one entry of the index. */
export interface PageSummary {
  id: string;
  title: string;
  slug?: string | null;
  /** Set when the page is a history-attached notebook; absent for a standalone report. */
  history_id?: string | null;
  latest_revision_id: string;
  revision_ids: string[];
  source_invocation_id?: string | null;
  model_class: "Page";
  username: string;
  email_hash: string;
  author_deleted: boolean;
  deleted: boolean;
  importable: boolean;
  published: boolean;
  tags: string[];
  create_time: string;
  update_time: string;
}

/**
 * GET /api/pages/{id}, and what create/update answer with. `content_editor` is the editable
 * markdown; `content` is the same document with its embeds expanded, and the ops drop it unless
 * it was asked for -- so it is optional here rather than required. An HTML page has its body
 * there and nowhere else, because Galaxy fills content_editor on the markdown path only and the
 * field defaults to the empty string; the ops drop `content` there too, which is what the other
 * server does and what `includeRendered` is for.
 */
export interface PageDetail extends PageSummary {
  content_editor: string | null;
  content?: string | null;
  content_format: string;
  annotation: string | null;
  /** Who wrote the latest revision: "user", "agent" or "restore". */
  edit_source?: string | null;
  generate_time?: string | null;
  generate_version?: string | null;
}

/** A page as get_page answers: the editable source and Galaxy's hash of it. */
export type HashedPage = PageDetail & { content_hash: string };

/**
 * Galaxy's page hash of the editable source, so a caller can tell whether a page changed.
 *
 * The source is `content_editor`, or `content` for an HTML page, which Galaxy fills on the
 * markdown path only. The hash is `_djb2_hash` in lib/galaxy/agents/page_assistant.py: djb2
 * over code points, eight hex digits, as the other server computes it. (Galaxy's client spells
 * it over UTF-16 code units, which differs only outside the Basic Multilingual Plane.)
 */
export function contentHash(page: Pick<PageDetail, "content_editor" | "content">): string {
  let h = 5381;
  for (const c of page.content_editor || page.content || "") {
    h = (h * 33 + c.codePointAt(0)!) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

/** Galaxy's heading rule, client/src/components/PageEditor/sectionDiffUtils.ts. */
const HEADING = /^#{1,6}\s/;

/**
 * Markdown split into sections at headings, as Galaxy's page editor splits it
 * (`markdownSections` in sectionDiffUtils.ts): the text before the first heading is a section
 * whose heading is "", and each section's text includes its heading line.
 */
export function markdownSections(content: string): { heading: string; content: string }[] {
  if (!content) return [];
  const lines = content.split("\n");
  const sections: { heading: string; content: string }[] = [];
  let heading = "";
  let current: string[] = [];
  lines.forEach((line, i) => {
    if (HEADING.test(line) && i > 0) {
      sections.push({ heading, content: current.join("\n") });
      heading = line;
      current = [line];
    } else if (HEADING.test(line)) {
      heading = line;
      current = [line];
    } else {
      current.push(line);
    }
  });
  if (current.length > 0) sections.push({ heading, content: current.join("\n") });
  return sections;
}

/**
 * Replace the section under `heading` with `section` (its heading line included), appending it
 * when no section has that heading -- `applySectionEdit` in sectionDiffUtils.ts, which is how
 * Galaxy applies its own page assistant's section patches.
 */
export function applySectionEdit(original: string, heading: string, section: string): string {
  let found = false;
  const parts = markdownSections(original).map((s) => {
    if (s.heading !== heading) return s.content;
    found = true;
    return section;
  });
  if (!found) parts.push(section);
  return parts.join("\n");
}

/** The directive arguments Galaxy decodes as encoded ids, lib/galaxy/managers/markdown_util.py. */
const ID_ARGUMENTS = [
  "history_id",
  "workflow_id",
  "history_dataset_id",
  "history_dataset_collection_id",
  "job_id",
  "implicit_collection_jobs_id",
  "invocation_id",
];
const ID_ARGUMENT = new RegExp(`\\b(${ID_ARGUMENTS.join("|")})\\s*=\\s*["']?([^\\s,)"']+)`, "g");
/** An encoded id: Galaxy's cipher works in 8-byte blocks, so hex in runs of sixteen. */
const ENCODED_ID = /^(?:[0-9a-f]{16})+$/;
/** Where Galaxy reads directives: fenced galaxy blocks and `${galaxy ...}` embeds. */
const DIRECTIVES = /^```[ \t]*galaxy[^\n]*\n[\s\S]*?^```|\$\{galaxy\s[^}]*\}/gm;

/**
 * Directive arguments that name a Galaxy object by something that is not an encoded id -- a
 * hid, a name, a decoded integer -- spelled `name=value`. Galaxy decodes these when it stores a
 * page, so such a value is one it cannot resolve. Prose outside directives is not read.
 */
export function malformedObjectIds(content: string | null | undefined): string[] {
  const found: string[] = [];
  for (const directive of (content ?? "").match(DIRECTIVES) ?? []) {
    for (const [, name, value] of directive.matchAll(ID_ARGUMENT)) {
      if (!ENCODED_ID.test(value!)) found.push(`${name}=${value}`);
    }
  }
  return found;
}

/** One entry of GET /api/pages/{id}/revisions. */
export interface PageRevisionSummary {
  id: string;
  page_id: string;
  edit_source?: string | null;
  create_time: string;
  update_time: string;
}

/**
 * GET /api/pages/{id}/revisions/{revision_id} and its revert, as the server sends them. Galaxy
 * started returning `content_editor` on a revision after 26.1 -- through 26.1.1 the revision
 * response model lists only title, content and content_format -- so the field is optional here.
 */
export interface PageRevisionResponse extends PageRevisionSummary {
  content_editor?: string | null;
  content?: string | null;
  content_format?: string | null;
  title?: string | null;
}

/** Which field of the response the editable markdown was taken from. */
export type ContentEditorSource = "server" | "content" | "none";

/**
 * A revision as the ops hand it on. `content_editor` is the editable markdown with its
 * directives intact and `content` is the same document with its embeds expanded for export, so
 * callers edit and send back `content_editor`.
 *
 * `content_editor_source` says where that text came from, as a fact about the response rather
 * than a guess from the server's version: "server" is the revision's own content_editor,
 * "content" is the expanded render standing in for it -- editing that and passing it to
 * update_page bakes the expansion into the page -- and "none" is a revision that carried
 * neither, where content_editor is null.
 */
export interface PageRevisionDetails extends PageRevisionResponse {
  content_editor: string | null;
  content_editor_source: ContentEditorSource;
}

/**
 * Give a revision one field to edit whatever the server sent, and say which it was.
 *
 * `content` is the only body an older Galaxy puts in a revision, so falling back to it beats
 * handing a caller nothing -- but silently, a caller cannot tell an expanded body from an
 * editable one, so the fallback is reported rather than inferred. Empty counts as missing:
 * content_editor defaults to the empty string and Galaxy fills it on the markdown path only,
 * so an HTML revision arrives with an empty one and its body in `content`. Copies rather than
 * handing back the parsed response.
 */
export function withEditableContent(rev: PageRevisionResponse): PageRevisionDetails {
  if (rev.content_editor) {
    return { ...rev, content_editor: rev.content_editor, content_editor_source: "server" };
  }
  if (rev.content != null) {
    return { ...rev, content_editor: rev.content, content_editor_source: "content" };
  }
  return { ...rev, content_editor: null, content_editor_source: "none" };
}

/**
 * Drop the expanded render, keeping the editable `content_editor`.
 *
 * Callers edit content_editor and send it back; the rendered form is only worth its size when
 * it is asked for. The drop is unconditional, as the other server's is: an HTML page arrives
 * with an empty content_editor, because Galaxy fills that field on the markdown path only, and
 * its body is in `content` -- and `content` goes anyway. That is the one field this cannot
 * keep on a hunch: a caller reading `content` from a page would be editing the expanded form
 * and baking the expansion in when it sent it back, and get_page takes `includeRendered` for
 * exactly the caller who does want it. Copies rather than handing back the parsed response.
 */
export function stripRendered(page: PageDetail, includeRendered: boolean): PageDetail {
  if (includeRendered) return { ...page };
  const { content: _rendered, ...rest } = page;
  return rest;
}
