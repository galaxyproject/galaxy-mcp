// Shared types and helpers for the Galaxy Pages ops.
//
// A Galaxy "Page" is a markdown document. Attached to a history it is a Notebook;
// standalone it is a Report. Content is Galaxy-flavored markdown whose directives
// carry ENCODED ids (e.g. history_dataset_display(history_dataset_id=f2db41e1fa331b3e)),
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
 * it was asked for -- so it is optional here rather than required. An HTML page is the exception:
 * Galaxy fills content_editor on the markdown path only, and the field defaults to the empty
 * string, so there the body is in `content` and the ops keep it.
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
 * it is asked for. Where there is no content_editor -- an HTML page, where Galaxy never fills
 * one -- `content` is the only body the page has, so it stays. Copies rather than handing back
 * the parsed response.
 */
export function stripRendered(page: PageDetail, includeRendered: boolean): PageDetail {
  if (includeRendered || !page.content_editor) return { ...page };
  const { content: _rendered, ...rest } = page;
  return rest;
}
