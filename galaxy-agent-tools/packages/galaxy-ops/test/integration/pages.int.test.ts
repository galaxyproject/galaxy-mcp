import { describe, it, expect } from "vitest";
import {
  createGalaxyContext,
  createHistory,
  createPage,
  getPage,
  listPages,
  updatePage,
  listPageRevisions,
  getPageRevision,
  revertPageRevision,
} from "../../src/index";

const URL = process.env.GALAXY_URL;
const KEY = process.env.GALAXY_API_KEY;
const run = URL && KEY ? describe : describe.skip;

// Needs Galaxy 26.1. The revision endpoints and several page payload fields these ops use are
// hand-typed from its schema -- the pinned 26.0.x bindings describe none of them -- so this is
// the only place those shapes meet a real server. Both halves matter: the report path exercises
// the endpoints the bindings do know, the notebook path exercises the 26.1-only history_id.
run("integration: pages on a real Galaxy", () => {
  const ctx = () => createGalaxyContext({ baseUrl: URL!, apiKey: KEY! });
  const stamp = Date.now();

  it("walks a standalone report from creation through a revert", async () => {
    const slug = `agent-tools-report-${stamp}`;
    const created = await createPage({ title: "agent-tools report", slug, content: "# first" }, ctx());
    expect(created.id).toBeTruthy();
    expect(created.content_editor).toContain("first");
    expect("content" in created).toBe(false);

    const rendered = await getPage({ pageId: created.id, includeRendered: true }, ctx());
    expect(rendered.content).toBeTypeOf("string");

    const updated = await updatePage({ pageId: created.id, content: "# second" }, ctx());
    expect(updated.content_editor).toContain("second");
    expect(updated.edit_source).toBe("agent");

    const revisions = await listPageRevisions({ pageId: created.id, sortDesc: false }, ctx());
    expect(revisions.length).toBeGreaterThanOrEqual(2);
    const first = revisions[0]!;
    expect(first.page_id).toBe(created.id);
    // the hand-written model marks these required -- prove the server really sends them
    expect(first.create_time).toBeTypeOf("string");
    expect(first.update_time).toBeTypeOf("string");

    const revision = await getPageRevision({ pageId: created.id, revisionId: first.id }, ctx());
    expect(revision.content).toContain("first");

    const restored = await revertPageRevision({ pageId: created.id, revisionId: first.id }, ctx());
    expect(restored.edit_source).toBe("restore");
    expect(restored.content).toContain("first");
  }, 60_000);

  it("creates a history-attached notebook and finds it by history", async () => {
    const history = await createHistory({ historyName: `agent-tools pages ${stamp}` }, ctx());
    const historyId = (history as { id: string }).id;

    // No title and no slug: a notebook needs neither, and Galaxy names an untitled one itself.
    // This is the 26.1 behavior the whole notebook-vs-report split rests on.
    const notebook = await createPage({ historyId, content: "# notebook" }, ctx());
    expect(notebook.history_id).toBe(historyId);
    expect(notebook.title).toBeTruthy();

    const listed = await listPages({ historyId }, ctx());
    expect(listed.map((p) => p.id)).toContain(notebook.id);
    // the filter must be doing real work, not being ignored by the server
    expect(listed.every((p) => p.history_id === historyId)).toBe(true);
  }, 60_000);

  it("lets Galaxy reject a duplicate slug", async () => {
    const slug = `agent-tools-dupe-${stamp}`;
    await createPage({ title: "first", slug }, ctx());
    // The client-side guard only catches a MISSING slug; uniqueness is the server's call, and
    // nothing else in the suite proves Galaxy still enforces it.
    await expect(createPage({ title: "second", slug }, ctx())).rejects.toThrow();
  }, 60_000);
});
