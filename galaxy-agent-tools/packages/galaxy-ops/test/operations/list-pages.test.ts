import { describe, it, expect } from "vitest";
import { listPagesOp, listPages } from "../../src/operations/list-pages";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const ok = (data: unknown) => ({ data, response: { status: 200 } });

describe("list_pages", () => {
  it("is read-only, so the MCP surface annotates it as one", () => {
    expect(listPagesOp.readOnly ?? true).toBe(true);
  });

  it("sends every visibility flag explicitly and returns the pages", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/pages");
        const q = init.params.query;
        expect(q.show_own).toBe(true);
        expect(q.show_published).toBe(false);
        expect(q.show_shared).toBe(false);
        expect(q.limit).toBe(100);
        expect(q.offset).toBe(0);
        expect(q.search).toBeNull();
        expect("history_id" in q).toBe(false);
        return ok([{ id: "page1", title: "Notebook 1" }]);
      },
    });
    const out = await listPages({}, ctxWith(client));
    expect(out).toHaveLength(1);
    expect(out[0]?.id).toBe("page1");
  });

  it("filters to one history's notebooks and can widen visibility", async () => {
    const client = mockClient({
      GET: (_path, init) => {
        const q = init.params.query;
        expect(q.history_id).toBe("hist1");
        expect(q.show_published).toBe(true);
        expect(q.search).toBe("rnaseq");
        return ok([{ id: "page1", history_id: "hist1" }]);
      },
    });
    const out = await listPages(
      { historyId: "hist1", showPublished: true, search: "rnaseq" },
      ctxWith(client),
    );
    expect(out[0]?.history_id).toBe("hist1");
  });

  it("asks for shared pages when showShared is set", async () => {
    const client = mockClient({
      GET: (_path, init) => {
        expect(init.params.query.show_shared).toBe(true);
        expect(init.params.query.show_published).toBe(false);
        return ok([]);
      },
    });
    expect(await listPages({ showShared: true }, ctxWith(client))).toEqual([]);
  });

  it("reports the requested window as pagination", async () => {
    const client = mockClient({ GET: () => ok([{ id: "page3" }]) });
    const r = await runWithEnvelope(listPagesOp as any, { limit: 1, offset: 2 }, ctxWith(client));
    expect(r.success).toBe(true);
    expect(r.pagination).toEqual({ offset: 2, limit: 1 });
    expect(r.message).toBe("1 page(s)");
  });

  it("envelopes an auth failure instead of throwing", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "nope" }, response: { status: 403 } }) });
    const r = await runWithEnvelope(listPagesOp as any, {}, ctxWith(client));
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("auth");
  });
});
