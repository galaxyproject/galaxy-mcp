import { describe, it, expect } from "vitest";
import { listPagesOp, listPages } from "../../src/operations/list-pages";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const ok = (data: unknown, totalMatches?: number) => ({
  data,
  response: {
    status: 200,
    // Galaxy reports how many pages matched on a header rather than in the body,
    // which is where the envelope's total comes from.
    headers: new Headers(totalMatches === undefined ? {} : { total_matches: String(totalMatches) }),
  },
});

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

  it("reports the window, and the total the response header carried", async () => {
    const client = mockClient({ GET: () => ok([{ id: "page3" }], 12) });
    const r = await runWithEnvelope(listPagesOp as any, { limit: 1, offset: 2 }, ctxWith(client));
    expect(r.success).toBe(true);
    expect(r.count).toBe(1);
    // The shared describer, like every other listing: it advances by what came back
    // and says in a sentence where the window sits. This tool used to build the block
    // by hand, to match a hand-built one on the other server; both are gone.
    expect(r.pagination).toEqual({
      total_items: 12,
      returned_items: 1,
      limit: 1,
      offset: 2,
      has_next: true,
      has_previous: true,
      next_offset: 3,
      previous_offset: 1,
      helper_text: "Showing 1 of 12 pages (offset 2). Use offset=3 for the next page.",
    });
    expect(r.message).toBe("1 page(s)");
  });

  /** An offset past the last page gets the sentence that says so, not a blank one. */
  it("says an offset is past the end, as the other listings do", async () => {
    const client = mockClient({ GET: () => ok([], 12) });
    const r = await runWithEnvelope(listPagesOp as any, { limit: 5, offset: 40 }, ctxWith(client));
    expect(r.pagination).toMatchObject({
      total_items: 12,
      returned_items: 0,
      has_next: false,
      next_offset: null,
      helper_text: "offset 40 is past the end of 12 pages; use a smaller offset",
    });
  });

  /**
   * The line that makes a total-from-a-header expressible through the shared
   * describer: rows in hand prove a floor, so a header that under-reports cannot
   * make the page it arrived with disappear.
   */
  it("does not let a header smaller than the page shrink the total", async () => {
    const client = mockClient({ GET: () => ok([{ id: "a" }, { id: "b" }, { id: "c" }], 1) });
    const r = await runWithEnvelope(listPagesOp as any, { limit: 5, offset: 0 }, ctxWith(client));
    expect(r.pagination).toMatchObject({ total_items: 3, returned_items: 3, has_next: false });
  });

  it("falls back to the page it was handed when no total came back", async () => {
    const client = mockClient({ GET: () => ok([{ id: "page3" }]) });
    const r = await runWithEnvelope(listPagesOp as any, { limit: 5, offset: 0 }, ctxWith(client));
    expect(r.pagination).toMatchObject({ total_items: 1, returned_items: 1, has_next: false });
  });

  /**
   * Two calls in flight at once, each with its own total.
   *
   * The fact the envelope needs -- how many pages matched -- arrives on a response
   * header and cannot travel in what run() returns without changing what every
   * library caller gets. It used to be parked in a module-level WeakMap keyed by
   * the array run() returned, and a client that answers two calls with the SAME
   * array (a cache handing out one frozen page) gave both of them whichever total
   * was written last. The channel is per call now, so it cannot be.
   */
  it("keeps two concurrent calls' totals apart, even sharing one page array", async () => {
    // One array, frozen, handed to both calls: object identity cannot tell them
    // apart, which is the whole point.
    const shared = Object.freeze([{ id: "page1", title: "Shared" }]);
    let arrived = 0;
    let release!: () => void;
    const both = new Promise<void>((resolve) => (release = resolve));
    const totals = [10, 20];
    const client = mockClient({
      GET: async () => {
        const total = totals[arrived] ?? 0;
        arrived += 1;
        if (arrived === totals.length) release();
        // Neither call comes back until both have been made, so the two runs
        // really are interleaved rather than one after the other.
        await both;
        return ok(shared, total);
      },
    });

    const [first, second] = await Promise.all([
      runWithEnvelope(listPagesOp as any, { limit: 1, offset: 0 }, ctxWith(client)),
      runWithEnvelope(listPagesOp as any, { limit: 1, offset: 0 }, ctxWith(client)),
    ]);
    expect([first.pagination?.total_items, second.pagination?.total_items]).toEqual([10, 20]);
  });

  it("envelopes an auth failure instead of throwing", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "nope" }, response: { status: 403 } }) });
    const r = await runWithEnvelope(listPagesOp as any, {}, ctxWith(client));
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("auth");
  });
});
