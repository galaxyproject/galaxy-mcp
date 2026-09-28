import { describe, it, expect } from "vitest";
import { getHistoryDetailsOp, getHistoryDetails } from "../../src/operations/get-history-details";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/** The history, and the contents the count is taken from. */
const serving = (contents: unknown[]) =>
  mockClient({
    GET: (path, init) => {
      expect(init.params.path.history_id).toBe("h1");
      if (path === "/api/histories/{history_id}/contents") {
        return { data: contents, response: { status: 200 } };
      }
      expect(path).toBe("/api/histories/{history_id}");
      return { data: { id: "h1", name: "alpha", state: "ok" }, response: { status: 200 } };
    },
  });

describe("get_history_details", () => {
  it("fetches a history by id", async () => {
    const out = await getHistoryDetails({ historyId: "h1" }, ctxWith(serving([])));
    expect((out as any).id).toBe("h1");
  });

  /**
   * The count is everything in the history, deleted and hidden included, which is
   * what the Python tool reports -- and no field of the history metadata means
   * that, so both surfaces pay for a second request to get it.
   */
  it("counts the whole history, which costs a second request", async () => {
    const items = Array.from({ length: 7 }, (_, k) => ({ id: `d${k}`, deleted: k === 0 }));
    const r = await runWithEnvelope(getHistoryDetailsOp as never, { historyId: "h1" } as never, ctxWith(serving(items)));
    expect(r.count).toBe(7);
    expect(r.pagination).toBeNull();
  });

  /**
   * Two calls in flight at once, each counting its own history.
   *
   * The count cannot travel in what run() returns without changing what every
   * library caller gets, and it used to be parked in a module-level WeakMap keyed
   * by the history record. A client that answers both calls with the SAME frozen
   * record -- one cached object for one id -- then gave both of them whichever
   * count was written last. The channel is per call now.
   */
  it("keeps two concurrent calls' counts apart, even sharing one history record", async () => {
    const shared = Object.freeze({ id: "h1", name: "alpha", state: "ok" });
    const sizes = [3, 9];
    let arrived = 0;
    let release!: () => void;
    const both = new Promise<void>((resolve) => (release = resolve));
    const client = mockClient({
      GET: async (path: string, init: any) => {
        if (path === "/api/histories/{history_id}") {
          return { data: shared, response: { status: 200 } };
        }
        const size = sizes[Number(init.params.path.history_id === "second")] ?? 0;
        arrived += 1;
        if (arrived === sizes.length) release();
        // Neither call comes back until both have asked, so the two runs really
        // are interleaved rather than one after the other.
        await both;
        return { data: Array.from({ length: size }, (_, k) => ({ id: `d${k}` })), response: { status: 200 } };
      },
    });

    const [first, second] = await Promise.all([
      runWithEnvelope(getHistoryDetailsOp as never, { historyId: "first" } as never, ctxWith(client)),
      runWithEnvelope(getHistoryDetailsOp as never, { historyId: "second" } as never, ctxWith(client)),
    ]);
    expect([first.count, second.count]).toEqual([3, 9]);
  });

  /**
   * The envelope wraps the record and states the count beside it, which is the
   * other server's data shape -- `{history, contents_summary}`, note included.
   * run() still hands a library caller the bare record.
   */
  it("answers with the history under its own key and the count beside it", async () => {
    const items = Array.from({ length: 4 }, (_, k) => ({ id: `d${k}` }));
    const r = await runWithEnvelope(getHistoryDetailsOp as never, { historyId: "h1" } as never, ctxWith(serving(items)));
    expect(r.data).toEqual({
      history: { id: "h1", name: "alpha", state: "ok" },
      contents_summary: {
        total_items: 4,
        note:
          "This is just a count. To get actual datasets, use " +
          "get_history_contents(history_id, limit=25, order='create_time-dsc') " +
          "for newest datasets first.",
      },
    });
    const bare = await getHistoryDetails({ historyId: "h1" }, ctxWith(serving(items)));
    expect(bare).toEqual({ id: "h1", name: "alpha", state: "ok" });
  });

  it("throws NotFound on 404", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(getHistoryDetails({ historyId: "x" }, ctxWith(client))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });
});
