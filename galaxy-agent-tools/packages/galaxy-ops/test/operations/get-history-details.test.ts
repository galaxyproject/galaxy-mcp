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

  it("throws NotFound on 404", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(getHistoryDetails({ historyId: "x" }, ctxWith(client))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });
});
