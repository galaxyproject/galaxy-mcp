import { describe, it, expect } from "vitest";
import { getHistoryDetailsOp, getHistoryDetails } from "../../src/operations/get-history-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_history_details", () => {
  it("fetches a history by id", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/histories/{history_id}");
        expect(init.params.path.history_id).toBe("h1");
        return { data: { id: "h1", name: "alpha", state: "ok" }, response: { status: 200 } };
      },
    });
    const out = await getHistoryDetails({ historyId: "h1" }, ctxWith(client));
    expect((out.history as any).id).toBe("h1");
  });

  it("reports the item count Galaxy already holds, without listing the contents", async () => {
    const asked: string[] = [];
    const client = mockClient({
      GET: (path) => {
        asked.push(path);
        return { data: { id: "h1", name: "alpha", state: "ok", count: 42 }, response: { status: 200 } };
      },
    });
    const out = await getHistoryDetails({ historyId: "h1" }, ctxWith(client));
    expect(out.contents_summary.total_items).toBe(42);
    expect(out.contents_summary.note).toContain("get_history_contents");
    // One request: the Python tool fetches the contents a second time to length them.
    expect(asked).toEqual(["/api/histories/{history_id}"]);
    expect(getHistoryDetailsOp.project!(out, { historyId: "h1" }).message).toBe(
      "History h1 state=ok (42 item(s))",
    );
  });

  it("says nought rather than nothing when Galaxy reports no count", async () => {
    const client = mockClient({ GET: () => ({ data: { id: "h1" }, response: { status: 200 } }) });
    const out = await getHistoryDetails({ historyId: "h1" }, ctxWith(client));
    expect(out.contents_summary.total_items).toBe(0);
  });
  it("throws NotFound on 404", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(getHistoryDetails({ historyId: "x" }, ctxWith(client))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });
});
