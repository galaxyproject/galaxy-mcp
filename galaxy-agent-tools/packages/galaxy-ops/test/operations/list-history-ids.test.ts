import { describe, it, expect } from "vitest";
import { listHistoryIdsOp, listHistoryIds } from "../../src/operations/list-history-ids";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("list_history_ids", () => {
  it("returns id/name pairs", async () => {
    const client = mockClient({ GET: () => ({ data: [{ id: "h1", name: "alpha" }], response: { status: 200 } }) });
    const out = await listHistoryIds({}, ctxWith(client));
    expect(out).toEqual([{ id: "h1", name: "alpha" }]);
    expect(listHistoryIdsOp.name).toBe("list_history_ids");
  });
});
