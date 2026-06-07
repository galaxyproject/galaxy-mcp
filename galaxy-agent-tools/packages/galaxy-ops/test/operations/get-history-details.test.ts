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
    expect((out as any).id).toBe("h1");
  });
  it("throws NotFound on 404", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(getHistoryDetails({ historyId: "x" }, ctxWith(client))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });
});
