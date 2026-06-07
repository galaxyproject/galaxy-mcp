import { describe, it, expect } from "vitest";
import { getHistoryContentsOp, getHistoryContents } from "../../src/operations/get-history-contents";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_history_contents", () => {
  it("lists a history's contents by id", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/histories/{history_id}/contents");
        expect(init.params.path.history_id).toBe("h1");
        return { data: [{ id: "ds1", history_content_type: "dataset" }], response: { status: 200 } };
      },
    });
    const out = await getHistoryContents({ historyId: "h1" }, ctxWith(client));
    expect((out as any[]).length).toBe(1);
  });
});
