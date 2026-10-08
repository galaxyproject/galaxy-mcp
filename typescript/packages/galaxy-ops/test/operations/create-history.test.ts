import { describe, it, expect } from "vitest";
import { createHistoryOp, createHistory } from "../../src/operations/create-history";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("create_history", () => {
  it("is a write op and posts the name", async () => {
    expect(createHistoryOp.readOnly).toBe(false);
    const client = mockClient({
      POST: (path, init) => {
        expect(path).toBe("/api/histories");
        expect(init.body.name).toBe("My analysis");
        return { data: { id: "h9", name: "My analysis" }, response: { status: 200 } };
      },
    });
    const out = await createHistory({ historyName: "My analysis" }, ctxWith(client));
    expect((out as any).id).toBe("h9");
  });
});
