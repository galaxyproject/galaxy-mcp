import { describe, it, expect } from "vitest";
import { getHistoriesOp, getHistories } from "../../src/operations/get-histories";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_histories", () => {
  it("passes limit/offset as query and filters name client-side", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/histories");
        expect(init.params.query.limit).toBe(2);
        return { data: [{ id: "h1", name: "alpha" }, { id: "h2", name: "beta" }], response: { status: 200 } };
      },
    });
    const out = await getHistories({ limit: 2, name: "alp" }, ctxWith(client));
    expect(out.map((h: any) => h.id)).toEqual(["h1"]);
  });
});
