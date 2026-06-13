import { describe, it, expect } from "vitest";
import { legacyGet } from "../src/legacy";
import { mockClient } from "./util/mock-client";
import { DEFAULT_POLL } from "../src/context";
import { GalaxyNotFoundError } from "../src/errors";
import type { GalaxyContext } from "../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("legacyGet", () => {
  it("returns the data for an off-schema path via the configured client", async () => {
    const client = mockClient({
      GET: (path) => {
        expect(path).toBe("/api/tools/{tool_id}");
        return { data: { id: "cat1", name: "Concatenate" }, response: { status: 200 } };
      },
    });
    const out = await legacyGet<{ id: string }>(ctxWith(client), "/api/tools/{tool_id}", {
      params: { path: { tool_id: "cat1" } },
    });
    expect(out.id).toBe("cat1");
  });

  it("classifies HTTP failures into typed errors", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(legacyGet(ctxWith(client), "/api/tools/{tool_id}", {})).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });
});
