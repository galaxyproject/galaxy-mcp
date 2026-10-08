import { describe, it, expect } from "vitest";
import { getToolDetailsOp, getToolDetails } from "../../src/operations/get-tool-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_tool_details", () => {
  it("fetches a tool via the legacy endpoint", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/tools/{tool_id}");
        expect(init.params.path.tool_id).toBe("cat1");
        return { data: { id: "cat1", name: "Concatenate", version: "1.0" }, response: { status: 200 } };
      },
    });
    const out = await getToolDetails({ toolId: "cat1" }, ctxWith(client));
    expect(out.id).toBe("cat1");
    expect(out.name).toBe("Concatenate");
  });
  it("throws NotFound on 404", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 404 } }) });
    await expect(getToolDetails({ toolId: "nope" }, ctxWith(client))).rejects.toBeInstanceOf(GalaxyNotFoundError);
  });
});
