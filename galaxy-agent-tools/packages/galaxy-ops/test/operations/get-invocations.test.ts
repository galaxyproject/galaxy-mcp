import { describe, it, expect } from "vitest";
import { getInvocationsOp, getInvocations } from "../../src/operations/get-invocations";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyNotFoundError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_invocations", () => {
  it("has parity name and returns the invocation", async () => {
    expect(getInvocationsOp.name).toBe("get_invocations");
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/invocations/{invocation_id}");
        expect(init.params.path.invocation_id).toBe("inv1");
        return { data: { id: "inv1", state: "scheduled", steps: [] }, response: { status: 200 } };
      },
    });
    const inv = await getInvocations({ invocationId: "inv1" }, ctxWith(client));
    expect((inv as any).id).toBe("inv1");
  });

  it("throws GalaxyNotFoundError on 404", async () => {
    const client = mockClient({ GET: () => ({ error: {}, response: { status: 404 } }) });
    await expect(getInvocations({ invocationId: "x" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });
});
