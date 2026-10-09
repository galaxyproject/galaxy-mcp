import { describe, it, expect } from "vitest";
import { cancelWorkflowInvocationOp, cancelWorkflowInvocation } from "../../src/operations/cancel-workflow-invocation";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("cancel_workflow_invocation", () => {
  it("is a write op and destructive", () => {
    expect(cancelWorkflowInvocationOp.readOnly).toBe(false);
    expect(cancelWorkflowInvocationOp.destructive).toBe(true);
  });

  it("DELETEs the invocation and returns cancelled result", async () => {
    const client = mockClient({
      DELETE: (path, init) => {
        expect(path).toBe("/api/invocations/{invocation_id}");
        expect(init.params.path.invocation_id).toBe("inv1");
        return { data: { id: "inv1", state: "cancelled" }, response: { status: 200 } };
      },
    });
    const out = await cancelWorkflowInvocation({ invocationId: "inv1" }, ctxWith(client));
    expect(out.cancelled).toBe(true);
    expect(out.invocation).toMatchObject({ id: "inv1", state: "cancelled" });
  });

  it("throws on DELETE error", async () => {
    const client = mockClient({
      DELETE: () => ({ error: "not found", response: { status: 404 } }),
    });
    await expect(cancelWorkflowInvocation({ invocationId: "inv2" }, ctxWith(client))).rejects.toThrow();
  });
});
