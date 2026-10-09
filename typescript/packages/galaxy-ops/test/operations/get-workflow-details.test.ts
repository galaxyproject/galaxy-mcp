import { describe, it, expect } from "vitest";
import { getWorkflowDetailsOp, getWorkflowDetails } from "../../src/operations/get-workflow-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_workflow_details", () => {
  it("shows a workflow by id", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/workflows/{workflow_id}");
        expect(init.params.path.workflow_id).toBe("w1");
        return { data: { id: "w1", name: "RNAseq", steps: {} }, response: { status: 200 } };
      },
    });
    const out = await getWorkflowDetails({ workflowId: "w1" }, ctxWith(client));
    expect((out as any).id).toBe("w1");
  });
});
