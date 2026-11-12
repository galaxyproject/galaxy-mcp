import { describe, it, expect } from "vitest";
import { importWorkflowOp, importWorkflow } from "../../src/operations/import-workflow";
import { GalaxyValidationError } from "../../src/errors";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const DEFINITION = { class: "GalaxyWorkflow", label: "wf", inputs: {}, steps: {} };

function recordingClient() {
  const calls: { path: string; body: any }[] = [];
  const client = mockClient({
    POST: (path, init) => {
      calls.push({ path, body: init?.body });
      return { data: { id: "w1", name: "wf" }, response: { status: 200 } };
    },
  });
  return { client, calls };
}

describe("import_workflow", () => {
  it("is a write op", () => {
    expect(importWorkflowOp.readOnly).toBe(false);
  });

  it("POSTs the definition to /api/workflows", async () => {
    const { client, calls } = recordingClient();
    const out = await importWorkflow({ definition: DEFINITION }, ctxWith(client));
    expect(calls).toEqual([{ path: "/api/workflows", body: { workflow: DEFINITION } }]);
    expect(out.id).toBe("w1");
  });

  it("accepts the definition as a JSON object string", async () => {
    const { client, calls } = recordingClient();
    await importWorkflow({ definition: JSON.stringify(DEFINITION) }, ctxWith(client));
    expect(calls[0]!.body.workflow).toEqual(DEFINITION);
  });

  it.each(["not json", "[1, 2]", "", {}])("refuses %j without posting", async (bad) => {
    const { client, calls } = recordingClient();
    await expect(importWorkflow({ definition: bad as any }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyValidationError,
    );
    expect(calls).toEqual([]);
  });

  it("names the stored workflow, or None like the Python server", () => {
    expect(importWorkflowOp.project!({ id: "w1", name: "wf" }, { definition: DEFINITION }).message).toBe(
      "Imported workflow 'wf'",
    );
    expect(importWorkflowOp.project!({ id: "w1" }, { definition: DEFINITION }).message).toBe(
      "Imported workflow 'None'",
    );
  });
});
