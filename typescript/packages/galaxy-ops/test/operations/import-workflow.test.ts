import { describe, it, expect } from "vitest";
import { importWorkflowOp, importWorkflow } from "../../src/operations/import-workflow";
import { GalaxyValidationError } from "../../src/errors";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const DEFINITION = { class: "GalaxyWorkflow", label: "wf", inputs: {}, steps: {} };
const NOT_INSTALLED = "Tool is not installed";

function recordingClient(exported: unknown = { a_galaxy_workflow: "true", steps: {} }, exportStatus = 200) {
  const calls: { method: string; path: string; body?: any; params?: any }[] = [];
  const client = mockClient({
    POST: (path, init) => {
      calls.push({ method: "POST", path, body: init?.body });
      return { data: { id: "w1", name: "wf" }, response: { status: 200 } };
    },
    GET: (path, init) => {
      calls.push({ method: "GET", path, params: init?.params });
      return exportStatus === 200
        ? { data: exported, response: { status: 200 } }
        : { error: { err_msg: "boom" }, response: { status: exportStatus } };
    },
  });
  return { client, calls };
}

describe("import_workflow", () => {
  it("is a write op", () => {
    expect(importWorkflowOp.readOnly).toBe(false);
  });

  it("POSTs the definition and fill_defaults to /api/workflows, then reads the export back", async () => {
    const { client, calls } = recordingClient();
    const out = await importWorkflow({ definition: DEFINITION, fillDefaults: true }, ctxWith(client));
    expect(calls).toEqual([
      { method: "POST", path: "/api/workflows", body: { workflow: DEFINITION, fill_defaults: true } },
      {
        method: "GET",
        path: "/api/workflows/{workflow_id}/download",
        params: { path: { workflow_id: "w1" }, query: { style: "ga" } },
      },
    ]);
    expect(out).toEqual({ id: "w1", name: "wf", missing_tools: [] });
  });

  it("accepts the definition as a JSON object string, fill_defaults false by default", async () => {
    const { client, calls } = recordingClient();
    await importWorkflow({ definition: JSON.stringify(DEFINITION) }, ctxWith(client));
    expect(calls[0]!.body).toEqual({ workflow: DEFINITION, fill_defaults: false });
  });

  it("hands Format2 YAML text to Galaxy to convert", async () => {
    const { client, calls } = recordingClient();
    const text = "class: GalaxyWorkflow\ninputs:\n  reads: data\nsteps: {}\n";
    await importWorkflow({ definition: text }, ctxWith(client));
    expect(calls[0]!.body).toEqual({ workflow: { yaml_content: text }, fill_defaults: false });
  });

  it.each(["not json", "[1, 2]", "", {}, '["GalaxyWorkflow"]'])("refuses %j without posting", async (bad) => {
    const { client, calls } = recordingClient();
    await expect(importWorkflow({ definition: bad as any }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyValidationError,
    );
    expect(calls).toEqual([]);
  });

  it("names the tools this Galaxy does not have, subworkflows included, once each", async () => {
    const { client } = recordingClient({
      a_galaxy_workflow: "true",
      steps: {
        "0": { type: "data_input", errors: null },
        "1": { type: "tool", tool_id: "cat1", errors: null },
        "2": { type: "tool", tool_id: "gone/1.0", errors: NOT_INSTALLED },
        "3": {
          type: "subworkflow",
          subworkflow: {
            steps: {
              "0": { type: "tool", tool_id: "nested", errors: NOT_INSTALLED },
              "1": { type: "tool", tool_id: "gone/1.0", errors: NOT_INSTALLED },
            },
          },
        },
      },
    });
    const out = await importWorkflow({ definition: DEFINITION }, ctxWith(client));
    expect(out.missing_tools).toEqual(["gone/1.0", "nested"]);
  });

  it("answers with null, not a failure, when the export cannot be read", async () => {
    const { client } = recordingClient(undefined, 500);
    const out = await importWorkflow({ definition: DEFINITION }, ctxWith(client));
    expect(out).toEqual({ id: "w1", name: "wf", missing_tools: null });
  });

  it("answers with null for an export that is not native .ga", async () => {
    const { client } = recordingClient({ class: "GalaxyWorkflow", steps: {} });
    const out = await importWorkflow({ definition: DEFINITION }, ctxWith(client));
    expect(out.missing_tools).toBeNull();
  });

  it("words its message like the Python server", () => {
    const message = (out: any) =>
      importWorkflowOp.project!(out, { definition: DEFINITION, fillDefaults: false }).message;
    expect(message({ id: "w1", name: "wf", missing_tools: [] })).toBe("Imported workflow 'wf'");
    expect(message({ id: "w1", missing_tools: [] })).toBe("Imported workflow 'None'");
    expect(message({ id: "w1", name: "wf", missing_tools: null })).toBe(
      "Imported workflow 'wf'; its tools could not be checked",
    );
    expect(message({ id: "w1", name: "wf", missing_tools: ["a", "b"] })).toBe(
      "Imported workflow 'wf', but this Galaxy does not have 2 of its tools (a, b); invoking it fails until they are installed",
    );
  });
});
