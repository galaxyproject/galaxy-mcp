import { describe, it, expect } from "vitest";
import { runToolOp, runTool } from "../../src/operations/run-tool";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/** What POST /api/tools answers a submission with: queued jobs, in their starting state. */
const SUBMISSION = {
  outputs: [{ id: "d20", hid: 5, state: "new", output_name: "html_file" }],
  output_collections: [],
  jobs: [{ model_class: "Job", id: "j20", state: "new", tool_id: "fastqc", tool_version: "0.74" }],
  implicit_collections: [],
};

describe("run_tool", () => {
  it("has parity name run_tool and a nested-input schema", () => {
    expect(runToolOp.name).toBe("run_tool");
    expect(Object.keys(runToolOp.input).sort()).toEqual(["historyId", "inputs", "toolId", "toolVersion"]);
  });

  it("submits to the classic tools endpoint and hands back what Galaxy said", async () => {
    // Queue-and-return, not queue-and-wait: the jobs come back "new" and nothing here
    // polls them. The other server's run_tool is this same POST and this same record.
    let seen: { path?: string; body?: Record<string, unknown> } = {};
    const client = mockClient({
      POST: (path, init) => {
        seen = { path, body: init.body };
        return { data: SUBMISSION, response: { status: 200 } };
      },
    });
    const out = await runTool(
      { toolId: "fastqc", historyId: "h1", inputs: { input_file: { src: "hda", id: "d1" } } },
      ctxWith(client),
    );
    expect(out).toEqual(SUBMISSION);
    expect(seen.path).toBe("/api/tools");
    expect(seen.body).toEqual({
      history_id: "h1",
      tool_id: "fastqc",
      input_format: "legacy",
      inputs: { input_file: { src: "hda", id: "d1" } },
    });
  });

  it("sends a pinned version in the payload, where Galaxy reads it", async () => {
    // Not the query string: services/tools.py `_create` takes tool_version out of the body,
    // which is the route the other server uses for the same case.
    let body: Record<string, unknown> | undefined;
    const client = mockClient({
      POST: (_path, init) => {
        body = init.body as Record<string, unknown>;
        return { data: SUBMISSION, response: { status: 200 } };
      },
    });
    await runTool(
      { toolId: "fastqc", historyId: "h1", inputs: {}, toolVersion: "0.74+galaxy1" },
      ctxWith(client),
    );
    expect(body?.tool_version).toBe("0.74+galaxy1");
  });

  it("leaves tool_version out entirely when none was asked for", async () => {
    let body: Record<string, unknown> | undefined;
    const client = mockClient({
      POST: (_path, init) => {
        body = init.body as Record<string, unknown>;
        return { data: SUBMISSION, response: { status: 200 } };
      },
    });
    await runTool({ toolId: "fastqc", historyId: "h1", inputs: {} }, ctxWith(client));
    expect("tool_version" in (body ?? {})).toBe(false);
  });

  it("counts the queued jobs in its summary line", () => {
    const message = runToolOp.project?.(SUBMISSION, { toolId: "fastqc", historyId: "h1", inputs: {} })
      ?.message;
    expect(message).toBe("Submitted fastqc to history h1 (1 job(s))");
  });
});
