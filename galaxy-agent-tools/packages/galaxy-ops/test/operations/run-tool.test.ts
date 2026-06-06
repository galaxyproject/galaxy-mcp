import { describe, it, expect } from "vitest";
import { runToolOp, runTool } from "../../src/operations/run-tool";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const fastPoll = { ...DEFAULT_POLL, intervalMs: 0, maxIntervalMs: 0, jitter: 0 };

describe("run_tool", () => {
  it("has parity name run_tool and a nested-input schema", () => {
    expect(runToolOp.name).toBe("run_tool");
    expect(Object.keys(runToolOp.input).sort()).toEqual(["historyId", "inputs", "toolId", "toolVersion"]);
  });

  it("drives a tool to ok via the typed path", async () => {
    let si = 0;
    const client = mockClient({
      POST: () => ({ data: { tool_request_id: "tr1", task_result: {} }, response: { status: 200 } }),
      GET: (path) => {
        if (path.endsWith("/state")) return { data: ["new", "submitted"][Math.min(si++, 1)], response: { status: 200 } };
        if (path === "/api/tool_requests/{id}")
          return { data: { id: "tr1", state: "submitted", jobs: [{ src: "job", id: "j1" }], implicit_collections: [] }, response: { status: 200 } };
        return { data: { id: "j1", state: "ok" }, response: { status: 200 } };
      },
    });
    const ctx: GalaxyContext = { client, poll: fastPoll };
    const run = await runTool(
      { toolId: "fastqc/0.74", historyId: "h1", inputs: { input_file: { src: "hda", id: "d1" } } },
      ctx,
    );
    expect(run.state).toBe("ok");
  });
});
