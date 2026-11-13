import { describe, expect, it } from "vitest";
import type { GalaxyContext } from "../../src/context";
import { DEFAULT_POLL } from "../../src/context";
import { evaluateUserToolPreflight } from "../../src/operations/evaluate-user-tool-preflight";
import { mockClient } from "../util/mock-client";

const JOB_ID = "0123456789abcdef";
const TOOL_UUID = "61d15277-a911-45ef-aa66-5385146578cc";
const MIB = 1024 * 1024;
const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

function clientFor(state = "ok", toolId = "resource_tool") {
  return mockClient({
    GET: (path) => {
      if (path === "/api/unprivileged_tools/{tool_uuid}") {
        return {
          data: {
            tool_id: "resource_tool",
            representation: {
              requirements: [{ type: "resource", ram_min: 512, ram_max: 4096, timelimit: 300 }],
            },
          },
          response: { status: 200 },
        };
      }
      if (path === "/api/jobs/{job_id}") {
        return { data: { id: JOB_ID, tool_id: toolId, state }, response: { status: 200 } };
      }
      return {
        data: [
          { plugin: "cgroup", name: "memory.peak", raw_value: String(800 * MIB) },
          { plugin: "core", name: "runtime_seconds", raw_value: "240" },
        ],
        response: { status: 200 },
      };
    },
  });
}

describe("evaluate_user_tool_preflight", () => {
  it("recommends resources from peak memory and runtime", async () => {
    const result = await evaluateUserToolPreflight(
      { toolUuid: TOOL_UUID, jobId: JOB_ID },
      ctxWith(clientFor()),
    );
    expect(result.recommendation).toEqual({ ram_min: 1200, timelimit: 360 });
    expect(result.warnings.map((warning) => warning.code)).toEqual([
      "ram_min_underestimated",
      "timelimit_underestimated",
    ]);
  });

  it("rejects unfinished jobs", async () => {
    await expect(
      evaluateUserToolPreflight({ toolUuid: TOOL_UUID, jobId: JOB_ID }, ctxWith(clientFor("running"))),
    ).rejects.toThrow("not finished");
  });

  it("rejects a job from another tool", async () => {
    await expect(
      evaluateUserToolPreflight({ toolUuid: TOOL_UUID, jobId: JOB_ID }, ctxWith(clientFor("ok", "other_tool"))),
    ).rejects.toThrow("did not run");
  });
});
