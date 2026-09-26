import { describe, it, expect } from "vitest";
import { getToolRunExamplesOp, getToolRunExamples } from "../../src/operations/get-tool-run-examples";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const TEST_CASES = [
  { name: "test1", inputs: {}, outputs: {} },
  { name: "test2", inputs: {}, outputs: {} },
];

describe("get_tool_run_examples", () => {
  it("fetches test_data and wraps in result shape", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/tools/{tool_id}/test_data");
        expect(init.params.path.tool_id).toBe("fastqc");
        return { data: TEST_CASES, response: { status: 200 } };
      },
    });
    const out = await getToolRunExamples({ toolId: "fastqc" }, ctxWith(client));
    expect(out.tool_id).toBe("fastqc");
    expect(out.test_cases).toEqual(TEST_CASES);
    // Stated as null rather than left out: an absent key reads as "no such field".
    expect(out.requested_version).toBeNull();
    expect("requested_version" in out).toBe(true);
  });

  it("passes tool_version in query when given", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(init.params.query.tool_version).toBe("0.74");
        return { data: [], response: { status: 200 } };
      },
    });
    const out = await getToolRunExamples({ toolId: "fastqc", toolVersion: "0.74" }, ctxWith(client));
    expect(out.requested_version).toBe("0.74");
  });

  it("project includes test case count and tool id", () => {
    const result = { tool_id: "fastqc", test_cases: [{}, {}] };
    const msg = getToolRunExamplesOp.project!(result as any, { toolId: "fastqc" });
    expect(msg.message).toBe("2 test case(s) for fastqc");
  });
});
