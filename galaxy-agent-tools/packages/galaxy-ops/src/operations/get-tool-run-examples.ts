import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface ToolRunExamples {
  tool_id: string;
  requested_version: string | null;
  test_cases: unknown[];
}

const input = {
  toolId: z.string().describe("tool id"),
  toolVersion: z.string().optional().describe("specific tool version to fetch test cases for"),
};
type In = { toolId: string; toolVersion?: string };

async function run(i: In, ctx: GalaxyContext): Promise<ToolRunExamples> {
  const testCases = await legacyGet<unknown[]>(ctx, "/api/tools/{tool_id}/test_data", {
    params: {
      path: { tool_id: i.toolId },
      query: i.toolVersion != null ? { tool_version: i.toolVersion } : {},
    },
  });
  return {
    tool_id: i.toolId,
    // Always stated: an absent key reads as "no such field", null as "no version pinned".
    requested_version: i.toolVersion ?? null,
    test_cases: testCases,
  };
}

export const getToolRunExamplesOp: Operation<typeof input, ToolRunExamples> = {
  name: "get_tool_run_examples",
  domain: "tools",
  summary: "Return test-data examples (inputs/outputs) for a Galaxy tool.",
  input,
  run,
  project: (out, i) => ({
    message: `${out.test_cases.length} test case(s) for ${i.toolId}`,
  }),
};

register(getToolRunExamplesOp as AnyOperation);

export const getToolRunExamples = (i: In, ctx: GalaxyContext) => runOperation(getToolRunExamplesOp, i, ctx);
